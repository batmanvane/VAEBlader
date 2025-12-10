# Implements part of DOI: 10.1093/jcde/qwaf002 by Kang et al 2025

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import glob
import os
import re
from scipy.interpolate import interp1d

# --- IMPORTS FROM TRAINING SCRIPT ---
# This ensures we use the exact same B-Spline topology (Clustered Knots) and Model Arch
from createPIVAE_refactor_LE_and_Input import AG_VAE, BSplineTransform, SEQ_LEN, NUM_CP, DEVICE

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
VAE_MODEL_PATH = './results/model/ag_vae_cps.pth'
AIRFOIL_DIR = '../VAEBladerData/data/airfoil/naca_gen'
POLAR_ROOT_DIR = '../VAEBladerData/aerodynamic_label/naca_gen'

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30

# Device Strategy:
# 1. Training (Surrogate) -> Fast Device (MPS/CUDA)
# 2. Data Gen (B-Spline Fitting) -> CPU (To avoid MPS linalg.solve crashes)
TRAIN_DEVICE = DEVICE
GEN_DEVICE = torch.device('cpu')

print(f"Training on: {TRAIN_DEVICE} | Data Gen on: {GEN_DEVICE}")


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================
def process_airfoil_to_latents(fpath, vae, bspline_tool, device):
    """
    Loads dat file, fits CPs using the Hybrid Topology, and encodes to z.
    """
    try:
        # Load .dat
        with open(fpath, 'r') as f:
            header = f.readline()
        skip = 1 if "XFOIL" in header or "Calculated" in header else 0
        try:
            raw = np.loadtxt(fpath, skiprows=skip)
        except:
            raw = np.loadtxt(fpath, skiprows=1)

        # Standardize Grid
        theta = np.linspace(0, np.pi, SEQ_LEN)
        x_grid = 0.5 * (1 - np.cos(theta))

        # Split & Interpolate
        le_idx = np.argmin(raw[:, 0])
        ux, uy = raw[:le_idx + 1, 0], raw[:le_idx + 1, 1]
        lx, ly = raw[le_idx:, 0], raw[le_idx:, 1]

        yu = interp1d(np.flip(ux), np.flip(uy), kind='linear', bounds_error=False, fill_value=0.0)(x_grid)
        yl = interp1d(lx, ly, kind='linear', bounds_error=False, fill_value=0.0)(x_grid)

        yt = np.clip(yu - yl, 0, None)
        yc = (yu + yl) / 2.0
        yt[-1] = yc[0] = yc[-1] = 0.0

        # Prepare Tensors on Generation Device (CPU)
        yt_ten = torch.tensor(yt, dtype=torch.float32, device=device).unsqueeze(0)
        yc_ten = torch.tensor(yc, dtype=torch.float32, device=device).unsqueeze(0)

        # --- HYBRID FITTING (Must match Training Logic) ---
        # 1. Thickness: Vertical LE (Round nose, Clustered Knots)
        cps_t = bspline_tool.fit_vertical_le(yt_ten)

        # 2. Camber: Standard Fit (Free inlet angle)
        # We use fit_standard (which might alias fit_vertical_le in code, but conceptually distinct)
        if hasattr(bspline_tool, 'fit_standard'):
            cps_c = bspline_tool.fit_standard(yc_ten)
        else:
            # Fallback if fit_standard isn't defined yet, but logically they use same solver
            cps_c = bspline_tool.fit_vertical_le(yc_ten)

        # Encode
        # Input to encoder is [1, 16] Control Points
        mu_t, _ = vae.enc_thick(cps_t)
        mu_c, _ = vae.enc_camber(cps_c)

        # Concat Latents [1, 12]
        z = torch.cat([mu_t, mu_c], dim=1).detach().cpu().numpy()[0]
        return z

    except Exception as e:
        return None


def parse_xfoil_polar(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    mach, re_num, data_start = None, None, -1

    for i, line in enumerate(lines):
        if 'Mach =' in line:
            m = re.search(r'Mach\s*=\s*([\d\.]+)', line)
            r = re.search(r'Re\s*=\s*([\d\.\s]+e\s*[\d]+)', line)
            if m: mach = float(m.group(1))
            if r: re_num = float(r.group(1).replace(' ', ''))
        if 'alpha    CL' in line:
            data_start = i + 2
            break

    if data_start == -1: return None, None, None

    rows = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 5:
            try:
                rows.append([float(p) for p in parts[:5]])
            except:
                continue

    if not rows: return None, None, None
    return mach, re_num, pd.DataFrame(rows, columns=['alpha', 'CL', 'CD', 'CDp', 'CM'])


# ============================================================================
# 3. SURROGATE MODEL
# ============================================================================
class AeroSurrogate(nn.Module):
    def __init__(self, latent_dim=12):
        super().__init__()
        # Input: Latent(12) + Mach + Re + Alpha = 15
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 256),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3)
        )

    def forward(self, x): return self.net(x)


# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    # 1. Load VAE on CPU (for stable generation)
    print("Loading VAE on CPU...")
    # Initialize using imported class
    vae = AG_VAE(num_cp=NUM_CP, seq_len=SEQ_LEN, device=GEN_DEVICE).to(GEN_DEVICE)
    try:
        # strict=False allows us to load encoders even if decoders mismatch slightly
        # (e.g., if you have dummy decoders in one script and full in another)
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=GEN_DEVICE), strict=False)
        print("VAE weights loaded.")
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return
    vae.eval()

    # Initialize B-Spline tool using imported class (Inherits Clustered Knots)
    bspline_tool = BSplineTransform(NUM_CP, degree=3, num_eval=SEQ_LEN, device=GEN_DEVICE)

    # 2. Build Dataset (On CPU)
    print("Generating Surrogate Dataset...")
    X, y = [], []
    geo_files = sorted(glob.glob(os.path.join(AIRFOIL_DIR, "*.dat")))

    for geo_path in geo_files:
        fid = os.path.splitext(os.path.basename(geo_path))[0]
        polar_dir = os.path.join(POLAR_ROOT_DIR, fid)
        if not os.path.exists(polar_dir): continue

        # Get Latent Z [12]
        z = process_airfoil_to_latents(geo_path, vae, bspline_tool, GEN_DEVICE)
        if z is None: continue

        # Process Polars
        for p_file in glob.glob(os.path.join(polar_dir, "*.txt")):
            mach, re_val, df = parse_xfoil_polar(p_file)
            if df is None: continue

            re_norm = re_val * 1e-6
            for _, row in df.iterrows():
                # Input: [z(12), Mach, Re, Alpha]
                x_row = np.concatenate([z, [mach, re_norm, row['alpha']]])
                y_row = [row['CL'], row['CD'], row['CM']]
                X.append(x_row);
                y.append(y_row)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"Dataset Size: {len(X)}")

    if len(X) == 0:
        print("Error: No valid data pairs found. Check paths.")
        return

    # 3. Train Surrogate (Move to TRAIN_DEVICE for speed)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_sz = int(0.9 * len(X))
    train, val = random_split(ds, [train_sz, len(X) - train_sz])

    train_dl = DataLoader(train, BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val, BATCH_SIZE)

    surrogate = AeroSurrogate(latent_dim=12).to(TRAIN_DEVICE)
    opt = optim.Adam(surrogate.parameters(), lr=LR)
    crit = nn.MSELoss()

    print(f"Training Surrogate on {TRAIN_DEVICE}...")
    for ep in range(EPOCHS):
        surrogate.train()
        tl = 0
        for bx, by in train_dl:
            # Move batch to high-speed device here
            bx, by = bx.to(TRAIN_DEVICE), by.to(TRAIN_DEVICE)
            opt.zero_grad()
            loss = crit(surrogate(bx), by)
            loss.backward()
            opt.step()
            tl += loss.item()

        if ep % 5 == 0:
            surrogate.eval()
            # Validation loop on TRAIN_DEVICE
            vl = sum([crit(surrogate(bx.to(TRAIN_DEVICE)), by.to(TRAIN_DEVICE)).item() for bx, by in val_dl])
            print(f"Ep {ep}: Train {tl / len(train_dl):.5f} | Val {vl / len(val_dl):.5f}")

    torch.save(surrogate.state_dict(), './results/model/aero_surrogate.pth')
    print("Done.")


if __name__ == "__main__":
    main()