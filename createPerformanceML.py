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

# Import VAE definitions
from createPIVAE import AirfoilVAE, SEQ_LEN, NUM_CP, DEVICE

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
VAE_MODEL_PATH = './results/model/airfoil_vae.pth'
AIRFOIL_DIR = '../VAEBladerData/data/airfoil/beziergan_gen'  # Contains 00000.dat
POLAR_ROOT_DIR = '../VAEBladerData/aerodynamic_label/beziergan_gen'  # Contains folders 00000/, 00001/

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30


# ============================================================================
# 2. HELPER: SINGLE AIRFOIL LOADER
# ============================================================================
def process_single_airfoil(fpath, n_points=SEQ_LEN):
    """
    Loads and normalizes a single .dat file exactly like RealAirfoilDataset.
    Returns: torch tensor [1, 2, 200]
    """
    try:
        # Check for header
        with open(fpath, 'r') as f:
            first_line = f.readline()
        skip = 1 if "XFOIL" in first_line or "Calculated" in first_line else 0

        try:
            raw = np.loadtxt(fpath, skiprows=skip)
        except ValueError:
            raw = np.loadtxt(fpath, skiprows=1)  # Try skipping 1 if auto detection failed

        # Standardize grid
        theta = np.linspace(0, np.pi, n_points)
        x_grid = 0.5 * (1 - np.cos(theta))

        # Find Leading Edge
        le_idx = np.argmin(raw[:, 0])
        ux, uy = raw[:le_idx + 1, 0], raw[:le_idx + 1, 1]
        lx, ly = raw[le_idx:, 0], raw[le_idx:, 1]

        # Interpolate
        u_int = interp1d(np.flip(ux), np.flip(uy), kind='linear', bounds_error=False, fill_value=0.0)
        l_int = interp1d(lx, ly, kind='linear', bounds_error=False, fill_value=0.0)
        yu, yl = u_int(x_grid), l_int(x_grid)

        # Convert to Thickness/Camber
        yt = np.clip(yu - yl, 0, None)
        yc = (yu + yl) / 2.0
        yt[-1] = 0.0

        # Stack and add batch dimension
        data = np.stack([yt, yc])  # Shape [2, 200]
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Shape [1, 2, 200]

    except Exception as e:
        # print(f"Error loading {fpath}: {e}")
        return None


# ============================================================================
# 3. HELPER: POLAR PARSER
# ============================================================================
def parse_xfoil_polar(filepath):
    """Parses XFOIL text output for Mach, Re, and polar table."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    mach = None
    re_num = None
    data_start_idx = -1

    for i, line in enumerate(lines):
        # Extract Mach and Re [cite: 2, 6]
        if 'Mach =' in line:
            m_match = re.search(r'Mach\s*=\s*([\d\.]+)', line)
            r_match = re.search(r'Re\s*=\s*([\d\.\s]+e\s*[\d]+)', line)
            if m_match: mach = float(m_match.group(1))
            if r_match: re_num = float(r_match.group(1).replace(' ', ''))

        if 'alpha    CL' in line:
            data_start_idx = i + 2
            break

    if data_start_idx == -1 or data_start_idx >= len(lines):
        return None, None, None

    data_rows = []
    for line in lines[data_start_idx:]:
        parts = line.split()
        if len(parts) >= 5:
            try:
                row = [float(p) for p in parts[:5]]
                data_rows.append(row)
            except ValueError:
                continue

    if not data_rows:
        return None, None, None

    df = pd.DataFrame(data_rows, columns=['alpha', 'CL', 'CD', 'CDp', 'CM'])
    return mach, re_num, df


# ============================================================================
# 4. DATA GENERATION
# ============================================================================
def generate_surrogate_dataset(vae, airfoil_dir, polar_root):
    vae.eval()
    dataset_X = []  # [z(10), Mach, Re, Alpha]
    dataset_y = []  # [CL, CD, CM]

    # Get list of all geometry files (00000.dat, 00001.dat...)
    geo_files = sorted(glob.glob(os.path.join(airfoil_dir, "*.dat")))

    print(f"Found {len(geo_files)} geometry files. Processing...")

    with torch.no_grad():
        for geo_path in geo_files:
            # 1. Extract ID
            filename = os.path.basename(geo_path)
            file_id = os.path.splitext(filename)[0]  # "00000"

            # 2. Check if corresponding polar folder exists
            polar_dir = os.path.join(polar_root, file_id)
            if not os.path.exists(polar_dir):
                # If folder ./aerodynamic_label/beziergan_gen/00000/ missing, skip
                continue

            # 3. Load and Encode Airfoil -> z
            x_in = process_single_airfoil(geo_path)
            if x_in is None: continue

            x_in = x_in.to(DEVICE)
            mu_t, _ = vae.enc_thick(x_in[:, 0])
            mu_c, _ = vae.enc_camber(x_in[:, 1])
            z_vec = torch.cat([mu_t, mu_c], dim=1).cpu().numpy()[0]  # Shape (10,)

            # 4. Iterate through polar files in the specific ID folder
            polar_files = glob.glob(os.path.join(polar_dir, "*.txt"))

            for p_file in polar_files:
                mach, re_val, df = parse_xfoil_polar(p_file)

                # Validation checks
                if df is None or mach is None or re_val is None:
                    continue

                # Normalize Re (Scaling 1e6 -> 1.0)
                re_norm = re_val * 1e-6

                # Create Training Points
                for _, row in df.iterrows():
                    # Input: [z (10), Mach, Re, Alpha]
                    x_row = np.concatenate([
                        z_vec,
                        np.array([mach, re_norm, row['alpha']])
                    ])
                    # Output: [CL, CD, CM]
                    y_row = np.array([row['CL'], row['CD'], row['CM']])

                    dataset_X.append(x_row)
                    dataset_y.append(y_row)

    print(f"Generated {len(dataset_X)} total training points.")
    return np.array(dataset_X, dtype=np.float32), np.array(dataset_y, dtype=np.float32)


# ============================================================================
# 5. SURROGATE MODEL
# ============================================================================
class AeroSurrogate(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        # Input: Latent(10) + Mach + Re + Alpha = 13
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)  # CL, CD, CM
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    # 1. Load VAE
    print(f"Loading VAE from {VAE_MODEL_PATH}...")
    vae = AirfoilVAE(seq_len=SEQ_LEN, num_cp=NUM_CP, device=DEVICE).to(DEVICE)
    if os.path.exists(VAE_MODEL_PATH):
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    else:
        print("Error: VAE model not found. Please run createPIVAE.py first.")
        return

    # 2. Build Dataset
    print("Building Surrogate Dataset (matching Geometry <-> Aerodynamics)...")
    X, y = generate_surrogate_dataset(vae, AIRFOIL_DIR, POLAR_ROOT_DIR)

    if len(X) == 0:
        print("No data found. Check directory paths.")
        return

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Train Surrogate
    model = AeroSurrogate().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Starting training...")
    for ep in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if ep % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_dl:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    pred = model(bx)
                    val_loss += criterion(pred, by).item()
            print(f"Ep {ep}: Train MSE {train_loss / len(train_dl):.5f} | Val MSE {val_loss / len(val_dl):.5f}")

    # 4. Save
    save_path = './results/model/aero_surrogate.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Surrogate model saved to {save_path}")


if __name__ == "__main__":
    main()