# Implements part of DOI: 10.1093/jcde/qwaf002 by Kang et al 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy.interpolate import interp1d

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
CLIP_GRAD = 1.0
LR_PATIENCE = 8
LR_FACTOR = 0.5

# Latent Space Definitions (Matches Paper's Physics-Aware approach)
# Thickness: [Max_T, Pos_Max_T, Radius] + Free vars
LATENT_PHYS_THICK = 3
LATENT_FREE_THICK = 3

# Camber: [Max_C, Pos_Max_C, TE_Dir] + Free vars
LATENT_PHYS_CAMBER = 3
LATENT_FREE_CAMBER = 3

NUM_CP = 16  # Number of Control Points
SEQ_LEN = 200  # Evaluation resolution
DEC_LAYERS = 3
DEC_NODES = 128

DATA_DIR = '../VAEBladerData/data/airfoil/naca_gen'
MAX_FILES = None


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_device()
print(f"Device selected: {DEVICE}")


# ============================================================================
# 2. B-SPLINE & FITTING UTILITIES
# ============================================================================
class BSplineTransform(nn.Module):
    def __init__(self, num_cp, degree=3, num_eval=200, device='cpu'):
        super().__init__()
        self.num_cp = num_cp
        self.degree = degree
        self.device = device

        # 1. Evaluation Grid (Cosine Spacing)
        theta = torch.linspace(0, np.pi, num_eval, device=device)
        self.u = 0.5 * (1 - torch.cos(theta))

        # 2. Clustered Knot Vector (The "Additional Set" for LE)
        # We manually define knots to cluster density at the leading edge.
        # This allocates ~30% of control points to the first 10% of the chord.
        self.basis = self._precompute_basis_clustered(self.u).to(device)

    def _precompute_basis_clustered(self, u):
        p = self.degree
        n = self.num_cp - 1
        m = n + p + 1

        # --- CUSTOM KNOT VECTOR ---
        # We want high density near 0.
        # Internal knots: (num_cp - 1 - degree) = 16 - 1 - 3 = 12 internal knots

        # We generate knots using a power law (x^2) to cluster them near 0
        internal_knots = torch.linspace(0, 1, m - 2 * p, device=u.device)
        internal_knots = internal_knots ** 2  # <--- CLUSTERING MAGIC

        kv = torch.zeros(m + 1, device=u.device)
        kv[p + 1: m - p + 1] = internal_knots  # Fill internal
        kv[m - p + 1:] = 1.0  # Clamp end

        # Cox-de Boor Recursion (Standard)
        N = torch.zeros(u.shape[0], m, device=u.device)
        for i in range(m):
            mask = (u >= kv[i]) & (u < kv[i + 1])
            if i == m - 1: mask = mask | (u == kv[i + 1])
            N[:, i] = mask.float()

        for d in range(1, p + 1):
            N_new = torch.zeros(u.shape[0], m - d, device=u.device)
            for i in range(m - d - 1):
                d1 = kv[i + d] - kv[i]
                d2 = kv[i + d + 1] - kv[i + 1]
                t1 = ((u - kv[i]) / d1) * N[:, i] if d1 > 1e-6 else 0.0
                t2 = ((kv[i + d + 1] - u) / d2) * N[:, i + 1] if d2 > 1e-6 else 0.0
                N_new[:, i] = t1 + t2
            N = N_new

        return N[:, :self.num_cp]

    def forward(self, cp_y):
        return torch.matmul(cp_y, self.basis.T)

    def fit_vertical_le(self, target_y, lambda_smooth=0.005):
        # Basis matrix for indices 1..N (Skip index 0 which is fixed at 0)
        M_red = self.basis[:, 1:]

        gram = torch.matmul(M_red.T, M_red)

        # Regularization matrix must match dimensions
        # num_free = num_cp - 1
        n_free = self.num_cp - 1
        D2 = torch.zeros(n_free - 2, n_free, device=self.device)
        for i in range(n_free - 2):
            D2[i, i] = 1;
            D2[i, i + 1] = -2;
            D2[i, i + 2] = 1

        reg = lambda_smooth * torch.matmul(D2.T, D2)

        A = gram + reg + 1e-6 * torch.eye(n_free, device=self.device)
        b = torch.matmul(M_red.T, target_y.T)
        cp_free = torch.linalg.solve(A, b).T

        zeros = torch.zeros(cp_free.shape[0], 1, device=self.device)
        return torch.cat([zeros, cp_free], dim=1)

    def fit_standard(self, target_y, lambda_smooth=0.005):
        return self.fit_vertical_le(target_y, lambda_smooth)


# ============================================================================
# 3. NETWORK MODULES
# ============================================================================
class MLPEncoder(nn.Module):
    """ Encodes CONTROL POINTS (not coordinates) """

    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_lv(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, layers=3, nodes=128):
        super().__init__()
        modules = []
        in_dim = latent_dim
        for _ in range(layers):
            modules.append(nn.Linear(in_dim, nodes))
            modules.append(nn.GELU())
            in_dim = nodes
        modules.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, z):
        return self.net(z)


class AG_VAE(nn.Module):
    def __init__(self, num_cp=16, seq_len=200, device='cpu'):
        super().__init__()
        self.device = device
        self.num_cp = num_cp

        self.dim_t = LATENT_PHYS_THICK + LATENT_FREE_THICK
        self.dim_c = LATENT_PHYS_CAMBER + LATENT_FREE_CAMBER

        self.bspline = BSplineTransform(num_cp, degree=3, num_eval=seq_len, device=device)

        # Encoders: Input is Control Points (size=num_cp)
        self.enc_thick = MLPEncoder(num_cp, self.dim_t)
        self.enc_camber = MLPEncoder(num_cp, self.dim_c)

        # Decoders: Predict CP 1..N (Size = num_cp - 1)
        self.dec_thick = MLPDecoder(self.dim_t, num_cp - 1, DEC_LAYERS, DEC_NODES)
        self.dec_camber = MLPDecoder(self.dim_c, num_cp - 1, DEC_LAYERS, DEC_NODES)

        self.log_prior = nn.Parameter(torch.tensor([-2.0], device=device))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_t, z_c):
        cp_t_raw = self.dec_thick(z_t)
        cp_c_free = self.dec_camber(z_c)

        # --- REFINED CONSTRAINTS FOR CLUSTERED KNOTS ---
        # Due to clustering, CP indices 0, 1, 2 are all "Nose Points".
        # We apply the bias to the first TWO free points (Index 0 and 1 of raw, which are CP_1 and CP_2)

        # 1. Nose Region (CP_1, CP_2): Ensure minimum thickness
        cp_t_nose = F.softplus(cp_t_raw[:, 0:2]) + 0.005

        # 2. Body Region (CP_3...): Standard positivity
        cp_t_body = F.softplus(cp_t_raw[:, 2:]) + 1e-4

        # Recombine
        cp_t_free = torch.cat([cp_t_nose, cp_t_body], dim=1)

        # ... rest same as before ...
        zeros = torch.zeros(cp_t_free.shape[0], 1, device=self.device)
        cp_t = torch.cat([zeros, cp_t_free], dim=1)
        cp_c = torch.cat([zeros, cp_c_free], dim=1)

        t_out = self.bspline(cp_t)
        c_out = self.bspline(cp_c)

        return t_out, c_out, cp_t, cp_c

    def forward(self, cp_input):
        cp_t_in = cp_input[:, 0]
        cp_c_in = cp_input[:, 1]

        # 1. Encode CPs
        mu_t, lv_t = self.enc_thick(cp_t_in)
        mu_c, lv_c = self.enc_camber(cp_c_in)
        z_t, z_c = self.reparameterize(mu_t, lv_t), self.reparameterize(mu_c, lv_c)

        # 2. Decode using the exposed method
        t_out, c_out, cp_t, cp_c = self.decode(z_t, z_c)

        return torch.stack([t_out, c_out], 1), torch.stack([cp_t, cp_c], 1), (mu_t, lv_t), (mu_c, lv_c)

    def calculate_physics(self, t_dist, c_dist):
        """ Helper to extract physics from GENERATED curves for Physics Loss """
        # Max Thickness & Pos
        max_t, idx_t = torch.max(t_dist, dim=1)
        pos_max_t = self.bspline.u[idx_t]

        # Radius Approx (using point near LE)
        # r ~ y^2 / 2x
        x_le = self.bspline.u[2]
        y_le = t_dist[:, 2]
        r_le = (y_le ** 2) / (8 * x_le + 1e-6)

        # Max Camber & Pos
        max_c, idx_c = torch.max(torch.abs(c_dist), dim=1)
        pos_max_c = self.bspline.u[idx_c]

        # TE Direction (Slope at trailing edge)
        te_dir = c_dist[:, -1] - c_dist[:, -2]

        return torch.stack([max_t, pos_max_t, r_le], 1), torch.stack([max_c, pos_max_c, te_dir], 1)
# ============================================================================
# 4. DATASET (FITTING CPS WITH CONSTRAINTS)
# ============================================================================
class RealAirfoilDataset(Dataset):
    def __init__(self, data_dir, num_cp=16, seq_len=200, max_files=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
        if max_files: self.files = self.files[:max_files]

        # Helper for fitting
        self.bspline_tool = BSplineTransform(num_cp, num_eval=seq_len, device='cpu')

        self.cps, self.features, self.coords = self._load_and_process()

    def _load_and_process(self):
        cp_list, feat_list, coord_list = [], [], []
        theta = np.linspace(0, np.pi, 200)
        x_grid = 0.5 * (1 - np.cos(theta))

        print(f"Processing {len(self.files)} files...")

        for fpath in self.files:
            try:
                # 1. Load & Interpolate
                raw = np.loadtxt(fpath, skiprows=1)
                le_idx = np.argmin(raw[:, 0])
                ux, uy = raw[:le_idx + 1, 0], raw[:le_idx + 1, 1]
                lx, ly = raw[le_idx:, 0], raw[le_idx:, 1]

                yu = interp1d(np.flip(ux), np.flip(uy), kind='linear', fill_value=0.0, bounds_error=False)(x_grid)
                yl = interp1d(lx, ly, kind='linear', fill_value=0.0, bounds_error=False)(x_grid)

                yt = np.clip(yu - yl, 0, None)
                yc = (yu + yl) / 2.0
                yt[-1], yc[0], yc[-1] = 0.0, 0.0, 0.0

                # 2. Physics Extraction
                max_t = np.max(yt)
                pos_max_t = x_grid[np.argmax(yt)]

                # Robust Radius Check
                mask = (x_grid > 0.002) & (x_grid < 0.02)
                if np.sum(mask) > 1:
                    r_est = np.median((yt[mask] ** 2) / (8 * x_grid[mask]))
                else:
                    r_est = 0.0

                max_c = np.max(yc)
                pos_max_c = x_grid[np.argmax(np.abs(yc))]
                te_dir = yc[-1] - yc[-5]

                # 3. Fit CPs (Enforcing Vertical LE)
                yt_ten = torch.tensor(yt, dtype=torch.float32).unsqueeze(0)
                yc_ten = torch.tensor(yc, dtype=torch.float32).unsqueeze(0)

                # 1. Thickness: Vertical LE Fit (Enforces Round Nose)
                cps_t = self.bspline_tool.fit_vertical_le(yt_ten)

                # 2. Camber: Standard Fit (Enforces Free Inlet Angle)
                # This ensures we do NOT force the camber line to be vertical or horizontal at the start.
                cps_c = self.bspline_tool.fit_standard(yc_ten)

                cp_list.append(torch.stack([cps_t.squeeze(), cps_c.squeeze()]).numpy())
                coord_list.append(torch.stack([yt_ten.squeeze(), yc_ten.squeeze()]).numpy())
                feat_list.append([max_t, pos_max_t, r_est, max_c, pos_max_c, te_dir])

            except Exception:
                continue

        feats = torch.tensor(np.array(feat_list), dtype=torch.float32)
        self.feat_mean = feats.mean(0)
        self.feat_std = feats.std(0)
        feats_norm = (feats - self.feat_mean) / (self.feat_std + 1e-6)

        return (torch.tensor(np.array(cp_list), dtype=torch.float32),
                feats_norm,
                torch.tensor(np.array(coord_list), dtype=torch.float32))

    def __len__(self):
        return len(self.cps)

    def __getitem__(self, i):
        return self.cps[i], self.features[i], self.coords[i]


# ============================================================================
# 5. LOSS FUNCTION
# ============================================================================
def loss_function(rec_x, true_x, rec_cp, true_cp, mt, lt, mc, lc, feat, p_lv, ep, model):
    # 1. Reconstruction (Weighted Kulfan Tolerance)
    weights = torch.ones_like(true_x)
    weights[:, :, :40] = 20.0  # High priority on LE
    mse_x = torch.sum(weights * (rec_x - true_x) ** 2)

    # 2. Control Point Guidance
    mse_cp = F.mse_loss(rec_cp, true_cp, reduction='sum')

    # 3. Physics Loss (Latent Consistency)
    p_t_rec, p_c_rec = model.calculate_physics(rec_x[:, 0], rec_x[:, 1])
    kl_phys_t = 0.5 * torch.sum((mt[:, :3] - feat[:, :3]) ** 2)
    kl_phys_c = 0.5 * torch.sum((mc[:, :3] - feat[:, 3:]) ** 2)

    # 4. KL Divergence (Latent Regularization)
    kl_div = 0.0
    for m, l in [(mt, lt), (mc, lc)]:
        kl_div += -0.5 * torch.sum(1 + l - m.pow(2) - l.exp())

    # 5. "Body-Only" Smoothness
    # We ignore the first 3 indices (Nose region) to allow high curvature there.
    # We only penalize "wiggles" in the main body and trailing edge.
    # indices: [start_index:] -> [3:] means we skip P0, P1, P2

    # Thickness Smoothness
    cp_t_body = rec_cp[:, 0, 3:]
    diff_t = cp_t_body[:, 2:] - 2 * cp_t_body[:, 1:-1] + cp_t_body[:, :-2]

    # Camber Smoothness
    cp_c_body = rec_cp[:, 1, 3:]
    diff_c = cp_c_body[:, 2:] - 2 * cp_c_body[:, 1:-1] + cp_c_body[:, :-2]

    reg_smooth = torch.sum(diff_t ** 2) + torch.sum(diff_c ** 2)

    # --- ANNEALING SCHEDULES ---
    # Beta (KL): Ramp up to 0.5 over 20 epochs
    beta = min(0.5, ep / 20.0)

    # Alpha (Physics): Ramp up to 10.0 over 30 epochs
    # This lets the model learn "How to draw an airfoil" first (0-10 epochs),
    # and then learns "What the variables mean" (10-30 epochs).
    alpha_phys = min(10.0, ep / 3.0)

    loss = mse_x + 10.0 * mse_cp + beta * kl_div + alpha_phys * (kl_phys_t + kl_phys_c) + 100.0 * reg_smooth

    return {
        "loss": loss,
        "rec": mse_x.item(),
        "cp": mse_cp.item(),
        "kl": kl_div.item(),
        "phys": (kl_phys_t + kl_phys_c).item(),
        "smooth": reg_smooth.item()
    }
    return {
        "loss": loss,
        "rec": mse_x.item(),
        "cp": mse_cp.item(),
        "kl": kl_div.item(),
        "phys": (kl_phys_t + kl_phys_c).item(),
        "smooth": reg_smooth.item()
    }


# ============================================================================
# 6. VISUALIZATION UTILS
# ============================================================================
def visualize_original_data(dataloader):
    # Plot raw coordinates from dataset to verify loading
    _, _, coords = next(iter(dataloader))
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, SEQ_LEN)))

    plt.figure(figsize=(10, 5))
    for i in range(min(50, len(coords))):
        c = coords[i]
        yu = c[1] + c[0] / 2
        yl = c[1] - c[0] / 2
        plt.plot(x_val, yu, 'k-', alpha=0.1)
        plt.plot(x_val, yl, 'k-', alpha=0.1)
    plt.title("Original Dataset Samples")
    plt.axis('equal')
    plt.savefig('./results/plots/original_data.png')
    plt.close()


def plot_correlation_matrix(model, dataloader, device):
    model.eval()
    latents, targets = [], []
    with torch.no_grad():
        for cp, f, _ in dataloader:
            cp = cp.to(device)
            mt, _ = model.enc_thick(cp[:, 0])
            mc, _ = model.enc_camber(cp[:, 1])
            z = torch.cat([mt, mc], dim=1).cpu().numpy()
            latents.append(z)
            targets.append(f.numpy())

    Z = np.concatenate(latents, axis=0)
    Y = np.concatenate(targets, axis=0)

    corr = np.zeros((Z.shape[1], Y.shape[1]))
    for i in range(Z.shape[1]):
        for j in range(Y.shape[1]):
            corr[i, j] = np.corrcoef(Z[:, i], Y[:, j])[0, 1]

    labels_phys = ['Max_T', 'Pos_T', 'Rad', 'Max_C', 'Pos_C', 'TE_Dir']
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.abs(corr), annot=True, fmt=".2f", cmap='viridis', xticklabels=labels_phys)
    plt.title("Latent-Physics Correlation")
    plt.savefig('./results/plots/correlation.png')
    plt.close()


def visualize_reconstruction(model, dataloader, device):
    model.eval()
    cp, _, coord = next(iter(dataloader))
    cp = cp.to(device)
    with torch.no_grad():
        rec_coord, _, _, _ = model(cp)

    rec = rec_coord.cpu().numpy()
    gt = coord.numpy()
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, SEQ_LEN)))

    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # GT
        plt.plot(x, gt[i, 1] + gt[i, 0] / 2, 'k-', label='GT')
        plt.plot(x, gt[i, 1] - gt[i, 0] / 2, 'k-')
        # Rec
        plt.plot(x, rec[i, 1] + rec[i, 0] / 2, 'r--', label='Rec')
        plt.plot(x, rec[i, 1] - rec[i, 0] / 2, 'r--')
        plt.axis('equal')
        if i == 0: plt.legend()
    plt.savefig('./results/plots/reconstruction.png')
    plt.close()


def plot_latent_traversals(model, dataloader, device):
    model.eval()
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, SEQ_LEN)))
    cp, _, _ = next(iter(dataloader))
    cp = cp[0:1].to(device)  # Reference

    with torch.no_grad():
        mu_t, _ = model.enc_thick(cp[:, 0])
        mu_c, _ = model.enc_camber(cp[:, 1])

        fig, axes = plt.subplots(6, 9, figsize=(15, 10))
        vals = np.linspace(-3, 3, 9)
        labels = ['Max_T', 'Pos_T', 'LE_Rad', 'Max_C', 'Pos_C', 'TE_Dir']

        for r in range(6):  # 6 Physical vars
            for c_idx, val in enumerate(vals):
                zt, zc = mu_t.clone(), mu_c.clone()
                if r < 3:
                    zt[0, r] = val
                else:
                    zc[0, r - 3] = val

                # Correctly call the decode method
                to, co, _, _ = model.decode(zt, zc)

                to = to.cpu().numpy()[0]
                co = co.cpu().numpy()[0]

                yu = co + to / 2
                yl = co - to / 2

                ax = axes[r, c_idx]
                ax.plot(x, yu, 'k')
                ax.plot(x, yl, 'k')
                ax.axis('off')
                ax.set_ylim(-0.3, 0.3)
                if c_idx == 4: ax.set_title(labels[r])
    plt.tight_layout()
    plt.savefig('./results/plots/traversals.png')
    plt.close()
    print("Saved latent traversals.")



def encode_airfoil_data(vae, bspline_tool, yt, yc, device):
    """
    Centralized logic to convert raw thickness/camber to latents.
    Ensures consistent fitting topology.
    """
    yt_ten = torch.tensor(yt, dtype=torch.float32, device=device).unsqueeze(0)
    yc_ten = torch.tensor(yc, dtype=torch.float32, device=device).unsqueeze(0)

    # Hybrid Fitting Topology (The Single Source of Truth)
    cps_t = bspline_tool.fit_vertical_le(yt_ten)
    cps_c = bspline_tool.fit_standard(yc_ten)

    # Encode
    with torch.no_grad():
        mu_t, _ = vae.enc_thick(cps_t)
        mu_c, _ = vae.enc_camber(cps_c)

    # Return 1D latent vector [12]
    return torch.cat([mu_t, mu_c], dim=1).cpu().numpy()[0]

# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    if not os.path.exists('./results/plots'): os.makedirs('./results/plots')
    if not os.path.exists('./results/model'): os.makedirs('./results/model')

    # 1. Load Data
    ds = RealAirfoilDataset(DATA_DIR, num_cp=NUM_CP, seq_len=SEQ_LEN, max_files=MAX_FILES)
    train_sz = int(0.9 * len(ds))
    train_ds, val_ds = random_split(ds, [train_sz, len(ds) - train_sz])

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    # Plot Original Data Distribution
    visualize_original_data(val_dl)

    # 2. Model
    model = AG_VAE(NUM_CP, SEQ_LEN, DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, factor=LR_FACTOR)

    print("Starting Training...")
    for ep in range(EPOCHS):
        model.train()
        ep_logs = {"loss": 0, "rec": 0, "cp": 0, "kl": 0, "phys": 0, "smooth": 0}

        for cp, f, x in train_dl:
            cp, f, x = cp.to(DEVICE), f.to(DEVICE), x.to(DEVICE)
            optimizer.zero_grad()

            rx, rcp, (mt, lt), (mc, lc) = model(cp)

            logs = loss_function(rx, x, rcp, cp, mt, lt, mc, lc, f, model.log_prior, ep, model)
            logs["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            for k, v in logs.items(): ep_logs[k] += v if isinstance(v, float) else v.item()

        # Avg Logs
        n = len(train_dl)
        print(f"Ep {ep:03d} | Loss: {ep_logs['loss'] / n:.1f} | Rec: {ep_logs['rec'] / n:.1f} | "
              f"Phys: {ep_logs['phys'] / n:.1f} | Smooth: {ep_logs['smooth'] / n:.1f}")

        scheduler.step(ep_logs['loss'] / n)

    # 3. Finalize
    torch.save(model.state_dict(), './results/model/ag_vae_cps.pth')
    plot_correlation_matrix(model, val_dl, DEVICE)
    visualize_reconstruction(model, val_dl, DEVICE)
    plot_latent_traversals(model, val_dl, DEVICE)
    print("Training Complete.")


if __name__ == "__main__":
    main()