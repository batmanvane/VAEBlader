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
# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
CLIP_GRAD = 1.0
LR_PATIENCE = 10
LR_FACTOR = 0.5

# Architecture
# Thickness Branch: Max Thickness, Pos Max Thickness, LE Radius (3 vars)
LATENT_PHYS_THICK = 3
LATENT_FREE_THICK = 3  # 1 free per aligned variable

# Camber Branch: Max Camber, Pos Max Camber (2 vars)
LATENT_PHYS_CAMBER = 2
LATENT_FREE_CAMBER = 2  # 1 free per aligned variable

NUM_CP = 15 # Number of Control Points for B-Spline
SEQ_LEN = 200 # Number of points in airfoil distributions
ENC_FILTERS = 64 # Base number of filters in encoder
ENC_KERNEL = 3 # Kernel size in encoder
DEC_LAYERS = 2 # Number of layers in decoder
DEC_NODES = 64 #

# Data
DATA_DIR = './data/airfoil/beziergan_gen'
MAX_FILES = None # Set to None to use all files


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
# 2. MODULES
# ============================================================================
class BSplineLayer(nn.Module):
    """
    Differentiable B-Spline Layer with Cosine Spacing.
    Maps Control Points -> Curve Coordinates.
    """

    def __init__(self, num_control_points, degree=3, num_eval_points=100, device='cpu'):
        super().__init__()
        self.num_control_points = num_control_points
        self.degree = degree

        theta = torch.linspace(0, np.pi, num_eval_points, device=device)
        u = 0.5 * (1 - torch.cos(theta))

        self.basis_matrix = self._precompute_basis(u).to(device)

    def _precompute_basis(self, u):
        n, p = self.num_control_points - 1, self.degree
        m = n + p + 1
        kv = torch.zeros(m + 1, device=u.device)

        start, end = p + 1, m - p
        if end > start:
            kv[start:end] = torch.linspace(0, 1, end - start + 2, device=u.device)[1:-1]
        kv[end:] = 1.0

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

        return N[:, :self.num_control_points]

    def forward(self, cp):
        return torch.matmul(cp, self.basis_matrix.T)


class EncoderBlock(nn.Module):
    """1D CNN Encoder"""

    def __init__(self, input_len, latent_dim, filters=64, kernel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, filters, kernel, padding=kernel // 2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(filters, filters * 2, kernel, padding=kernel // 2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(filters * 2, filters * 4, kernel, padding=kernel // 2), nn.GELU(), nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat_size = self.net(dummy).shape[1]

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_lv = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        h = self.net(x.unsqueeze(1))
        return self.fc_mu(h), self.fc_lv(h)


class DecoderBlock(nn.Module):
    """FCN Decoder"""

    def __init__(self, latent_dim, output_dim, layers=2, nodes=32):
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


# ============================================================================
# 3. VAE MODEL
# ============================================================================
class AirfoilVAE(nn.Module):
    def __init__(self, seq_len=200, num_cp=15, device='cpu'):
        super().__init__()
        self.device = device
        self.num_cp = num_cp

        # Dimensions
        self.dim_t = LATENT_PHYS_THICK + LATENT_FREE_THICK
        self.dim_c = LATENT_PHYS_CAMBER + LATENT_FREE_CAMBER

        # -- Encoders --
        self.enc_thick = EncoderBlock(seq_len, self.dim_t, ENC_FILTERS, ENC_KERNEL)
        self.enc_camber = EncoderBlock(seq_len, self.dim_c, ENC_FILTERS, ENC_KERNEL)

        # -- Decoders --
        self.dec_thick = DecoderBlock(self.dim_t, num_cp - 2, DEC_LAYERS, DEC_NODES)
        self.dec_camber = DecoderBlock(self.dim_c, num_cp, DEC_LAYERS, DEC_NODES)

        # -- B-Spline --
        self.bspline = BSplineLayer(num_cp, degree=3, num_eval_points=seq_len, device=device)

        # -- Physics Prior --
        self.log_prior = nn.Parameter(torch.tensor([-2.0], device=device))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        t_in, c_in = x[:, 0], x[:, 1]

        # 1. Encode
        mu_t, lv_t = self.enc_thick(t_in)
        mu_c, lv_c = self.enc_camber(c_in)

        z_t = self.reparameterize(mu_t, lv_t)
        z_c = self.reparameterize(mu_c, lv_c)

        # 2. Decode Control Points
        cp_t_raw = self.dec_thick(z_t)
        cp_c = self.dec_camber(z_c)

        # 3. Geometric Constraints
        cp_t_pos = F.softplus(cp_t_raw)

        zeros = torch.zeros(cp_t_pos.shape[0], 1, device=self.device)
        cp_t = torch.cat([zeros, cp_t_pos, zeros], dim=1)

        # 4. Curve Generation
        t_out = self.bspline(cp_t)
        c_out = self.bspline(cp_c)

        return torch.stack([t_out, c_out], 1), (mu_t, lv_t), (mu_c, lv_c), (cp_t, cp_c)


# ============================================================================
# 4. DATASET
# ============================================================================
class RealAirfoilDataset(Dataset):
    def __init__(self, data_dir, n_points=200, max_files=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
        if max_files: self.files = self.files[:max_files]
        self.n_points = n_points
        self.data, self.features = self._load_data()

    def _load_data(self):
        data_list, feat_list = [], []
        theta = np.linspace(0, np.pi, self.n_points)
        x_grid = 0.5 * (1 - np.cos(theta))

        for fpath in self.files:
            try:
                raw = np.loadtxt(fpath, skiprows=1) if self._has_header(fpath) else np.loadtxt(fpath)

                # Find Leading Edge
                le_idx = np.argmin(raw[:, 0])
                ux, uy = raw[:le_idx + 1, 0], raw[:le_idx + 1, 1]
                lx, ly = raw[le_idx:, 0], raw[le_idx:, 1]

                # Interpolate
                u_int = interp1d(np.flip(ux), np.flip(uy), kind='linear', bounds_error=False, fill_value=0.0)
                l_int = interp1d(lx, ly, kind='linear', bounds_error=False, fill_value=0.0)
                yu, yl = u_int(x_grid), l_int(x_grid)

                # Distributions
                yt = np.clip(yu - yl, 0, None)
                yc = (yu + yl) / 2.0
                yt[-1] = 0.0

                # --- Physics Extraction ---

                # 1. Max Thickness & Position
                max_t = np.max(yt)
                pos_max_t = x_grid[np.argmax(yt)]

                # 2. Leading Edge Radius
                # Approximation: Fit y = a * sqrt(x) for x near 0. R = a^2 / 2
                mask_le = (x_grid > 0.001) & (x_grid < 0.05)
                if np.sum(mask_le) > 3:
                    x_fit = x_grid[mask_le]
                    y_fit = yt[mask_le]
                    # y ~ 2 * sqrt(2 * R * x) => yt^2 ~ 8 * R * x => R ~ yt^2 / (8x)
                    r_estimates = (y_fit ** 2) / (8 * x_fit)
                    radius_le = np.mean(r_estimates)
                else:
                    radius_le = 0.0

                # 3. Max Camber & Position
                max_c = np.max(yc)
                pos_max_c = x_grid[np.argmax(np.abs(yc))]

                data_list.append(np.stack([yt, yc]))
                # Feature Vector: [T_max, X_Tmax, R_LE, C_max, X_Cmax]
                feat_list.append([max_t, pos_max_t, radius_le, max_c, pos_max_c])

            except Exception:
                pass

        if not data_list: raise RuntimeError("No valid data found.")

        feats = torch.tensor(np.array(feat_list), dtype=torch.float32)
        # Normalize features
        self.feat_mean = feats.mean(0)
        self.feat_std = feats.std(0)
        feats_norm = (feats - self.feat_mean) / (self.feat_std + 1e-6)

        print(f"Loaded {len(data_list)} airfoils.")
        return torch.tensor(np.array(data_list), dtype=torch.float32), feats_norm

    def _has_header(self, fpath):
        try:
            np.loadtxt(fpath)
            return False
        except ValueError:
            return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.features[i]


# ============================================================================
# 5. LOSS & TRAINING UTILS
# ============================================================================
def gaussian_kl(mu, lv, p_mu, p_lv):
    return 0.5 * torch.sum(p_lv - lv - 1 + (lv.exp() + (mu - p_mu) ** 2) / p_lv.exp(), 1)


def loss_function(rec, x, mt, lt, mc, lc, ct, cc, feat, p_lv, ep):
    """
    mt, lt: Thickness latent params (Dim = LATENT_PHYS_THICK + LATENT_FREE_THICK)
    mc, lc: Camber latent params (Dim = LATENT_PHYS_CAMBER + LATENT_FREE_CAMBER)
    feat: [T_max, X_Tmax, R_LE, C_max, X_Cmax]
    """
    # 1. Reconstruction
    mse = F.mse_loss(rec, x, reduction='sum')

    # 2. Physics KL (Aligned Variables)
    # Thickness Physics (Indices 0, 1, 2) <-> Feat (0, 1, 2)
    kl_tp = gaussian_kl(mt[:, :LATENT_PHYS_THICK], lt[:, :LATENT_PHYS_THICK],
                        feat[:, :LATENT_PHYS_THICK], p_lv)

    # Camber Physics (Indices 0, 1) <-> Feat (3, 4)
    kl_cp = gaussian_kl(mc[:, :LATENT_PHYS_CAMBER], lc[:, :LATENT_PHYS_CAMBER],
                        feat[:, LATENT_PHYS_THICK:], p_lv)

    # 3. Free KL (Latent -> N(0,1))
    # Thickness Free
    kl_tf = -0.5 * torch.sum(1 + lt[:, LATENT_PHYS_THICK:] -
                             mt[:, LATENT_PHYS_THICK:].pow(2) -
                             lt[:, LATENT_PHYS_THICK:].exp(), 1)

    # Camber Free
    kl_cf = -0.5 * torch.sum(1 + lc[:, LATENT_PHYS_CAMBER:] -
                             mc[:, LATENT_PHYS_CAMBER:].pow(2) -
                             lc[:, LATENT_PHYS_CAMBER:].exp(), 1)

    # 4. Smoothness Reg
    reg_t = torch.mean(torch.diff(ct, n=2) ** 2)
    reg_c = torch.mean(torch.diff(cc, n=2) ** 2)

    BETAMAX = 4.0
    BETACURRENT = min(BETAMAX, (ep+1) / 10) # Linear ramp-up over first 10 epochs
    REG_WEIGHT = 100.0

    total_kl = torch.sum(kl_tp + kl_cp + kl_tf + kl_cf)
    return mse + BETACURRENT * total_kl + REG_WEIGHT * (reg_t + reg_c)


def plot_correlation_matrix(model, dataloader, device):
    model.eval()
    latents_t, latents_c, targets = [], [], []

    with torch.no_grad():
        for x, f in dataloader:
            x = x.to(device)
            mt, _ = model.enc_thick(x[:, 0])
            mc, _ = model.enc_camber(x[:, 1])

            latents_t.append(mt.cpu().numpy())
            latents_c.append(mc.cpu().numpy())
            targets.append(f.numpy())

    Z_t = np.concatenate(latents_t, axis=0)
    Z_c = np.concatenate(latents_c, axis=0)
    Y = np.concatenate(targets, axis=0)

    # Combine latents for visualization
    # Cols: [T_Phys(3), T_Free(6), C_Phys(2), C_Free(4)]
    Z = np.concatenate([Z_t, Z_c], axis=1)

    n_latent = Z.shape[1]
    n_phys = Y.shape[1]  # 5 physical features
    corrs = np.zeros((n_latent, n_phys))

    for i in range(n_latent):
        for j in range(n_phys):
            corrs[i, j] = np.corrcoef(Z[:, i], Y[:, j])[0, 1]

    # Labels
    ytick = ([f'T_Phys_{i}' for i in range(LATENT_PHYS_THICK)] +
             [f'T_Free_{i}' for i in range(LATENT_FREE_THICK)] +
             [f'C_Phys_{i}' for i in range(LATENT_PHYS_CAMBER)] +
             [f'C_Free_{i}' for i in range(LATENT_FREE_CAMBER)])

    xtick = ['T_max', 'X_Tmax', 'R_LE', 'C_max', 'X_Cmax']

    plt.figure(figsize=(10, 12))
    sns.heatmap(np.abs(corrs), annot=True, fmt=".2f", cmap='viridis',
                xticklabels=xtick, yticklabels=ytick)
    plt.title("Latent-Physics Correlation (|Pearson|)")
    plt.tight_layout()
    plt.savefig('./results/plots/correlation_matrix.png')
    print("Saved ./results/plots/correlation_matrix.png")


#plot all validation airfoils on top of each other
def visualize_airfoils(dataloader):
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, SEQ_LEN)))
    plt.figure(figsize=(8, 6))
    #restrict to first 500 airfoils for visibility
    #plot upper and lower surfaces, no fills between
    for i, (x, _) in enumerate(dataloader):
        if i * BATCH_SIZE >= 500:
            break
        x = x.numpy()
        for j in range(x.shape[0]):
            yu = x[j, 1] + x[j, 0] / 2
            yl = x[j, 1] - x[j, 0] / 2
            plt.plot(x_val, yu, 'b-', alpha=0.1)
            plt.plot(x_val, yl, 'b-', alpha=0.1)
    plt.title("Airfoil Shapes from Dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.savefig('./results/plots/airfoil_shapes.png')
    print("Saved ./results/plots/airfoil_shapes.png")




def visualize_reconstruction(model, dataloader, device):
    model.eval()
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, SEQ_LEN)))
    x, _ = next(iter(dataloader))
    x = x.to(device)
    with torch.no_grad():
        rec, _, _, _ = model(x)

    x = x.cpu().numpy();
    rec = rec.cpu().numpy()
    plt.figure(figsize=(12, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        yu_gt = x[i, 1] + x[i, 0] / 2
        yl_gt = x[i, 1] - x[i, 0] / 2
        yu_rc = rec[i, 1] + rec[i, 0] / 2
        yl_rc = rec[i, 1] - rec[i, 0] / 2
        plt.plot(x_val, yu_gt, 'k-', label='GT')
        plt.plot(x_val, yl_gt, 'k-')
        plt.plot(x_val, yu_rc, 'r--', label='Rec')
        plt.plot(x_val, yl_rc, 'r--')
        plt.axis('equal');
        plt.title(f"Sample {i}")
        if i == 0: plt.legend()
    plt.savefig('./results/plots/reconstruction.png')
    print("Saved ./results/plots/reconstruction.png")


def plot_latent_traversals(model, dataloader, device):
    model.eval()
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, SEQ_LEN)))
    x, _ = next(iter(dataloader))
    x_ref = x[0:1].to(device)

    with torch.no_grad():
        mu_t, _ = model.enc_thick(x_ref[:, 0])
        mu_c, _ = model.enc_camber(x_ref[:, 1])

        n_steps = 11
        vals = np.linspace(-5, 5, n_steps)

        # We have 5 physical dimensions to traverse (3 Thick, 2 Camber)
        phys_labels = ['Thick Max', 'Pos T_Max', 'LE Radius', 'Camber Max', 'Pos C_Max']

        fig, axes = plt.subplots(5, n_steps, figsize=(18, 15))

        # --- Thickness Traversals (Indices 0, 1, 2) ---
        for dim_idx in range(3):
            for step_idx, val in enumerate(vals):
                z_t = mu_t.clone()
                z_t[0, dim_idx] = val
                z_c = mu_c.clone()  # Keep camber constant

                cp_t_raw = model.dec_thick(z_t)
                cp_t_pos = F.softplus(cp_t_raw)
                zeros = torch.zeros(cp_t_pos.shape[0], 1, device=device)
                cp_t = torch.cat([zeros, cp_t_pos, zeros], dim=1)

                cp_c = model.dec_camber(z_c)

                t_out = model.bspline(cp_t).cpu().numpy()[0]
                c_out = model.bspline(cp_c).cpu().numpy()[0]
                yu = c_out + t_out / 2
                yl = c_out - t_out / 2

                ax = axes[dim_idx, step_idx]
                ax.plot(x_val, yu, 'b-', x_val, yl, 'b-')
                ax.fill_between(x_val, yl, yu, alpha=0.3)
                ax.set_aspect('equal')
                ax.set_ylim(-0.3, 0.3)
                ax.axis('off')
                if step_idx == 0:
                    ax.set_title(f"{phys_labels[dim_idx]}\nVal: {val:.1f}")
                else:
                    ax.set_title(f"{val:.1f}")

        # --- Camber Traversals (Indices 0, 1) ---
        for dim_idx in range(2):
            plot_row = dim_idx + 3
            for step_idx, val in enumerate(vals):
                z_t = mu_t.clone()  # Keep thick constant
                z_c = mu_c.clone()
                z_c[0, dim_idx] = val

                cp_t_raw = model.dec_thick(z_t)
                cp_t_pos = F.softplus(cp_t_raw)
                zeros = torch.zeros(cp_t_pos.shape[0], 1, device=device)
                cp_t = torch.cat([zeros, cp_t_pos, zeros], dim=1)

                cp_c = model.dec_camber(z_c)

                t_out = model.bspline(cp_t).cpu().numpy()[0]
                c_out = model.bspline(cp_c).cpu().numpy()[0]
                yu = c_out + t_out / 2
                yl = c_out - t_out / 2

                ax = axes[plot_row, step_idx]
                ax.plot(x_val, yu, 'r-', x_val, yl, 'r-')
                ax.fill_between(x_val, yl, yu, alpha=0.3, color='red')
                ax.set_aspect('equal')
                ax.set_ylim(-0.3, 0.3)
                ax.axis('off')
                if step_idx == 0:
                    ax.set_title(f"{phys_labels[dim_idx + 3]}\nVal: {val:.1f}")
                else:
                    ax.set_title(f"{val:.1f}")

        plt.tight_layout()
        plt.savefig('./results/plots/latent_traversals.png')
        print("Saved ./results/plots/latent_traversals.png")


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found. Please create it and add .dat files.")
        return

    # 1. Data
    full_dataset = RealAirfoilDataset(DATA_DIR, n_points=SEQ_LEN, max_files=MAX_FILES)
    if len(full_dataset) < 2: return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model & Optimizer
    model = AirfoilVAE(SEQ_LEN, NUM_CP, DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)}...")

    # 3. Training Loop
    for ep in range(EPOCHS):
        model.train()
        loss_acc = 0

        for x, f in train_dl:
            x, f = x.to(DEVICE), f.to(DEVICE)
            optimizer.zero_grad()

            rec, (mt, lt), (mc, lc), (ct, cc) = model(x)
            loss = loss_function(rec, x, mt, lt, mc, lc, ct, cc, f, model.log_prior, ep)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            loss_acc += loss.item()

        avg_loss = loss_acc / len(train_dl)
        scheduler.step(avg_loss)

        if ep % 5 == 0:
            model.eval()
            val_rec = 0
            with torch.no_grad():
                for vx, vf in val_dl:
                    vx = vx.to(DEVICE)
                    vrec, _, _, _ = model(vx)
                    val_rec += F.mse_loss(vrec, vx).item()
            print(f"Ep {ep + 1}/{EPOCHS} | Train Loss: {avg_loss:.2f} | Val Recon MSE: {val_rec / len(val_dl):.5f}")

    # 4. Analysis
    print("Generating plots...")
    visualize_airfoils(val_dl)
    visualize_reconstruction(model, val_dl, DEVICE)
    plot_correlation_matrix(model, val_dl, DEVICE)
    plot_latent_traversals(model, val_dl, DEVICE)

    # 5. Save Model
    print("Saving model...")
    torch.save(model.state_dict(), './results/model/airfoil_vae.pth')

    print("Done.")


if __name__ == "__main__":
    main()