import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
CONFIG = {
    'BATCH_SIZE': 64,
    'EPOCHS': 50,
    'LR': 1e-3,
    'NUM_CP': 15,  # Control Points per curve (Thickness/Camber)
    'SEQ_LEN': 200,  # Resolution for plotting/curve evaluation
    'LATENT_PHYS_T': 3,  # Physics dims for Thickness (Max T, Pos T, Radius)
    'LATENT_FREE_T': 3,  # Free dims for Thickness
    'LATENT_PHYS_C': 2,  # Physics dims for Camber (Max C, Pos C)
    'LATENT_FREE_C': 2,  # Free dims for Camber
    'DATA_DIR': '../VAEBladerData/data/airfoil/naca_gen',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'RESULTS_DIR': './results_pipca'
}

os.makedirs(f"{CONFIG['RESULTS_DIR']}/plots", exist_ok=True)
os.makedirs(f"{CONFIG['RESULTS_DIR']}/models", exist_ok=True)
print(f"Running on {CONFIG['DEVICE']}")


# ============================================================================
# 2. MATH & FITTING UTILS
# ============================================================================

class BSplineUtils:
    """Utilities for B-Spline Basis calculation (Numpy & Torch compatible logic)"""

    @staticmethod
    def get_basis(u, n_cp, degree, backend='numpy'):
        """Cox-De Boor recursion"""
        xp = np if backend == 'numpy' else torch

        n = n_cp - 1
        m = n + degree + 1
        kv = xp.zeros(m + 1)
        if backend == 'torch': kv = kv.to(u.device)

        # Clamped Knot Vector
        start, end = degree + 1, m - degree
        linspace_func = np.linspace if backend == 'numpy' else torch.linspace

        # Proper linspace handling for torch vs numpy
        if backend == 'numpy':
            kv[start:end] = linspace_func(0, 1, end - start + 2)[1:-1]
        else:
            kv[start:end] = linspace_func(0, 1, end - start + 2, device=u.device)[1:-1]

        kv[end:] = 1.0

        # Basis 0
        N = xp.zeros((len(u), m))
        if backend == 'torch': N = N.to(u.device)

        for i in range(m):
            mask = (u >= kv[i]) & (u < kv[i + 1])
            if i == m - 1:
                mask = mask | (u == kv[i + 1])
            if backend == 'numpy':
                N[:, i] = mask.astype(float)
            else:
                N[:, i] = mask.float()

        # Recursion
        for d in range(1, degree + 1):
            N_new = xp.zeros((len(u), m - d))
            if backend == 'torch': N_new = N_new.to(u.device)

            for i in range(m - d - 1):
                d1 = kv[i + d] - kv[i]
                d2 = kv[i + d + 1] - kv[i + 1]

                term1 = 0.0
                if d1 > 1e-6:
                    term1 = ((u - kv[i]) / d1) * N[:, i]

                term2 = 0.0
                if d2 > 1e-6:
                    term2 = ((kv[i + d + 1] - u) / d2) * N[:, i + 1]

                N_new[:, i] = term1 + term2
            N = N_new

        return N[:, :n_cp]


class ConstrainedBSplineFitter:
    """
    Fits 1D curves (Thickness/Camber) to Control Points.
    Enforces CPs[0] = 0 and CPs[-1] = 0 (Closure).
    """

    def __init__(self, num_cp=15, degree=3, num_eval_points=200):
        self.num_cp = num_cp

        # Precompute Basis for Fitting (Cosine Spacing for better LE resolution)
        theta = np.linspace(0, np.pi, num_eval_points)
        u = 0.5 * (1 - np.cos(theta))
        self.basis = BSplineUtils.get_basis(u, num_cp, degree, backend='numpy')

    def fit(self, y_curve):
        """
        Solves A*x = b subject to x[0]=0, x[-1]=0.
        We remove first and last col from Basis and solve for inner CPs.
        """
        # System: Basis @ CPs = y_curve
        # Constraint: CPs[0] = 0, CPs[-1] = 0
        # Reduced System: Basis[:, 1:-1] @ CPs[1:-1] = y_curve

        A_reduced = self.basis[:, 1:-1]
        b = y_curve

        # Least Squares
        cps_inner, _, _, _ = lstsq(A_reduced, b)

        # Reconstruct full vector
        cps = np.zeros(self.num_cp)
        cps[1:-1] = cps_inner
        return cps


class BSplineLayer(nn.Module):
    """Differentiable Torch Layer for Curve Generation"""

    def __init__(self, num_cp, degree=3, num_eval_points=200, device='cpu'):
        super().__init__()
        theta = torch.linspace(0, np.pi, num_eval_points, device=device)
        u = 0.5 * (1 - torch.cos(theta))
        self.register_buffer('basis', BSplineUtils.get_basis(u, num_cp, degree, backend='torch'))

    def forward(self, cps):
        # cps: [Batch, Num_CP]
        # Basis: [Seq_Len, Num_CP]
        # Output: [Batch, Seq_Len]
        return torch.matmul(cps, self.basis.T)


# ============================================================================
# 3. PI-PCA MODEL
# ============================================================================
class PIPCA:
    """
    Physics-Informed PCA (Linear Benchmark).
    Decomposes shape into Physics (Linear Regression) + Style (PCA on Residuals).
    """

    def __init__(self, n_slack_components=5):
        self.n_slack = n_slack_components
        self.phys_regressor = LinearRegression(fit_intercept=False)
        self.pca = PCA(n_components=n_slack_components)

    def fit(self, X_cps, Y_phys):
        """
        X_cps: (N, M) Flattened Control Points
        Y_phys: (N, P) Physics Features
        """
        # 1. Learn Physics -> Shape mapping
        self.phys_regressor.fit(Y_phys, X_cps)
        X_predicted_by_physics = self.phys_regressor.predict(Y_phys)

        # 2. Compute Residuals (Slack)
        Residuals = X_cps - X_predicted_by_physics

        # 3. PCA on Residuals
        self.pca.fit(Residuals)

        print(f"[PI-PCA] Explained Variance (Slack): {np.sum(self.pca.explained_variance_ratio_):.4f}")

    def transform(self, X_cps, Y_phys):
        """Returns [Y_phys, Z_slack]"""
        X_phys = self.phys_regressor.predict(Y_phys)
        Residuals = X_cps - X_phys
        Z_slack = self.pca.transform(Residuals)
        return np.hstack([Y_phys, Z_slack])

    def inverse_transform(self, Y_phys, Z_slack):
        """Returns reconstructed X_cps"""
        X_phys = self.phys_regressor.predict(Y_phys)
        Residuals_rec = self.pca.inverse_transform(Z_slack)
        return X_phys + Residuals_rec


# ============================================================================
# 4. VAE MODEL
# ============================================================================
class AirfoilVAE_CP(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cp = CONFIG['NUM_CP']
        self.device = CONFIG['DEVICE']

        # -- Dimensions --
        self.dim_t = CONFIG['LATENT_PHYS_T'] + CONFIG['LATENT_FREE_T']
        self.dim_c = CONFIG['LATENT_PHYS_C'] + CONFIG['LATENT_FREE_C']

        # -- Encoders (CPs -> Latent) --
        # Input: 15 CPs. Output: Latent Mean/Var
        self.enc_thick = nn.Sequential(
            nn.Linear(self.num_cp, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU()
        )
        self.enc_camber = nn.Sequential(
            nn.Linear(self.num_cp, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU()
        )

        self.head_mu_t = nn.Linear(32, self.dim_t)
        self.head_lv_t = nn.Linear(32, self.dim_t)
        self.head_mu_c = nn.Linear(32, self.dim_c)
        self.head_lv_c = nn.Linear(32, self.dim_c)

        # -- Decoders (Latent -> Inner CPs) --
        # Output is Num_CP - 2 because first/last are clamped to 0
        out_dim = self.num_cp - 2

        self.dec_thick = nn.Sequential(
            nn.Linear(self.dim_t, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, out_dim)
        )

        self.dec_camber = nn.Sequential(
            nn.Linear(self.dim_c, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, out_dim)
        )

        # -- Curve Generation --
        self.bspline = BSplineLayer(self.num_cp, degree=3,
                                    num_eval_points=CONFIG['SEQ_LEN'],
                                    device=self.device)

        # Physics Prior (for Free KL)
        self.log_prior = nn.Parameter(torch.tensor([-2.0], device=self.device))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_cps):
        # x_cps shape: [Batch, 30] -> [T_CP (15), C_CP (15)]
        cp_t_in = x_cps[:, :self.num_cp]
        cp_c_in = x_cps[:, self.num_cp:]

        # 1. Encode
        ht = self.enc_thick(cp_t_in)
        mt, lt = self.head_mu_t(ht), self.head_lv_t(ht)
        zt = self.reparameterize(mt, lt)

        hc = self.enc_camber(cp_c_in)
        mc, lc = self.head_mu_c(hc), self.head_lv_c(hc)
        zc = self.reparameterize(mc, lc)

        # 2. Decode (Inner CPs)
        cp_t_inner = self.dec_thick(zt)
        cp_c_inner = self.dec_camber(zc)

        # 3. Apply Geometric Constraints
        # Thickness: Positive & Clamped
        cp_t_pos = F.softplus(cp_t_inner)  # Ensure positive thickness
        zeros = torch.zeros(cp_t_pos.shape[0], 1, device=self.device)
        cp_t_out = torch.cat([zeros, cp_t_pos, zeros], dim=1)

        # Camber: Clamped only
        cp_c_out = torch.cat([zeros, cp_c_inner, zeros], dim=1)

        # 4. Generate Curves (for Loss calculation)
        curve_t = self.bspline(cp_t_out)
        curve_c = self.bspline(cp_c_out)

        return (torch.stack([curve_t, curve_c], 1),
                (mt, lt), (mc, lc),
                (cp_t_out, cp_c_out))


# ============================================================================
# 5. DATASET & LOADING
# ============================================================================
class AirfoilDataset(Dataset):
    def __init__(self):
        self.fitter = ConstrainedBSplineFitter(num_cp=CONFIG['NUM_CP'],
                                               num_eval_points=CONFIG['SEQ_LEN'])
        self.data_cps, self.features, self.data_curves = self._load_data()

    def _load_data(self):
        files = sorted(glob.glob(os.path.join(CONFIG['DATA_DIR'], "*.dat")))
        if not files:
            print("WARNING: No .dat files found. Generating dummy data for test.")
            return self._generate_dummy_data()

        cps_list, feat_list, curve_list = [], [], []
        theta = np.linspace(0, np.pi, CONFIG['SEQ_LEN'])
        x_grid = 0.5 * (1 - np.cos(theta))

        print(f"Loading {len(files)} airfoils...")
        for fpath in files:
            try:
                # Basic parsing
                raw = np.loadtxt(fpath, skiprows=1) if self._has_header(fpath) else np.loadtxt(fpath)

                # Split Upper/Lower
                le_idx = np.argmin(raw[:, 0])
                ux, uy = raw[:le_idx + 1, 0], raw[:le_idx + 1, 1]
                lx, ly = raw[le_idx:, 0], raw[le_idx:, 1]

                # Interpolate to common grid
                u_int = interp1d(np.flip(ux), np.flip(uy), bounds_error=False, fill_value=0.0)
                l_int = interp1d(lx, ly, bounds_error=False, fill_value=0.0)
                yu, yl = u_int(x_grid), l_int(x_grid)

                # Convert to Thickness/Camber
                yt = np.clip(yu - yl, 0, None)
                yc = (yu + yl) / 2.0
                # Clamp ends manually to be safe
                yt[-1] = 0;
                yc[0] = 0;
                yc[-1] = 0

                # Extract Physics Features
                max_t = np.max(yt)
                pos_max_t = x_grid[np.argmax(yt)]
                # Simple Radius Estimate (parabolic fit near LE)
                mask_le = (x_grid > 0.001) & (x_grid < 0.05)
                if np.sum(mask_le) > 3:
                    r_est = np.mean((yt[mask_le] ** 2) / (8 * x_grid[mask_le]))
                else:
                    r_est = 0.0

                max_c = np.max(yc)
                pos_max_c = x_grid[np.argmax(np.abs(yc))]

                # Fit B-Splines (Get CPs)
                cp_t = self.fitter.fit(yt)
                cp_c = self.fitter.fit(yc)

                cps_combined = np.concatenate([cp_t, cp_c])

                cps_list.append(cps_combined)
                feat_list.append([max_t, pos_max_t, r_est, max_c, pos_max_c])
                curve_list.append(np.stack([yt, yc]))

            except Exception as e:
                continue

        if not cps_list:
            print("No valid airfoils found. Check DATA_DIR path or file format.")
            return self._generate_dummy_data()

        # Normalize Features
        feats = np.array(feat_list, dtype=np.float32)
        self.feat_mean = feats.mean(0)
        self.feat_std = feats.std(0)
        feats_norm = (feats - self.feat_mean) / (self.feat_std + 1e-6)

        return (torch.tensor(np.array(cps_list), dtype=torch.float32),
                torch.tensor(feats_norm, dtype=torch.float32),
                torch.tensor(np.array(curve_list), dtype=torch.float32))

    def _generate_dummy_data(self):
        # Generate 100 fake airfoils if no data found
        N = 100
        cps = torch.randn(N, CONFIG['NUM_CP'] * 2)
        feats = torch.randn(N, 5)
        curves = torch.randn(N, 2, CONFIG['SEQ_LEN'])
        self.feat_mean, self.feat_std = np.zeros(5), np.ones(5)
        return cps, feats, curves

    def _has_header(self, fpath):
        try:
            np.loadtxt(fpath); return False
        except:
            return True

    def __len__(self):
        return len(self.data_cps)

    def __getitem__(self, i):
        return self.data_cps[i], self.features[i]


# ============================================================================
# 6. LOSS FUNCTION
# ============================================================================
def vae_loss_fn(rec_curves, gt_curves, mt, lt, mc, lc, feat, ep):
    """
    rec_curves: Generated from Decoder CPs
    gt_curves:  Generated from Encoder CPs (Self-Reconstruction) OR Ground Truth
    """
    # 1. Reconstruction Loss (MSE on the curve points)
    mse = F.mse_loss(rec_curves, gt_curves, reduction='sum')

    # 2. KL Divergence (Aligned Physics)
    # Thick: indices 0,1,2 <-> Feat 0,1,2
    kl_tp = 0.5 * torch.sum(lt[:, :3].exp() + (mt[:, :3] - feat[:, :3]) ** 2 - 1 - lt[:, :3], 1)
    # Camber: indices 0,1 <-> Feat 3,4
    kl_cp = 0.5 * torch.sum(lc[:, :2].exp() + (mc[:, :2] - feat[:, 3:]) ** 2 - 1 - lc[:, :2], 1)

    # 3. KL Divergence (Free Latents -> Standard Normal)
    kl_tf = -0.5 * torch.sum(1 + lt[:, 3:] - mt[:, 3:].pow(2) - lt[:, 3:].exp(), 1)
    kl_cf = -0.5 * torch.sum(1 + lc[:, 2:] - mc[:, 2:].pow(2) - lc[:, 2:].exp(), 1)

    beta = min(1.0, (ep + 1) / 10.0)  # Warmup
    return mse + beta * (torch.sum(kl_tp + kl_cp + kl_tf + kl_cf))


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================
def main():
    # --- Data ---
    ds = AirfoilDataset()
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    # --- 1. Train VAE ---
    print("\n[1/3] Training VAE...")
    vae = AirfoilVAE_CP().to(CONFIG['DEVICE'])
    opt = optim.Adam(vae.parameters(), lr=CONFIG['LR'])

    for ep in range(CONFIG['EPOCHS']):
        vae.train()
        train_loss = 0
        for cps, feats in train_dl:
            cps, feats = cps.to(CONFIG['DEVICE']), feats.to(CONFIG['DEVICE'])
            opt.zero_grad()

            # Forward
            rec_curves, (mt, lt), (mc, lc), (cp_t_out, cp_c_out) = vae(cps)

            # Ground Truth Curves for Loss (Reconstructed from input CPs)
            with torch.no_grad():
                gt_t = vae.bspline(cps[:, :CONFIG['NUM_CP']])
                gt_c = vae.bspline(cps[:, CONFIG['NUM_CP']:])
                gt_curves = torch.stack([gt_t, gt_c], 1)

            loss = vae_loss_fn(rec_curves, gt_curves, mt, lt, mc, lc, feats, ep)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep + 1}: Loss {train_loss / len(train_dl):.2f}")

    # --- 2. Train PI-PCA ---
    print("\n[2/3] Training PI-PCA...")

    # Collect Training Data for PCA Fit (No Data Leakage!)
    train_cps_list = []
    train_feats_list = []

    for cps, feats in train_dl:
        train_cps_list.append(cps.cpu().numpy())
        train_feats_list.append(feats.cpu().numpy())

    train_cps_np = np.concatenate(train_cps_list, axis=0)
    train_feats_np = np.concatenate(train_feats_list, axis=0)

    # Fit PCA only on Training Data
    pipca = PIPCA(n_slack_components=5)
    pipca.fit(train_cps_np, train_feats_np)

    # --- 3. Benchmark & Plot ---
    print("\n[3/3] Benchmarking & Plotting...")
    vae.eval()

    # Get one batch from VALIDATION set for reconstruction comparison
    val_iter = iter(val_dl)
    cps_batch, feats_batch = next(val_iter)
    cps_batch = cps_batch.to(CONFIG['DEVICE'])

    # A) VAE Reconstruction
    with torch.no_grad():
        rec_vae_curves, _, _, (cp_t_vae, cp_c_vae) = vae(cps_batch)

        # Ground Truth Curves (from original CPs)
        gt_t = vae.bspline(cps_batch[:, :CONFIG['NUM_CP']])
        gt_c = vae.bspline(cps_batch[:, CONFIG['NUM_CP']:])
        gt_curves = torch.stack([gt_t, gt_c], 1)

    mse_vae = F.mse_loss(rec_vae_curves, gt_curves).item()

    # B) PI-PCA Reconstruction
    # Transform Validation Batch
    latents_pca = pipca.transform(cps_batch.cpu().numpy(), feats_batch.numpy())

    # Inverse (Split Latents back to Phys and Slack)
    rec_pca_cps = pipca.inverse_transform(latents_pca[:, :5], latents_pca[:, 5:])

    # Post-process PCA CPs (Manual Clamping)
    rec_pca_cps[:, 0] = 0;
    rec_pca_cps[:, CONFIG['NUM_CP'] - 1] = 0  # Thick ends
    rec_pca_cps[:, CONFIG['NUM_CP']] = 0;
    rec_pca_cps[:, -1] = 0  # Camber ends

    # Convert PCA CPs to Curves using Torch Layer
    t_pca_torch = torch.tensor(rec_pca_cps[:, :CONFIG['NUM_CP']]).float().to(CONFIG['DEVICE'])
    c_pca_torch = torch.tensor(rec_pca_cps[:, CONFIG['NUM_CP']:]).float().to(CONFIG['DEVICE'])

    with torch.no_grad():
        curve_t_pca = vae.bspline(t_pca_torch)
        curve_c_pca = vae.bspline(c_pca_torch)
        rec_pca_curves = torch.stack([curve_t_pca, curve_c_pca], 1)

    mse_pca = F.mse_loss(rec_pca_curves, gt_curves).item()

    print(f"\nRESULTS:\nVAE MSE: {mse_vae:.6f}\nPI-PCA MSE: {mse_pca:.6f}")

    # --- Visualization: Max Thickness Sweep ---
    vals = np.linspace(-2, 2, 5)

    # Use Mean of TRAINING features for the sweep baseline
    mean_feat = train_feats_np.mean(axis=0)

    plt.figure(figsize=(12, 6))

    for i, val in enumerate(vals):
        # 1. VAE Gen
        # Construct Latent: [Phys_T(3), Free_T(3)]
        # FIX: Explicit device usage
        zt = torch.zeros(1, 6, device=CONFIG['DEVICE'])
        zt[0, 0] = val  # Set Max Thickness (Aligned Index 0)

        zc = torch.zeros(1, 4, device=CONFIG['DEVICE'])  # Mean Camber

        with torch.no_grad():
            cp_t = vae.dec_thick(zt)
            # FIX: Explicit device usage in cat
            cp_t = torch.cat([
                torch.zeros(1, 1, device=CONFIG['DEVICE']),
                F.softplus(cp_t),
                torch.zeros(1, 1, device=CONFIG['DEVICE'])
            ], 1)

            cp_c = vae.dec_camber(zc)
            cp_c = torch.cat([
                torch.zeros(1, 1, device=CONFIG['DEVICE']),
                cp_c,
                torch.zeros(1, 1, device=CONFIG['DEVICE'])
            ], 1)

            cv_t = vae.bspline(cp_t).cpu().numpy()[0]
            cv_c = vae.bspline(cp_c).cpu().numpy()[0]

        # 2. PCA Gen
        feat_sweep = mean_feat.copy()
        feat_sweep[0] = val  # Set Max Thickness
        slack_sweep = np.zeros(5)  # Mean slack (since PCA is centered)

        rec_p = pipca.inverse_transform([feat_sweep], [slack_sweep])[0]
        # Clamp
        rec_p[0] = 0;
        rec_p[CONFIG['NUM_CP'] - 1] = 0
        rec_p[CONFIG['NUM_CP']] = 0;
        rec_p[-1] = 0

        # Convert to curve (Manual basis mult)
        basis_np = ds.fitter.basis  # (200, 15)
        cp_t_p_np = rec_p[:CONFIG['NUM_CP']]
        cp_c_p_np = rec_p[CONFIG['NUM_CP']:]
        cv_t_p = basis_np @ cp_t_p_np
        cv_c_p = basis_np @ cp_c_p_np

        # Plot VAE
        plt.subplot(2, 5, i + 1)
        plt.plot(cv_c + cv_t / 2, 'b')
        plt.plot(cv_c - cv_t / 2, 'b')
        plt.title(f"VAE T={val:.1f}")
        plt.axis('off')
        plt.axis('equal')

        # Plot PCA
        plt.subplot(2, 5, i + 6)
        plt.plot(cv_c_p + cv_t_p / 2, 'r--')
        plt.plot(cv_c_p - cv_t_p / 2, 'r--')
        plt.title(f"PCA T={val:.1f}")
        plt.axis('off')
        plt.axis('equal')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULTS_DIR']}/plots/benchmark_comparison.png")
    print(f"Comparison plot saved to {CONFIG['RESULTS_DIR']}/plots/benchmark_comparison.png")


if __name__ == "__main__":
    main()