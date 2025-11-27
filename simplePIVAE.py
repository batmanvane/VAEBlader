import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os


#This file implements a Variational Autoencoder (VAE) for generating airfoil shapes using differentiable B-splines.
#The model encodes airfoil thickness and camber distributions into a latent space, decodes
#them into B-spline control points, and reconstructs the airfoil shapes.
#The training process includes reconstruction loss, KL divergence, physics-based constraints,
#and smoothness regularization. The code also includes synthetic data generation and visualization of results.
#Synthetic airfoil data is generated using NACA-like thickness and parabolic camber distributions.


# ============================================================================
# 1. Device Configuration
# ============================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_device()
print(f"Running on device: {DEVICE}")


# ============================================================================
# 2. Differentiable B-Spline Layer
# ============================================================================
class BSplineLayer(nn.Module):
    """
    Computes B-spline curves from control points using a precomputed basis matrix.
    Supports backpropagation to control points.
    """

    def __init__(self, num_control_points, degree=3, num_eval_points=100, device='cpu'):
        super().__init__()
        self.num_control_points = num_control_points
        self.degree = degree

        # Evaluation points u along the curve [0, 1]
        u = torch.linspace(0, 1, num_eval_points, device=device)
        self.basis_matrix = self._precompute_basis(u).to(device)

    def _precompute_basis(self, u):
        """Cox-de Boor recursion to compute basis functions N_{i,p}(u)"""
        n = self.num_control_points - 1
        p = self.degree
        m = n + p + 1

        # Create Clamped Uniform Knot Vector
        kv = torch.zeros(m + 1, device=u.device)
        start_internal = p + 1
        end_internal = m - p
        num_internal = end_internal - start_internal

        if num_internal > 0:
            # Internal knots uniformly spaced in (0, 1)
            internal = torch.linspace(0, 1, num_internal + 2, device=u.device)[1:-1]
            kv[start_internal:end_internal] = internal
        kv[end_internal:] = 1.0

        # Recursive Basis Calculation
        N = torch.zeros(u.shape[0], m, device=u.device)

        # Degree 0
        for i in range(m):
            mask = (u >= kv[i]) & (u < kv[i + 1])
            if i == m - 1:  # Include last point
                mask = mask | (u == kv[i + 1])
            N[:, i] = mask.float()

        # Degree 1...p
        for d in range(1, p + 1):
            N_new = torch.zeros(u.shape[0], m - d, device=u.device)
            for i in range(m - d - 1):
                # Left term
                denom1 = kv[i + d] - kv[i]
                term1 = 0.0
                if denom1 > 1e-6:
                    term1 = ((u - kv[i]) / denom1) * N[:, i]

                # Right term
                denom2 = kv[i + d + 1] - kv[i + 1]
                term2 = 0.0
                if denom2 > 1e-6:
                    term2 = ((kv[i + d + 1] - u) / denom2) * N[:, i + 1]

                N_new[:, i] = term1 + term2
            N = N_new

        # Return basis for the n+1 control points
        return N[:, :self.num_control_points]

    def forward(self, control_points):
        # (Batch, ControlPoints) @ (EvalPoints, ControlPoints)^T -> (Batch, EvalPoints)
        return torch.matmul(control_points, self.basis_matrix.T)


# ============================================================================
# 3. Model Architecture (Airfoil Generator)
# ============================================================================
class EncoderCNN(nn.Module):
    """1D CNN to encode thickness/camber distributions"""

    def __init__(self, input_len=100, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute flatten size automatically
        dummy_input = torch.zeros(1, 1, input_len)
        with torch.no_grad():
            dummy_out = self.net(dummy_input)
        flat_size = dummy_out.shape[1]

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class DecoderFCN(nn.Module):
    """Decodes latent vector to B-Spline Control Points"""

    def __init__(self, latent_dim=8, num_control_points=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, num_control_points)
        )

    def forward(self, z):
        return self.net(z)


class AirfoilGeneratorVAE(nn.Module):
    def __init__(self, seq_len=100, latent_dim_phys=2, latent_dim_free=4, num_cp=15, device='cpu'):
        super().__init__()
        total_latent = latent_dim_phys + latent_dim_free

        # Two-Way Encoder
        self.encoder_thick = EncoderCNN(seq_len, total_latent)
        self.encoder_camber = EncoderCNN(seq_len, total_latent)

        # Two-Way Decoder
        self.decoder_thick_cp = DecoderFCN(total_latent, num_cp)
        self.decoder_camber_cp = DecoderFCN(total_latent, num_cp)

        # Shared B-Spline Layer
        self.bspline = BSplineLayer(num_cp, degree=3, num_eval_points=seq_len, device=device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Input x: [Batch, 2, SeqLen] (Channel 0: Thickness, Channel 1: Camber)
        thick_in = x[:, 0, :]
        camber_in = x[:, 1, :]

        # 1. Encode
        mu_t, logvar_t = self.encoder_thick(thick_in)
        mu_c, logvar_c = self.encoder_camber(camber_in)

        z_t = self.reparameterize(mu_t, logvar_t)
        z_c = self.reparameterize(mu_c, logvar_c)

        # 2. Decode to Control Points
        cp_t_raw = self.decoder_thick_cp(z_t)
        cp_c = self.decoder_camber_cp(z_c)

        # 3. Apply Constraints
        # Thickness CP must be positive -> Use Softplus
        cp_t = F.softplus(cp_t_raw)

        # 4. Generate Curves from CPs
        thick_out = self.bspline(cp_t)
        camber_out = self.bspline(cp_c)

        # Stack for output
        x_recon = torch.stack([thick_out, camber_out], dim=1)

        return x_recon, (mu_t, logvar_t, z_t, cp_t), (mu_c, logvar_c, z_c, cp_c)


# ============================================================================
# 4. Data Generation & Loss
# ============================================================================
class SyntheticAirfoilDataset(Dataset):
    """Generates synthetic airfoils (NACA-like thickness + Parabolic camber)"""

    def __init__(self, n_samples=2000, n_points=100):
        self.n_samples = n_samples
        self.n_points = n_points
        self.data, self.features = self._generate_data()

    def _generate_data(self):
        # Random Physical params
        t_max = np.random.uniform(0.05, 0.20, self.n_samples)  # Thickness
        m_max = np.random.uniform(0.00, 0.06, self.n_samples)  # Camber
        p_max = np.random.uniform(0.3, 0.5, self.n_samples)  # Camber pos
        x = np.linspace(0, 1, self.n_points)

        thickness_data = []
        camber_data = []

        for i in range(self.n_samples):
            # Thickness distribution
            t = t_max[i]
            yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
            yt = np.abs(yt - x * yt[-1])  # Fix TE to 0 and ensure positive

            # Camber distribution
            m = m_max[i];
            p = p_max[i]
            yc = np.zeros_like(x)
            if m > 0:
                idx_f = x <= p
                yc[idx_f] = (m / p ** 2) * (2 * p * x[idx_f] - x[idx_f] ** 2)
                idx_a = x > p
                yc[idx_a] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[idx_a] - x[idx_a] ** 2)

            thickness_data.append(yt)
            camber_data.append(yc)

        # Shape: (N, 2, 100)
        data = np.stack([np.array(thickness_data), np.array(camber_data)], axis=1)
        # Features: [t_max, m_max]
        features = np.stack([t_max, m_max], axis=1)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.features[idx]


def loss_function(recon_x, x, mu_t, logvar_t, mu_c, logvar_c, z_t, z_c, features, cp_t, cp_c):
    # 1. Reconstruction Loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence
    kld_t = -0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
    kld_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())

    # 3. Physics Loss (Force z[0] to match physical feature)
    # Normalize targets roughly to N(0,1) for balanced gradient
    target_t = (features[:, 0] - 0.125) / 0.04
    target_c = (features[:, 1] - 0.03) / 0.015
    phys_loss_t = F.mse_loss(z_t[:, 0], target_t, reduction='sum')
    phys_loss_c = F.mse_loss(z_c[:, 0], target_c, reduction='sum')

    # 4. Smoothness Regularization (2nd derivative of Control Points)
    reg_t = torch.mean((cp_t[:, :-2] - 2 * cp_t[:, 1:-1] + cp_t[:, 2:]) ** 2) * 10
    reg_c = torch.mean((cp_c[:, :-2] - 2 * cp_c[:, 1:-1] + cp_c[:, 2:]) ** 2) * 10

    loss = recon_loss + (kld_t + kld_c) + 10.0 * (phys_loss_t + phys_loss_c) + (reg_t + reg_c)
    return loss


# ============================================================================
# 5. Training & Visualization Loop
# ============================================================================
def run_experiment():
    batch_size = 64
    epochs = 40
    lr = 1e-3

    dataset = SyntheticAirfoilDataset(n_samples=2000, n_points=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AirfoilGeneratorVAE(seq_len=100, num_cp=15, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting Training...")
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for data, feat in dataloader:
            data, feat = data.to(DEVICE), feat.to(DEVICE)
            optimizer.zero_grad()
            recon, (mu_t, lv_t, z_t, cp_t), (mu_c, lv_c, z_c, cp_c) = model(data)
            loss = loss_function(recon, data, mu_t, lv_t, mu_c, lv_c, z_t, z_c, feat, cp_t, cp_c)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Avg Loss {train_loss / len(dataset):.4f}")

    # --- Plotting Results ---
    model.eval()

    # 1. Reconstruction Check
    data, _ = next(iter(dataloader))
    data = data.to(DEVICE)
    with torch.no_grad():
        recon, _, _ = model(data)

    # Convert back to Airfoil coords (Upper/Lower)
    x_val = np.linspace(0, 1, 100)
    data = data.cpu().numpy();
    recon = recon.cpu().numpy()

    yu_gt = data[0, 1, :] + data[0, 0, :] / 2
    yl_gt = data[0, 1, :] - data[0, 0, :] / 2
    yu_rc = recon[0, 1, :] + recon[0, 0, :] / 2
    yl_rc = recon[0, 1, :] - recon[0, 0, :] / 2

    plt.figure(figsize=(10, 5))
    plt.plot(x_val, yu_gt, 'k', label='Ground Truth')
    plt.plot(x_val, yl_gt, 'k')
    plt.plot(x_val, yu_rc, 'r--', label='Reconstruction')
    plt.plot(x_val, yl_rc, 'r--')
    plt.title("Airfoil Reconstruction (B-Spline VAE)")
    plt.axis('equal');
    plt.legend()
    plt.savefig('reconstruction_result.png')

    # 2. Latent Traversal (Physical Control)
    plt.figure(figsize=(12, 5))
    z_sweep = torch.linspace(-2, 2, 5).to(DEVICE)
    base_z = torch.zeros(1, 6).to(DEVICE)  # Assuming 2 phys + 4 free latent dim

    # Sweep Thickness Latent (Index 0 of Thickness Branch)
    plt.subplot(1, 2, 1)
    for val in z_sweep:
        z_in = base_z.clone();
        z_in[0, 0] = val
        with torch.no_grad():
            cp_t = F.softplus(model.decoder_thick_cp(z_in))  # Decode CP
            t = model.bspline(cp_t).cpu().numpy().flatten()  # CP -> Curve
            c = model.bspline(model.decoder_camber_cp(base_z)).cpu().numpy().flatten()
        plt.plot(x_val, c + t / 2, label=f'z={val:.1f}')
        plt.plot(x_val, c - t / 2)
    plt.title("Latent Traversal: Thickness")
    plt.axis('equal');
    plt.legend()

    # Sweep Camber Latent (Index 0 of Camber Branch)
    plt.subplot(1, 2, 2)
    for val in z_sweep:
        z_in = base_z.clone();
        z_in[0, 0] = val
        with torch.no_grad():
            cp_c = model.decoder_camber_cp(z_in)
            c = model.bspline(cp_c).cpu().numpy().flatten()
            t = model.bspline(F.softplus(model.decoder_thick_cp(base_z))).cpu().numpy().flatten()
        plt.plot(x_val, c + t / 2, label=f'z={val:.1f}')
        plt.plot(x_val, c - t / 2)
    plt.title("Latent Traversal: Camber")
    plt.axis('equal');
    plt.legend()

    plt.tight_layout()
    plt.savefig('latent_traversal.png')
    print("Results saved: reconstruction_result.png, latent_traversal.png")


if __name__ == "__main__":
    run_experiment()