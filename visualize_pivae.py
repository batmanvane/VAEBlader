import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from scipy.stats import pearsonr

# --- IMPORT YOUR MODEL ---
# Adjust import to match your filename
from createPIVAE_refactor_LE_and_Input import AG_VAE, RealAirfoilDataset, SEQ_LEN, NUM_CP, DEVICE, DATA_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = './results/model/ag_vae_cps.pth'
SAVE_DIR = './results/plots'
LATENT_NAMES = ['Max Thickness', 'Pos Max Thickness', 'LE Radius',
                'Max Camber', 'Pos Max Camber', 'TE Direction']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_model_and_data():
    print("Loading Data & Model...")
    # Load Dataset
    ds = RealAirfoilDataset(DATA_DIR, num_cp=NUM_CP, seq_len=SEQ_LEN)
    # Split same as training to see validation performance
    train_sz = int(0.9 * len(ds))
    _, val_ds = random_split(ds, [train_sz, len(ds) - train_sz])
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)  # Large batch for scatter plot

    # Load Model
    model = AG_VAE(num_cp=NUM_CP, seq_len=SEQ_LEN, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, val_dl


def get_latent_embeddings(model, dataloader):
    """ Extract latent vectors z and ground truth physics labels y from validation set """
    latents = []
    labels = []

    with torch.no_grad():
        for cp, f, _ in dataloader:
            cp = cp.to(DEVICE)
            # Encode
            mu_t, _ = model.enc_thick(cp[:, 0])
            mu_c, _ = model.enc_camber(cp[:, 1])

            # Combine Physical Latents (First 3 of Thickness + First 3 of Camber)
            z_phys = torch.cat([mu_t[:, :3], mu_c[:, :3]], dim=1)

            latents.append(z_phys.cpu().numpy())
            labels.append(f.numpy())  # Ground truth physics labels

    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)


# ============================================================================
# PLOTTING LOGIC
# ============================================================================
def plot_alignment_dashboard(model, z_val, y_val):
    print("Generating Alignment Dashboard...")

    # Setup Grid: 6 Rows (Variables) x 2 Columns (Geometry | Scatter)
    fig, axes = plt.subplots(6, 2, figsize=(12, 18), gridspec_kw={'width_ratios': [1, 1.5]})
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    x_grid = 0.5 * (1 - np.cos(np.linspace(0, np.pi, SEQ_LEN)))  # Cosine x-axis

    # Iterate over 6 Physical Variables
    for i in range(6):
        ax_geo = axes[i, 0]
        ax_scat = axes[i, 1]
        var_name = LATENT_NAMES[i]

        # --- 1. GEOMETRY TRAVERSAL (Left) ---
        # Traverse latent from -2.5 to 2.5 sigma
        traversal_vals = np.linspace(-2.5, 2.5, 7)
        colors = plt.cm.viridis(np.linspace(0, 1, 7))

        # Base vector (mean of dataset)
        z_base_t = torch.zeros(1, 6).to(DEVICE)
        z_base_c = torch.zeros(1, 6).to(DEVICE)

        for j, val in enumerate(traversal_vals):
            zt, zc = z_base_t.clone(), z_base_c.clone()

            # Modify the specific latent variable
            if i < 3:
                zt[0, i] = val  # Thickness Vars
            else:
                zc[0, i - 3] = val  # Camber Vars

            # Decode
            with torch.no_grad():
                t_out, c_out, _, _ = model.decode(zt, zc)

            yu = (c_out + t_out / 2).cpu().numpy()[0]
            yl = (c_out - t_out / 2).cpu().numpy()[0]

            # Plot Airfoil
            ax_geo.plot(x_grid, yu, color=colors[j], alpha=0.8, linewidth=1.5)
            ax_geo.plot(x_grid, yl, color=colors[j], alpha=0.8, linewidth=1.5)

        ax_geo.set_title(f"Geometry Variation: {var_name}")
        ax_geo.axis('equal')
        ax_geo.axis('off')

        # --- 2. PHYSICS ALIGNMENT SCATTER (Right) ---
        # Scatter: Ground Truth Physics (X) vs Latent Value (Y)
        # Note: y_val is normalized physics (standard score), z_val is latent

        # Calculate Correlation
        corr, _ = pearsonr(y_val[:, i], z_val[:, i])

        # Plot Data Points
        sns.scatterplot(x=y_val[:, i], y=z_val[:, i], ax=ax_scat,
                        s=10, alpha=0.3, color='#4c72b0', edgecolor=None)

        # Plot Perfect Alignment Line (y=x)
        min_val = min(y_val[:, i].min(), z_val[:, i].min())
        max_val = max(y_val[:, i].max(), z_val[:, i].max())
        ax_scat.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (1:1)')

        # Styling
        ax_scat.set_title(f"Latent Alignment (R = {corr:.2f})")
        ax_scat.set_xlabel(f"True Normalized {var_name}")
        ax_scat.set_ylabel(f"Latent Variable z_{i}")
        ax_scat.grid(True, alpha=0.3)
        ax_scat.legend(loc='upper left')

    # Save
    plt.tight_layout()
    save_path = f"{SAVE_DIR}/pivae_alignment_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"Analysis saved to {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import os

    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 1. Load Resources
    vae, loader = load_model_and_data()

    # 2. Get Embeddings
    z_values, y_values = get_latent_embeddings(vae, loader)

    # 3. Plot
    plot_alignment_dashboard(vae, z_values, y_values)