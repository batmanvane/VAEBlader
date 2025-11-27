import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import os

# --- Compatibility Patch for Gradio/HuggingFace Hub ---
try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def save_token(token): pass

            @staticmethod
            def get_token(): return None


        huggingface_hub.HfFolder = HfFolder
except ImportError:
    pass
# ------------------------------------------------------

import gradio as gr

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# Model Paths
VAE_MODEL_PATH = './results/model/airfoil_vae.pth'
SURROGATE_MODEL_PATH = './results/model/aero_surrogate.pth'

# Data Paths (Reference only, not needed for inference)
AIRFOIL_DIR = '../VAEBladerData/data/airfoil/beziergan_gen'
POLAR_ROOT_DIR = '../VAEBladerData/aerodynamic_label/beziergan_gen'

SEQ_LEN = 200
NUM_CP = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {DEVICE}")


# ============================================================================
# 2. MODEL DEFINITIONS
# ============================================================================
class BSplineLayer(nn.Module):
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


class AirfoilVAE(nn.Module):
    def __init__(self, seq_len=200, num_cp=15, device='cpu'):
        super().__init__()
        self.device = device
        self.num_cp = num_cp
        self.dim_t = 3 + 3
        self.dim_c = 2 + 2
        ENC_FILTERS = 64
        ENC_KERNEL = 3
        DEC_LAYERS = 2
        DEC_NODES = 64

        self.enc_thick = EncoderBlock(seq_len, self.dim_t, ENC_FILTERS, ENC_KERNEL)
        self.enc_camber = EncoderBlock(seq_len, self.dim_c, ENC_FILTERS, ENC_KERNEL)
        self.dec_thick = DecoderBlock(self.dim_t, num_cp - 2, DEC_LAYERS, DEC_NODES)
        self.dec_camber = DecoderBlock(self.dim_c, num_cp, DEC_LAYERS, DEC_NODES)
        self.bspline = BSplineLayer(num_cp, degree=3, num_eval_points=seq_len, device=device)
        self.log_prior = nn.Parameter(torch.tensor([-2.0], device=device))

    def decode_from_latent(self, z_t, z_c):
        cp_t_raw = self.dec_thick(z_t)
        cp_c = self.dec_camber(z_c)
        cp_t_pos = F.softplus(cp_t_raw)
        zeros = torch.zeros(cp_t_pos.shape[0], 1, device=self.device)
        cp_t = torch.cat([zeros, cp_t_pos, zeros], dim=1)
        t_out = self.bspline(cp_t)
        c_out = self.bspline(cp_c)
        return t_out, c_out


class AeroSurrogate(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
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
# 3. INITIALIZATION
# ============================================================================
def load_models():
    print("Loading VAE model...")
    vae = AirfoilVAE(seq_len=SEQ_LEN, num_cp=NUM_CP, device=DEVICE).to(DEVICE)
    try:
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
        print("VAE weights loaded.")
    except Exception as e:
        print(f"WARNING: Failed to load VAE: {e}. Using random weights.")
    vae.eval()

    print("Loading Surrogate model...")
    surrogate = AeroSurrogate(latent_dim=10).to(DEVICE)
    try:
        surrogate.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=DEVICE))
        print("Surrogate weights loaded.")
    except Exception as e:
        print(f"WARNING: Failed to load Surrogate: {e}. Using random weights.")
    surrogate.eval()

    return vae, surrogate


vae_model, surrogate_model = load_models()


# ============================================================================
# 4. INTERFACE LOGIC
# ============================================================================
def update_prediction(t_max, t_pos, t_le, t_free1, t_free2, t_free3,
                      c_max, c_pos, c_free1, c_free2,
                      mach, reynolds, alpha):
    # 1. Prepare Latents (Geometry)
    z_t = torch.tensor([[t_max, t_pos, t_le, t_free1, t_free2, t_free3]], dtype=torch.float32, device=DEVICE)
    z_c = torch.tensor([[c_max, c_pos, c_free1, c_free2]], dtype=torch.float32, device=DEVICE)

    # Concatenate for surrogate: [Batch, 10]
    z_combined = torch.cat([z_t, z_c], dim=1)

    # 2. Prepare Conditions (Physics)
    # Normalize Reynolds: Input 1e6 -> Model sees 1.0
    re_norm = reynolds * 1e-6
    conds = torch.tensor([[mach, re_norm, alpha]], dtype=torch.float32, device=DEVICE)

    # Combine: [Batch, 13]
    surrogate_input = torch.cat([z_combined, conds], dim=1)

    # 3. Inference
    with torch.no_grad():
        # Geometry Decoding
        t_out, c_out = vae_model.decode_from_latent(z_t, z_c)
        t_dist = t_out.cpu().numpy()[0]
        c_dist = c_out.cpu().numpy()[0]

        # Aerodynamic Prediction
        preds = surrogate_model(surrogate_input).cpu().numpy()[0]
        cl, cd, cm = preds[0], preds[1], preds[2]

    # 4. Geometry Plotting
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(t_dist))))
    yu = c_dist + t_dist / 2
    yl = c_dist - t_dist / 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_val, y=yu, mode='lines', name='Upper', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=x_val, y=yl, mode='lines', name='Lower', line=dict(color='firebrick', width=2),
                             fill='tonexty', fillcolor='rgba(200, 200, 200, 0.2)'))
    fig.add_trace(
        go.Scatter(x=x_val, y=c_dist, mode='lines', name='Camber', line=dict(color='black', width=1, dash='dash')))

    fig.update_layout(
        title="Generated Airfoil Geometry",
        xaxis_title="x/c", yaxis_title="y/c",
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[-0.5, 0.5]),
        xaxis=dict(range=[-0.05, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20), height=400, showlegend=True
    )

    # 5. Performance Text Generation
    l_d_ratio = cl / (cd + 1e-6)

    perf_html = f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb;">
        <h3 style="margin-top:0; color: #374151;">Aerodynamic Performance</h3>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div><strong style="color: black;">Lift (CL):</strong> <span style="color: blue;">{cl:.4f}</span></div>
            <div><strong  style="color: black;">Drag (CD):</strong> <span style="color: red;">{cd:.5f}</span></div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <div><strong style="color: black;">Moment (CM) :</strong> <span style="color: purple;">{cm:.4f}</span></div>
            <div><strong style="color: black;">L/D Ratio:</strong> <span style="color: green; font-weight: bold;">{l_d_ratio:.2f}</span></div>
        </div>
    </div>
    """

    return fig, perf_html


# ============================================================================
# 5. GRADIO UI LAYOUT
# ============================================================================
with gr.Blocks(title="Airfoil VAE & Aero Surrogate Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Airfoil Design & Performance Explorer")
    gr.Markdown(
        "Modify geometry (left) and operating conditions (right) to see real-time shape and aerodynamic predictions.")

    with gr.Row():
        # --- LEFT COLUMN: GEOMETRY ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Geometry Controls (Latent Space)")
            with gr.Tab("Thickness"):
                t_max = gr.Slider(-4, 4, value=0, label="Max Thickness", step=0.1)
                t_pos = gr.Slider(-4, 4, value=0, label="Pos Max Thickness", step=0.1)
                t_le = gr.Slider(-4, 4, value=0, label="LE Radius", step=0.1)
                t_f1 = gr.Slider(-4, 4, value=0, label="Free Var 1", step=0.1)
                t_f2 = gr.Slider(-4, 4, value=0, label="Free Var 2", step=0.1)
                t_f3 = gr.Slider(-4, 4, value=0, label="Free Var 3", step=0.1)

            with gr.Tab("Camber"):
                c_max = gr.Slider(-4, 4, value=0, label="Max Camber", step=0.1)
                c_pos = gr.Slider(-4, 4, value=0, label="Pos Max Camber", step=0.1)
                c_f1 = gr.Slider(-4, 4, value=0, label="Free Var 1", step=0.1)
                c_f2 = gr.Slider(-4, 4, value=0, label="Free Var 2", step=0.1)

        # --- RIGHT COLUMN: CONDITIONS & RESULTS ---
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 2. Operating Conditions")
                    mach = gr.Slider(0.0, 1.0, value=0.4, label="Mach Number", step=0.05)
                    alpha = gr.Slider(-10.0, 15.0, value=2.0, label="Alpha (Angle of Attack)", step=0.5)
                    reynolds = gr.Slider(1e5, 5e6, value=1e6, label="Reynolds Number", step=1e5)

                with gr.Column():
                    gr.Markdown("### 3. Prediction")
                    perf_output = gr.HTML(label="Performance")

            plot_output = gr.Plot(label="Airfoil Geometry")

    # Inputs list (Geometry + Physics)
    inputs = [t_max, t_pos, t_le, t_f1, t_f2, t_f3,
              c_max, c_pos, c_f1, c_f2,
              mach, reynolds, alpha]

    # Outputs
    outputs = [plot_output, perf_output]

    # Triggers
    for inp in inputs:
        inp.change(fn=update_prediction, inputs=inputs, outputs=outputs)

    # Initial Load
    demo.load(fn=update_prediction, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()