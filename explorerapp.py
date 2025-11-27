import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import os

# --- Compatibility Patch for Gradio/HuggingFace Hub ---
# Fixes ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
# This happens when using older Gradio versions with newer huggingface_hub versions.
try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "HfFolder"):
        # Mock the missing class that older Gradio versions expect
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
MODEL_PATH = './results/model/airfoil_vae.pth'
SEQ_LEN = 200
NUM_CP = 15

# Check device (Force CPU for web app stability usually, but CUDA works if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {DEVICE}")


# ============================================================================
# 2. MODEL DEFINITION (Must match training script exactly)
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

        # Architecture Constants
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

    def forward(self, x):
        pass


# ============================================================================
# 3. INITIALIZATION
# ============================================================================
def load_model():
    print("Loading model...")
    model = AirfoilVAE(seq_len=SEQ_LEN, num_cp=NUM_CP, device=DEVICE).to(DEVICE)
    try:
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Model weights loaded.")
        else:
            print(f"WARNING: '{MODEL_PATH}' not found. Using random weights.")
    except Exception as e:
        print(f"Error loading model: {e}")
    model.eval()
    return model


model = load_model()


# ============================================================================
# 4. INTERFACE LOGIC
# ============================================================================
def update_geometry(t_max, t_pos, t_le, t_free1, t_free2, t_free3,
                    c_max, c_pos, c_free1, c_free2):
    # Construct Latent Vectors
    # Thickness: [Phys(3) + Free(3)]
    z_t = torch.tensor([[t_max, t_pos, t_le, t_free1, t_free2, t_free3]],
                       dtype=torch.float32, device=DEVICE)

    # Camber: [Phys(2) + Free(2)]
    z_c = torch.tensor([[c_max, c_pos, c_free1, c_free2]],
                       dtype=torch.float32, device=DEVICE)

    # Decode
    with torch.no_grad():
        t_out, c_out = model.decode_from_latent(z_t, z_c)
        t_dist = t_out.cpu().numpy()[0]
        c_dist = c_out.cpu().numpy()[0]

    # Calculate Coordinates
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(t_dist))))
    yu = c_dist + t_dist / 2
    yl = c_dist - t_dist / 2

    # Create Plotly Figure
    fig = go.Figure()

    # Upper Surface
    fig.add_trace(go.Scatter(
        x=x_val, y=yu, mode='lines', name='Upper',
        line=dict(color='royalblue', width=2)
    ))

    # Lower Surface
    fig.add_trace(go.Scatter(
        x=x_val, y=yl, mode='lines', name='Lower',
        line=dict(color='firebrick', width=2),
        fill='tonexty',  # Fill area between traces
        fillcolor='rgba(200, 200, 200, 0.2)'
    ))

    # Camber Line
    fig.add_trace(go.Scatter(
        x=x_val, y=c_dist, mode='lines', name='Camber',
        line=dict(color='black', width=1, dash='dash')
    ))

    # Layout styling
    fig.update_layout(
        title="Generated Airfoil Geometry",
        xaxis_title="x/c",
        yaxis_title="y/c",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            range=[-0.5, 0.5]
        ),
        xaxis=dict(range=[-0.05, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
        showlegend=True
    )

    return fig


# ============================================================================
# 5. GRADIO UI LAYOUT
# ============================================================================
with gr.Blocks(title="Airfoil VAE Explorer") as demo:
    gr.Markdown("# Interactive Airfoil VAE Explorer")
    gr.Markdown(
        "Adjust the sliders to explore the latent space. The model runs in the background and updates the geometry.")

    with gr.Row():
        # --- LEFT COLUMN: CONTROLS ---
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Thickness Branch")
                gr.Markdown("**Physical Properties**")
                t_max = gr.Slider(-4, 4, value=0, label="Max Thickness", step=0.1)
                t_pos = gr.Slider(-4, 4, value=0, label="Pos Max Thickness", step=0.1)
                t_le = gr.Slider(-4, 4, value=0, label="LE Radius", step=0.1)

                gr.Markdown("**Free Variables**")
                t_f1 = gr.Slider(-4, 4, value=0, label="Free 1", step=0.1)
                t_f2 = gr.Slider(-4, 4, value=0, label="Free 2", step=0.1)
                t_f3 = gr.Slider(-4, 4, value=0, label="Free 3", step=0.1)

            with gr.Group():
                gr.Markdown("### Camber Branch")
                gr.Markdown("**Physical Properties**")
                c_max = gr.Slider(-4, 4, value=0, label="Max Camber", step=0.1)
                c_pos = gr.Slider(-4, 4, value=0, label="Pos Max Camber", step=0.1)

                gr.Markdown("**Free Variables**")
                c_f1 = gr.Slider(-4, 4, value=0, label="Free 1", step=0.1)
                c_f2 = gr.Slider(-4, 4, value=0, label="Free 2", step=0.1)

        # --- RIGHT COLUMN: PLOT ---
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Airfoil Geometry")

    # Inputs list must match function arguments
    inputs = [t_max, t_pos, t_le, t_f1, t_f2, t_f3, c_max, c_pos, c_f1, c_f2]

    # Event listeners
    # Trigger update on any slider change
    for slider in inputs:
        slider.change(fn=update_geometry, inputs=inputs, outputs=plot_output)

    # Initial load
    demo.load(fn=update_geometry, inputs=inputs, outputs=plot_output)

if __name__ == "__main__":
    demo.launch()