# Implements part of DOI: 10.1093/jcde/qwaf002 by Kang et al 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import os

# --- Compatibility Patch for Gradio/HuggingFace Hub ---
# Fixes ImportError: cannot import name 'HfFolder'
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

# --- IMPORTS FROM TRAINING SCRIPT ---
from createPIVAE_refactor_LE_and_Input import AG_VAE, SEQ_LEN, NUM_CP, DEVICE

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
VAE_MODEL_PATH = './results/model/ag_vae_cps.pth'
SURROGATE_MODEL_PATH = './results/model/aero_surrogate.pth'


DEVIE='cpu'  # Force CPU for app --> for hugging face
#print(f"App running on: {DEVICE}")

# ============================================================================
# 2. SURROGATE MODEL
# ============================================================================
class AeroSurrogate(nn.Module):
    def __init__(self, latent_dim=12):
        super().__init__()
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
# 3. INITIALIZATION
# ============================================================================
def load_models():
    print("Loading VAE...")
    vae = AG_VAE(num_cp=NUM_CP, seq_len=SEQ_LEN, device=DEVICE).to(DEVICE)
    try:
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
        print("VAE loaded.")
    except Exception as e:
        print(f"Error loading VAE (Attempting strict=False): {e}")
        try:
            vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE), strict=False)
            print("VAE loaded (strict=False).")
        except Exception as e2:
            print(f"CRITICAL: Failed to load VAE: {e2}")

    vae.eval()

    print("Loading Surrogate...")
    surrogate = AeroSurrogate(latent_dim=12).to(DEVICE)
    try:
        surrogate.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=DEVICE))
        print("Surrogate loaded.")
    except Exception as e:
        print(f"WARNING: Surrogate not found or error: {e}")

    surrogate.eval()
    return vae, surrogate


vae_model, surrogate_model = load_models()


# ============================================================================
# 4. LOGIC
# ============================================================================
def update_prediction(
        t_max, t_pos, t_rad, t_f1, t_f2, t_f3,
        c_max, c_pos, c_te, c_f1, c_f2, c_f3,
        mach, reynolds, alpha
):
    # 1. Inputs
    z_t = torch.tensor([[t_max, t_pos, t_rad, t_f1, t_f2, t_f3]], dtype=torch.float32, device=DEVICE)
    z_c = torch.tensor([[c_max, c_pos, c_te, c_f1, c_f2, c_f3]], dtype=torch.float32, device=DEVICE)

    # 2. Geometry
    with torch.no_grad():
        t_out, c_out, _, _ = vae_model.decode(z_t, z_c)
        t_dist = t_out.cpu().numpy()[0]
        c_dist = c_out.cpu().numpy()[0]

        # Surrogate Input
        z_comb = torch.cat([z_t, z_c], dim=1)
        re_norm = reynolds * 1e-6
        conds = torch.tensor([[mach, re_norm, alpha]], dtype=torch.float32, device=DEVICE)
        surr_in = torch.cat([z_comb, conds], dim=1)

        preds = surrogate_model(surr_in).cpu().numpy()[0]
        cl, cd, cm = preds[0], preds[1], preds[2]

    # 3. Plotting (Fixed Coloring & Legend)
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(t_dist))))
    yu = c_dist + t_dist / 2
    yl = c_dist - t_dist / 2

    fig = go.Figure()

    # Upper Surface
    fig.add_trace(go.Scatter(
        x=x_val, y=yu,
        mode='lines',
        name='Upper Surface',
        line=dict(color='#1f77b4', width=2)  # Professional Blue
    ))

    # Lower Surface with Neutral Fill
    fig.add_trace(go.Scatter(
        x=x_val, y=yl,
        mode='lines',
        name='Lower Surface',
        line=dict(color='#ff7f0e', width=2),  # Professional Orange
        fill='tonexty',
        fillcolor='rgba(200, 200, 200, 0.4)'  # Neutral Light Grey Fill
    ))

    # Camber Line
    fig.add_trace(go.Scatter(
        x=x_val, y=c_dist,
        mode='lines',
        name='Mean Camber',
        line=dict(color='black', width=1, dash='dash')
    ))

    fig.update_layout(
        title="Generated Airfoil (AG-VAE)",
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[-0.5, 0.5]),
        xaxis=dict(range=[-0.05, 1.05]),
        height=400,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)  # Legend on top-right, horizontal
    )

    # 4. Text
    ld = cl / (cd + 1e-6)
    html = f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb;">
        <h3 style="margin-top: 0; color: #111827 !important;">Performance Prediction</h3>
        <div style="display: flex; gap: 20px; font-size: 1.1em; margin-bottom: 10px;">
            <div><strong style="color: #111827 !important;">CL:</strong> <span style="color: #1d4ed8 !important;">{cl:.4f}</span></div>
            <div><strong style="color: #111827 !important;">CD:</strong> <span style="color: #b91c1c !important;">{cd:.5f}</span></div>
            <div><strong style="color: #111827 !important;">CM:</strong> <span style="color: #7e22ce !important;">{cm:.4f}</span></div>
        </div>
        <div style="font-size: 1.3em; font-weight: bold; color: #047857 !important;">
            L/D: {ld:.2f}
        </div>
    </div>
    """
    return fig, html


# ============================================================================
# 5. UI
# ============================================================================
with gr.Blocks(title="AG-VAE Explorer") as demo:
    gr.Markdown("## Airfoil Design Explorer (AG-VAE)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Thickness")
            t_max = gr.Slider(-3, 3, 0, label="Max Thickness")
            t_pos = gr.Slider(-3, 3, 0, label="Pos Max Thickness")
            t_rad = gr.Slider(-3, 3, 0, label="LE Radius")
            t_f1 = gr.Slider(-3, 3, 0, label="T-Free 1")
            t_f2 = gr.Slider(-3, 3, 0, label="T-Free 2")
            t_f3 = gr.Slider(-3, 3, 0, label="T-Free 3")

            gr.Markdown("### Camber")
            c_max = gr.Slider(-3, 3, 0, label="Max Camber")
            c_pos = gr.Slider(-3, 3, 0, label="Pos Max Camber")
            c_te = gr.Slider(-3, 3, 0, label="TE Direction")
            c_f1 = gr.Slider(-3, 3, 0, label="C-Free 1")
            c_f2 = gr.Slider(-3, 3, 0, label="C-Free 2")
            c_f3 = gr.Slider(-3, 3, 0, label="C-Free 3")

        with gr.Column(scale=2):
            gr.Markdown("### Operating Conditions")
            with gr.Row():
                mach = gr.Slider(0, 1, 0.4, label="Mach")
                alpha = gr.Slider(-10, 20, 2, label="Alpha")
                reynolds = gr.Slider(1e5, 1e7, 1e6, label="Reynolds")

            gr.Markdown("### Analysis")
            out_plot = gr.Plot()
            out_txt = gr.HTML()

    inputs = [t_max, t_pos, t_rad, t_f1, t_f2, t_f3,
              c_max, c_pos, c_te, c_f1, c_f2, c_f3,
              mach, reynolds, alpha]

    for x in inputs: x.change(update_prediction, inputs, [out_plot, out_txt])
    demo.load(update_prediction, inputs, [out_plot, out_txt])

if __name__ == "__main__":
    demo.launch()