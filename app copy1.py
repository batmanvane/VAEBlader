# Implements part of DOI: 10.1093/jcde/qwaf002 by Kang et al 2025

import os
# --- FIX: Set Matplotlib to Headless Mode (Prevents server crashes) ---
import matplotlib

matplotlib.use('Agg')
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# --- IMPORTS FROM TRAINING SCRIPT ---
from createPIVAE_refactor_LE_and_Input import AG_VAE, SEQ_LEN, NUM_CP, DEVICE

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
VAE_MODEL_PATH = './results/model/ag_vae_cps.pth'
SURROGATE_MODEL_PATH = './results/model/aero_surrogate.pth'

# Force CPU for App deployment
DEVICE = torch.device('cpu')
print(f"App running on: {DEVICE}")


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
def calculate_polar(z_comb, mach, re_norm, current_alpha):
    """ Sweeps Alpha from -10 to 15 to generate a full polar. """
    alphas = np.linspace(-10, 15, 30)
    batch_z = z_comb.repeat(30, 1)
    batch_conds = torch.tensor(
        [[mach, re_norm, a] for a in alphas],
        dtype=torch.float32, device=DEVICE
    )
    surr_in = torch.cat([batch_z, batch_conds], dim=1)

    with torch.no_grad():
        preds = surrogate_model(surr_in).cpu().numpy()

    return alphas, preds[:, 0], preds[:, 1]


def calculate_contour(z_t, z_c, mach, re_norm, alpha):
    """ Varies T_Max and C_Max to create contour map. """
    resolution = 20
    t_range = np.linspace(-3, 3, resolution)
    c_range = np.linspace(-3, 3, resolution)

    # Base latents
    base_t = z_t.clone()
    base_c = z_c.clone()

    batch_list = []
    for r in range(resolution):
        for c in range(resolution):
            base_t[0, 0] = t_range[r]
            base_c[0, 0] = c_range[c]
            z_comb = torch.cat([base_t, base_c], dim=1)
            cond = torch.tensor([[mach, re_norm, alpha]], dtype=torch.float32, device=DEVICE)
            batch_list.append(torch.cat([z_comb, cond], dim=1))

    batch_in = torch.cat(batch_list, dim=0)
    with torch.no_grad():
        preds = surrogate_model(batch_in).cpu().numpy()

    ld = preds[:, 0] / (preds[:, 1] + 1e-6)
    return t_range, c_range, ld.reshape(resolution, resolution)


def update_analysis(
        t_max, t_pos, t_rad, t_f1, t_f2, t_f3,
        c_max, c_pos, c_te, c_f1, c_f2, c_f3,
        mach, reynolds, alpha, show_advanced
):
    # --- 1. Prepare Inputs ---
    z_t = torch.tensor([[t_max, t_pos, t_rad, t_f1, t_f2, t_f3]], dtype=torch.float32, device=DEVICE)
    z_c = torch.tensor([[c_max, c_pos, c_te, c_f1, c_f2, c_f3]], dtype=torch.float32, device=DEVICE)
    re_norm = reynolds * 1e-6

    # --- 2. Decode Geometry ---
    with torch.no_grad():
        t_out, c_out, _, _ = vae_model.decode(z_t, z_c)
        t_dist = t_out.cpu().numpy()[0]
        c_dist = c_out.cpu().numpy()[0]

    # --- 3. Single Point Prediction ---
    z_comb = torch.cat([z_t, z_c], dim=1)
    conds = torch.tensor([[mach, re_norm, alpha]], dtype=torch.float32, device=DEVICE)
    surr_in = torch.cat([z_comb, conds], dim=1)

    with torch.no_grad():
        preds = surrogate_model(surr_in).cpu().numpy()[0]
        cl_pt, cd_pt, cm_pt = preds[0], preds[1], preds[2]
        ld_pt = cl_pt / (cd_pt + 1e-6)

    # --- 4. Plot Geometry ---
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(t_dist))))
    yu = c_dist + t_dist / 2
    yl = c_dist - t_dist / 2

    fig_geo = go.Figure()
    fig_geo.add_trace(go.Scatter(x=x_val, y=yu, mode='lines', name='Upper', line=dict(color='#1f77b4', width=2)))
    fig_geo.add_trace(
        go.Scatter(x=x_val, y=yl, mode='lines', name='Lower', line=dict(color='#ff7f0e', width=2), fill='tonexty',
                   fillcolor='rgba(200, 200, 200, 0.4)'))
    fig_geo.add_trace(
        go.Scatter(x=x_val, y=c_dist, mode='lines', name='Camber', line=dict(color='black', width=1, dash='dash')))
    fig_geo.update_layout(title="Geometry", yaxis=dict(scaleanchor="x", scaleratio=1, range=[-0.5, 0.5]),
                          xaxis=dict(range=[-0.05, 1.05]), height=350, margin=dict(t=30, b=20, l=20, r=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # --- 5. Metrics Text ---
    html = f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb;">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div><div style="color: #6b7280; font-size: 0.9em;">LIFT (CL)</div><div style="color: #111827; font-size: 1.2em; font-weight: bold;">{cl_pt:.4f}</div></div>
            <div><div style="color: #6b7280; font-size: 0.9em;">DRAG (CD)</div><div style="color: #111827; font-size: 1.2em; font-weight: bold;">{cd_pt:.5f}</div></div>
            <div><div style="color: #6b7280; font-size: 0.9em;">MOMENT (CM)</div><div style="color: #111827; font-size: 1.2em; font-weight: bold;">{cm_pt:.4f}</div></div>
            <div><div style="color: #059669; font-size: 0.9em;">L/D RATIO</div><div style="color: #059669; font-size: 1.4em; font-weight: bold;">{ld_pt:.1f}</div></div>
        </div>
    </div>
    """

    # --- 6. Conditional Advanced Plots ---
    if show_advanced:
        # Polars
        alphas, cls, cds = calculate_polar(z_comb, mach, re_norm, alpha)
        fig_polar = make_subplots(rows=1, cols=2, subplot_titles=("Lift Curve (CL-α)", "Drag Polar (CL-CD)"))
        fig_polar.add_trace(go.Scatter(x=alphas, y=cls, mode='lines', name='Polar'), row=1, col=1)
        fig_polar.add_trace(
            go.Scatter(x=[alpha], y=[cl_pt], mode='markers', marker=dict(color='red', size=10), name='Current'), row=1,
            col=1)
        fig_polar.add_trace(go.Scatter(x=cds, y=cls, mode='lines', name='Polar', showlegend=False), row=1, col=2)
        fig_polar.add_trace(
            go.Scatter(x=[cd_pt], y=[cl_pt], mode='markers', marker=dict(color='red', size=10), showlegend=False),
            row=1, col=2)
        fig_polar.update_layout(height=350, margin=dict(t=50, b=20, l=40, r=20), showlegend=False)
        fig_polar.update_xaxes(title_text="Alpha", row=1, col=1);
        fig_polar.update_yaxes(title_text="CL", row=1, col=1)
        fig_polar.update_xaxes(title_text="CD", row=1, col=2);
        fig_polar.update_yaxes(title_text="CL", row=1, col=2)

        # Contours
        t_rng, c_rng, ld_grid = calculate_contour(z_t, z_c, mach, re_norm, alpha)
        fig_contour = go.Figure()
        fig_contour.add_trace(
            go.Contour(z=ld_grid, x=c_rng, y=t_rng, colorscale='Viridis', colorbar=dict(title='L/D', thickness=15),
                       contours=dict(start=0, end=np.max(ld_grid), size=2, showlabels=True)))
        fig_contour.add_trace(go.Scatter(x=[c_max], y=[t_max], mode='markers',
                                         marker=dict(symbol='x', color='red', size=15, line=dict(width=2)),
                                         name="Current"))
        fig_contour.update_layout(title="Design Space (L/D Map)", xaxis_title="Max Camber (Latent)",
                                  yaxis_title="Max Thickness (Latent)", height=350, margin=dict(t=40, b=40, l=40, r=20),
                                  showlegend=False)

    else:
        # Return Empty Placeholders
        fig_polar = go.Figure().update_layout(title="Enable Advanced Analysis to view Polars", height=350)
        fig_contour = go.Figure().update_layout(title="Enable Advanced Analysis to view Design Space", height=350)

    return fig_geo, html, fig_polar, fig_contour


# ============================================================================
# 5. UI
# ============================================================================
with gr.Blocks(title="AG-VAE Designer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Airfoil Designer (AG-VAE)")

    with gr.Row():
        # CONTROLS
        with gr.Column(scale=1, min_width=300):
            with gr.Accordion("Geometry Controls", open=True):
                gr.Markdown("**Thickness Distribution**")
                t_max = gr.Slider(-3, 3, 0, label="Max Thickness")
                t_pos = gr.Slider(-3, 3, 0, label="Pos Max Thickness")
                t_rad = gr.Slider(-3, 3, 0, label="LE Radius")
                with gr.Accordion("Free Variables (Fine Tuning)", open=False):
                    t_f1 = gr.Slider(-3, 3, 0, label="T-Free 1")
                    t_f2 = gr.Slider(-3, 3, 0, label="T-Free 2")
                    t_f3 = gr.Slider(-3, 3, 0, label="T-Free 3")

                gr.Markdown("**Camber Distribution**")
                c_max = gr.Slider(-3, 3, 0, label="Max Camber")
                c_pos = gr.Slider(-3, 3, 0, label="Pos Max Camber")
                c_te = gr.Slider(-3, 3, 0, label="TE Direction")
                with gr.Accordion("Free Variables (Fine Tuning)", open=False):
                    c_f1 = gr.Slider(-3, 3, 0, label="C-Free 1")
                    c_f2 = gr.Slider(-3, 3, 0, label="C-Free 2")
                    c_f3 = gr.Slider(-3, 3, 0, label="C-Free 3")

            with gr.Accordion("Flight Conditions", open=True):
                mach = gr.Slider(0, 1, 0.4, label="Mach Number")
                reynolds = gr.Slider(1e5, 1e7, 1e6, label="Reynolds Number")
                alpha = gr.Slider(-10, 20, 2, label="Angle of Attack (deg)")

            # TOGGLE
            chk_advanced = gr.Checkbox(label="Show Advanced Analysis (Polars & Design Space)", value=False)

        # VISUALIZATION
        with gr.Column(scale=2):
            # Top Row: Geometry + Metrics
            out_geo = gr.Plot(label="Geometry")
            out_metrics = gr.HTML(label="Metrics")

            # Bottom Row: Analysis
            with gr.Row():
                out_polar = gr.Plot(label="Polars")
                out_contour = gr.Plot(label="Design Space")

    inputs = [t_max, t_pos, t_rad, t_f1, t_f2, t_f3,
              c_max, c_pos, c_te, c_f1, c_f2, c_f3,
              mach, reynolds, alpha, chk_advanced]

    outputs = [out_geo, out_metrics, out_polar, out_contour]

    # Real-time updates
    for x in inputs: x.change(update_analysis, inputs, outputs)
    demo.load(update_analysis, inputs, outputs)

if __name__ == "__main__":
#    demo.launch(ssr_mode=False)
    demo.launch()