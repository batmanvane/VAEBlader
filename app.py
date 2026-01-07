# Implements part of DOI: 10.1093/jcde/qwaf002 by Kang et al 2025

import os
# --- FIX: Set Matplotlib to Headless Mode (Prevents server crashes) ---
import matplotlib

matplotlib.use('Agg')
# ----------------------------------------------------------------------

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

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from scipy.optimize import minimize
import time
import itertools



# --- IMPORTS FROM TRAINING SCRIPT ---
from createPIVAE_refactor_LE_and_Input import AG_VAE, SEQ_LEN, NUM_CP, DEVICE

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
VAE_MODEL_PATH = './results/model/ag_vae_cps.pth'
SURROGATE_MODEL_PATH = './results/model/aero_surrogate.pth'
PLOT_DIR = './results/plots'

LATENT_MAP = {
    "t_max": 0, "t_pos": 1, "t_rad": 2,
    "t_free_1": 3, "t_free_2": 4, "t_free_3": 5,
    "c_max": 6, "c_pos": 7, "c_te": 8,
    "c_free_1": 9, "c_free_2": 10, "c_free_3": 11
}

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
# 3. OPTIMIZER CLASS
# ============================================================================
class AirfoilOptimizer:
    def __init__(self, vae_model, surrogate_model, device=DEVICE):
        self.vae = vae_model
        self.surrogate = surrogate_model
        self.device = device
        self.vae.eval()
        self.surrogate.eval()

    def _predict_batch_grid(self, z_numpy, machs, reynolds, alphas):
        combinations = list(itertools.product(machs, alphas))
        n_total = len(combinations)
        z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device).repeat(n_total, 1)
        re_norm = reynolds * 1e-6
        cond_list = [[m, re_norm, a] for m, a in combinations]
        conds = torch.tensor(cond_list, dtype=torch.float32, device=self.device)
        surr_in = torch.cat([z_tensor, conds], dim=1)
        preds = self.surrogate(surr_in)
        return preds

    def _decode_geometry(self, z_numpy):
        z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            t_out, c_out, _, _ = self.vae.decode(z_tensor[:, :6], z_tensor[:, 6:])
            t_dist = t_out.cpu().numpy()[0]
            c_dist = c_out.cpu().numpy()[0]

        t_dist[-1], c_dist[-1] = 0.0, 0.0
        x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(t_dist))))
        yu = c_dist + t_dist / 2
        yl = c_dist - t_dist / 2
        return x_val, yu, yl

    def optimize(self, initial_z, design_vars, objective="max_LD", constraints=[], bounds=(-3.0, 3.0),
                 machs=[0.5], reynolds=1e6, alphas=[2.0], log_callback=None):

        active_indices = [LATENT_MAP[k] for k in design_vars]
        z_current = np.array(initial_z, dtype=np.float32)
        x0 = z_current[active_indices]

        combinations = list(itertools.product(machs, alphas))
        re_norm = reynolds * 1e-6
        cond_list = [[m, re_norm, a] for m, a in combinations]
        conds_tensor = torch.tensor(cond_list, dtype=torch.float32, device=self.device)
        n_total = len(cond_list)

        iter_count = [0]

        def fun(x):
            z_full = torch.tensor(z_current, dtype=torch.float32, device=self.device, requires_grad=False)
            z_active = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)

            z_list = []
            ptr = 0
            for i in range(12):
                if i in active_indices:
                    z_list.append(z_active[ptr])
                    ptr += 1
                else:
                    z_list.append(z_full[i])
            z_in = torch.stack(z_list).unsqueeze(0)

            z_batch = z_in.repeat(n_total, 1)
            preds = self.surrogate(torch.cat([z_batch, conds_tensor], 1))
            cl, cd = preds[:, 0], preds[:, 1]

            if objective == "max_LD":
                ld = cl / (cd + 1e-6)
                metric = torch.mean(ld)
                val = -1.0 * metric
            elif objective == "min_CD":
                metric = torch.mean(cd)
                val = metric * 100.0
            elif objective == "max_CL":
                metric = torch.mean(cl)
                val = -1.0 * metric
            else:
                val = torch.tensor(0.0)

            val.backward()

            if log_callback and iter_count[0] % 5 == 0:
                log_callback(f"Iter {iter_count[0]}: Objective ({objective}) = {metric.item():.4f}")
            iter_count[0] += 1

            return val.item(), z_active.grad.cpu().numpy()

        bnds = [bounds for _ in range(len(active_indices))]

        proc_constraints = []
        for c in constraints:
            def constr_fun(x, metric=c['metric'], limit=c['value'], sign=c['sign']):
                z_temp = z_current.copy()
                z_temp[active_indices] = x
                with torch.no_grad():
                    p = self._predict_batch_grid(z_temp, machs, reynolds, alphas)
                    p_mean = torch.mean(p, dim=0)
                idx = {'CL': 0, 'CD': 1, 'CM': 2}[metric]
                val = p_mean[idx].item()
                if sign == '>':
                    return val - limit
                else:
                    return limit - val

            proc_constraints.append({'type': 'ineq', 'fun': constr_fun})

        start_t = time.time()
        res = minimize(fun, x0, method='SLSQP', jac=True, bounds=bnds, constraints=proc_constraints,
                       options={'disp': False, 'maxiter': 50})

        z_final = z_current.copy()
        z_final[active_indices] = res.x

        return res, z_final

    def calculate_polar_sweep(self, z_numpy, mach, reynolds):
        alphas_sweep = np.linspace(-5, 15, 30)
        with torch.no_grad():
            preds = self._predict_batch_grid(z_numpy, [mach], reynolds, alphas_sweep).cpu().numpy()
        return alphas_sweep, preds[:, 0], preds[:, 1]


# ============================================================================
# 4. INITIALIZATION
# ============================================================================
def load_models():
    print("Loading VAE...")
    vae = AG_VAE(num_cp=NUM_CP, seq_len=SEQ_LEN, device=DEVICE).to(DEVICE)
    try:
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE))
    except Exception:
        vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE), strict=False)
    vae.eval()

    print("Loading Surrogate...")
    surrogate = AeroSurrogate(latent_dim=12).to(DEVICE)
    surrogate.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=DEVICE))
    surrogate.eval()
    return vae, surrogate


vae_model, surrogate_model = load_models()
optimizer = AirfoilOptimizer(vae_model, surrogate_model)


# ============================================================================
# 5. UI LOGIC (DESIGNER & OPTIMIZER)
# ============================================================================
def run_optimization_ui(
        design_vars, objective,
        constr_metric, constr_sign, constr_val,
        mach, reynolds, alphas_str,
        bound_min, bound_max
):
    try:
        alphas = [float(x.strip()) for x in alphas_str.split(',')]
    except:
        return None, "Error parsing Alphas", None, ""

    z_init = np.zeros(12)
    logs = []

    def logger(msg):
        logs.append(msg)

    constraints = []
    if constr_metric and constr_sign:
        constraints.append({'metric': constr_metric, 'sign': constr_sign, 'value': constr_val})

    logger("--- Starting Optimization ---")
    start_t = time.time()
    res, z_opt = optimizer.optimize(
        initial_z=z_init, design_vars=design_vars, objective=objective, constraints=constraints,
        bounds=(bound_min, bound_max), machs=[mach], reynolds=reynolds, alphas=alphas,
        log_callback=logger
    )

    logger(f"Finished in {time.time() - start_t:.2f}s")
    logger(f"Success: {res.success}")
    logger(f"Message: {res.message}")

    with torch.no_grad():
        p_i = optimizer._predict_batch_grid(z_init, [mach], reynolds, alphas).cpu().numpy()
        p_o = optimizer._predict_batch_grid(z_opt, [mach], reynolds, alphas).cpu().numpy()

    avg_i = np.mean(p_i, axis=0)
    avg_o = np.mean(p_o, axis=0)
    ld_i = np.mean(p_i[:, 0] / p_i[:, 1])
    ld_o = np.mean(p_o[:, 0] / p_o[:, 1])
    gain = (ld_o - ld_i) / ld_i * 100

    # --- RESULTS TABLE (Bulletproof Colors) ---
    results_html = f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb;">
        <h3 style="margin-top:0; color:#111827 !important;">Optimization Results (Average over Alphas)</h3>
        <table style="width:100%; text-align:center; font-size:1.1em; color:#111827 !important; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #d1d5db;">
                <th style="padding: 8px; color:#6b7280 !important; font-weight:normal;">Metric</th>
                <th style="padding: 8px; color:#6b7280 !important; font-weight:normal;">Initial</th>
                <th style="padding: 8px; color:#6b7280 !important; font-weight:normal;">Optimized</th>
                <th style="padding: 8px; color:#6b7280 !important; font-weight:normal;">Change</th>
            </tr>
            <tr>
                <td style="padding: 8px; color:#111827 !important; font-weight:bold;">L/D</td>
                <td style="padding: 8px; color:#111827 !important;">{ld_i:.1f}</td>
                <td style="padding: 8px; color:#059669 !important; font-weight:bold;">{ld_o:.1f}</td>
                <td style="padding: 8px; color:#059669 !important;">+{gain:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 8px; color:#111827 !important; font-weight:bold;">CL</td>
                <td style="padding: 8px; color:#111827 !important;">{avg_i[0]:.3f}</td>
                <td style="padding: 8px; color:#111827 !important;">{avg_o[0]:.3f}</td>
                <td></td>
            </tr>
            <tr>
                <td style="padding: 8px; color:#111827 !important; font-weight:bold;">CD</td>
                <td style="padding: 8px; color:#111827 !important;">{avg_i[1]:.4f}</td>
                <td style="padding: 8px; color:#111827 !important;">{avg_o[1]:.4f}</td>
                <td></td>
            </tr>
            <tr>
                <td style="padding: 8px; color:#111827 !important; font-weight:bold;">CM</td>
                <td style="padding: 8px; color:#111827 !important;">{avg_i[2]:.3f}</td>
                <td style="padding: 8px; color:#111827 !important;">{avg_o[2]:.3f}</td>
                <td></td>
            </tr>
        </table>
    </div>
    """

    # PLOTS
    x, yu_i, yl_i = optimizer._decode_geometry(z_init)
    _, yu_o, yl_o = optimizer._decode_geometry(z_opt)

    fig_opt = go.Figure()
    fig_opt.add_trace(go.Scatter(x=x, y=yu_i, mode='lines', line=dict(color='gray', dash='dash'), name=f'Initial'))
    fig_opt.add_trace(go.Scatter(x=x, y=yl_i, mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))
    fig_opt.add_trace(go.Scatter(x=x, y=yu_o, mode='lines', line=dict(color='#dc2626', width=3), name=f'Optimized'))
    fig_opt.add_trace(go.Scatter(x=x, y=yl_o, mode='lines', line=dict(color='#dc2626', width=3), showlegend=False))
    fig_opt.update_layout(title="Geometry Comparison", yaxis=dict(scaleanchor="x", scaleratio=1), height=400)

    a_i, cl_i, cd_i = optimizer.calculate_polar_sweep(z_init, mach, reynolds)
    a_o, cl_o, cd_o = optimizer.calculate_polar_sweep(z_opt, mach, reynolds)

    fig_pol = make_subplots(rows=1, cols=2, subplot_titles=("Lift Curve", "Drag Polar"))
    fig_pol.add_trace(go.Scatter(x=a_i, y=cl_i, mode='lines', line=dict(color='gray', dash='dash'), name='Initial'),
                      row=1, col=1)
    fig_pol.add_trace(go.Scatter(x=a_o, y=cl_o, mode='lines', line=dict(color='red', width=2), name='Optimized'), row=1,
                      col=1)
    fig_pol.add_trace(
        go.Scatter(x=alphas, y=p_o[:, 0], mode='markers', marker=dict(color='black', size=8), name='Design Pts'), row=1,
        col=1)
    fig_pol.add_trace(go.Scatter(x=cd_i, y=cl_i, mode='lines', line=dict(color='gray', dash='dash'), showlegend=False),
                      row=1, col=2)
    fig_pol.add_trace(go.Scatter(x=cd_o, y=cl_o, mode='lines', line=dict(color='red', width=2), showlegend=False),
                      row=1, col=2)
    fig_pol.add_trace(
        go.Scatter(x=p_o[:, 1], y=p_o[:, 0], mode='markers', marker=dict(color='black', size=8), showlegend=False),
        row=1, col=2)
    fig_pol.update_layout(height=400)

    return fig_opt, "\n".join(logs), fig_pol, results_html


# ============================================================================
# 6. DESIGNER LOGIC
# ============================================================================
def calculate_polar(z_comb, mach, re_norm, current_alpha):
    alphas = np.linspace(-10, 15, 30)
    batch_z = z_comb.repeat(30, 1)
    batch_conds = torch.tensor([[mach, re_norm, a] for a in alphas], dtype=torch.float32, device=DEVICE)
    surr_in = torch.cat([batch_z, batch_conds], dim=1)
    with torch.no_grad():
        preds = surrogate_model(surr_in).cpu().numpy()
    return alphas, preds[:, 0], preds[:, 1]


def calculate_contour(z_t, z_c, mach, re_norm, alpha):
    resolution = 20
    t_range = np.linspace(-3, 3, resolution)
    c_range = np.linspace(-3, 3, resolution)
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


def update_analysis(t_max, t_pos, t_rad, t_f1, t_f2, t_f3, c_max, c_pos, c_te, c_f1, c_f2, c_f3, mach, reynolds, alpha,
                    show_advanced):
    z_t = torch.tensor([[t_max, t_pos, t_rad, t_f1, t_f2, t_f3]], dtype=torch.float32, device=DEVICE)
    z_c = torch.tensor([[c_max, c_pos, c_te, c_f1, c_f2, c_f3]], dtype=torch.float32, device=DEVICE)
    re_norm = reynolds * 1e-6
    with torch.no_grad():
        t_out, c_out, _, _ = vae_model.decode(z_t, z_c)
        t_dist = t_out.cpu().numpy()[0]
        c_dist = c_out.cpu().numpy()[0]
    t_dist[-1], c_dist[-1] = 0.0, 0.0
    z_comb = torch.cat([z_t, z_c], dim=1)
    conds = torch.tensor([[mach, re_norm, alpha]], dtype=torch.float32, device=DEVICE)
    surr_in = torch.cat([z_comb, conds], dim=1)
    with torch.no_grad():
        preds = surrogate_model(surr_in).cpu().numpy()[0]
        cl_pt, cd_pt, cm_pt = preds[0], preds[1], preds[2]
        ld_pt = cl_pt / (cd_pt + 1e-6)
    x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(t_dist))))
    yu = c_dist + t_dist / 2
    yl = c_dist - t_dist / 2
    x_poly = np.concatenate([x_val, x_val[::-1]])
    y_poly = np.concatenate([yu, yl[::-1]])
    fig_geo = go.Figure()
    fig_geo.add_trace(go.Scatter(x=x_poly, y=y_poly, mode='lines', fill='toself', fillcolor='rgba(200, 200, 200, 0.4)',
                                 line=dict(color='#1f77b4', width=2), name='Airfoil'))
    fig_geo.add_trace(
        go.Scatter(x=x_val, y=c_dist, mode='lines', name='Camber', line=dict(color='black', width=1, dash='dash')))
    fig_geo.update_layout(title="Geometry", yaxis=dict(scaleanchor="x", scaleratio=1, range=[-0.5, 0.5]),
                          xaxis=dict(range=[-0.05, 1.05]), height=350, margin=dict(t=30, b=20, l=20, r=20),
                          legend=dict(orientation="h", y=1.02, x=1))

    html = f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb;">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="color: #6b7280 !important; font-size: 0.9em;">LIFT (CL)</div>
                <div style="color: #111827 !important; font-size: 1.2em; font-weight: bold;">{cl_pt:.4f}</div>
            </div>
            <div>
                <div style="color: #6b7280 !important; font-size: 0.9em;">DRAG (CD)</div>
                <div style="color: #111827 !important; font-size: 1.2em; font-weight: bold;">{cd_pt:.5f}</div>
            </div>
            <div>
                <div style="color: #6b7280 !important; font-size: 0.9em;">MOMENT (CM)</div>
                <div style="color: #111827 !important; font-size: 1.2em; font-weight: bold;">{cm_pt:.4f}</div>
            </div>
            <div>
                <div style="color: #059669 !important; font-size: 0.9em;">L/D RATIO</div>
                <div style="color: #059669 !important; font-size: 1.4em; font-weight: bold;">{ld_pt:.1f}</div>
            </div>
        </div>
    </div>
    """
    if show_advanced:
        alphas, cls, cds = calculate_polar(z_comb, mach, re_norm, alpha)
        fig_polar = make_subplots(rows=1, cols=2, subplot_titles=("Lift Curve", "Drag Polar"))
        fig_polar.add_trace(go.Scatter(x=alphas, y=cls, mode='lines', name='Polar'), row=1, col=1)
        fig_polar.add_trace(
            go.Scatter(x=[alpha], y=[cl_pt], mode='markers', marker=dict(color='red', size=10), name='Current'), row=1,
            col=1)
        fig_polar.add_trace(go.Scatter(x=cds, y=cls, mode='lines', name='Polar', showlegend=False), row=1, col=2)
        fig_polar.add_trace(
            go.Scatter(x=[cd_pt], y=[cl_pt], mode='markers', marker=dict(color='red', size=10), showlegend=False),
            row=1, col=2)
        fig_polar.update_layout(height=350, margin=dict(t=50, b=20, l=40, r=20), showlegend=False)
        t_rng, c_rng, ld_grid = calculate_contour(z_t, z_c, mach, re_norm, alpha)
        fig_contour = go.Figure()
        fig_contour.add_trace(go.Contour(z=ld_grid, x=c_rng, y=t_rng, colorscale='Viridis',
                                         contours=dict(start=0, end=np.max(ld_grid), size=2, showlabels=True)))
        fig_contour.add_trace(go.Scatter(x=[c_max], y=[t_max], mode='markers',
                                         marker=dict(symbol='x', color='red', size=15, line=dict(width=2)),
                                         name="Current"))
        fig_contour.update_layout(title="Design Space (L/D Map)", xaxis_title="Max Camber", yaxis_title="Max Thickness",
                                  height=350, margin=dict(t=40, b=40, l=40, r=20), showlegend=False)
    else:
        fig_polar = go.Figure().update_layout(title="Enable Advanced Analysis", height=350)
        fig_contour = go.Figure().update_layout(title="Enable Advanced Analysis", height=350)
    return fig_geo, html, fig_polar, fig_contour


# ============================================================================
# 7. UI LAYOUT
# ============================================================================
with gr.Blocks(title="AG-VAE Designer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Airfoil Designer (AG-VAE)")

    with gr.Tabs():
        # --- TAB 1: DESIGNER ---
        with gr.Tab("Designer"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Accordion("Geometry Controls", open=True):
                        gr.Markdown("**Thickness**")
                        t_max = gr.Slider(-3, 3, 0, label="Max Thickness")
                        t_pos = gr.Slider(-3, 3, 0, label="Pos Max Thickness")
                        t_rad = gr.Slider(-3, 3, 0, label="LE Radius")
                        with gr.Accordion("Free Variables", open=False):
                            t_f1 = gr.Slider(-3, 3, 0, label="T-Free 1")
                            t_f2 = gr.Slider(-3, 3, 0, label="T-Free 2")
                            t_f3 = gr.Slider(-3, 3, 0, label="T-Free 3")
                        gr.Markdown("**Camber**")
                        c_max = gr.Slider(-3, 3, 0, label="Max Camber")
                        c_pos = gr.Slider(-3, 3, 0, label="Pos Max Camber")
                        c_te = gr.Slider(-3, 3, 0, label="TE Direction")
                        with gr.Accordion("Free Variables", open=False):
                            c_f1 = gr.Slider(-3, 3, 0, label="C-Free 1")
                            c_f2 = gr.Slider(-3, 3, 0, label="C-Free 2")
                            c_f3 = gr.Slider(-3, 3, 0, label="C-Free 3")
                    with gr.Accordion("Flight Conditions", open=True):
                        mach = gr.Slider(0, 1, 0.4, label="Mach Number")
                        reynolds = gr.Slider(1e5, 1e7, 1e6, label="Reynolds Number")
                        alpha = gr.Slider(-10, 20, 2, label="Angle of Attack (deg)")
                    chk_advanced = gr.Checkbox(label="Show Advanced Analysis", value=False)
                with gr.Column(scale=2):
                    out_geo = gr.Plot(label="Geometry")
                    out_metrics = gr.HTML(label="Metrics")
                    with gr.Row():
                        out_polar = gr.Plot(label="Polars")
                        out_contour = gr.Plot(label="Design Space")
            inputs_1 = [t_max, t_pos, t_rad, t_f1, t_f2, t_f3, c_max, c_pos, c_te, c_f1, c_f2, c_f3, mach, reynolds,
                        alpha, chk_advanced]
            outputs_1 = [out_geo, out_metrics, out_polar, out_contour]
            for x in inputs_1: x.change(update_analysis, inputs_1, outputs_1)
            demo.load(update_analysis, inputs_1, outputs_1)

        # --- TAB 2: OPTIMIZER ---
        with gr.Tab("Optimizer"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Optimization Settings")
                    opt_obj = gr.Dropdown(["max_LD", "min_CD", "max_CL"], value="max_LD", label="Objective")
                    opt_vars = gr.CheckboxGroup(
                        choices=list(LATENT_MAP.keys()),
                        value=["t_max", "c_max", "c_free_1"],  # Default Selection
                        label="Design Variables"
                    )
                    with gr.Row():
                        opt_mach = gr.Number(value=0.5, label="Mach")
                        opt_re = gr.Number(value=1e6, label="Reynolds")
                    opt_alphas = gr.Textbox(value="0, 2, 4", label="Robustness Alphas (comma sep)")

                    gr.Markdown("**Constraint (Average over Alphas)**")
                    with gr.Accordion("Help: What are these constraints?", open=False):
                        gr.Markdown("""
                        - **CM (Pitching Moment):** Tendency to rotate. > -0.1 generally stable.
                        - **CD (Drag Coefficient):** Air resistance. Lower is better.
                        - **CL (Lift Coefficient):** Upward force. Higher is better.
                        """)
                    with gr.Row():
                        opt_c_met = gr.Dropdown(["CM", "CL", "CD"], value="CM", label="Metric")
                        opt_c_sign = gr.Dropdown([">", "<"], value=">", label="Sign")
                        opt_c_val = gr.Number(value=-0.1, label="Value")
                    with gr.Row():
                        opt_b_min = gr.Number(value=-3.0, label="Min Bound")
                        opt_b_max = gr.Number(value=3.0, label="Max Bound")
                    btn_opt = gr.Button("Run Optimization", variant="primary")
                    out_status = gr.TextArea(label="Optimization Log", lines=6)
                with gr.Column(scale=2):
                    out_results = gr.HTML(label="Results")
                    out_opt_geo = gr.Plot(label="Geometry Optimization")
                    out_opt_pol = gr.Plot(label="Performance Verification")
            btn_opt.click(
                run_optimization_ui,
                inputs=[opt_vars, opt_obj, opt_c_met, opt_c_sign, opt_c_val, opt_mach, opt_re, opt_alphas, opt_b_min,
                        opt_b_max],
                outputs=[out_opt_geo, out_status, out_opt_pol, out_results]
            )

        # --- TAB 3: DIAGNOSTICS ---
        with gr.Tab("Model Diagnostics"):
            gr.Markdown("### VAE Training Performance")
            with gr.Row():
                with gr.Column():
                    gr.Image(value=f"{PLOT_DIR}/correlation.png", label="Latent-Physics Correlation")
                    gr.Image(value=f"{PLOT_DIR}/pivae_alignment_analysis.png", label="Detailed Alignment")
                with gr.Column():
                    gr.Image(value=f"{PLOT_DIR}/traversals.png", label="Latent Traversals")
                    gr.Image(value=f"{PLOT_DIR}/reconstruction.png", label="Reconstruction Quality")

if __name__ == "__main__":
    demo.launch()