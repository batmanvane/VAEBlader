import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import os

# --- IMPORT MODELS ---
from createPIVAE_refactor_LE_and_Input import AG_VAE, SEQ_LEN, NUM_CP, DEVICE

# ============================================================================
# 1. CONFIGURATION & MAPPING
# ============================================================================
VAE_MODEL_PATH = './results/model/ag_vae_cps.pth'
SURROGATE_MODEL_PATH = './results/model/aero_surrogate.pth'
PLOT_DIR = './results/optimization_plots'

LATENT_MAP = {
    # Thickness Distribution
    "t_max": 0, "t_pos": 1, "t_rad": 2,
    "t_free_1": 3, "t_free_2": 4, "t_free_3": 5,
    # Camber Distribution
    "c_max": 6, "c_pos": 7, "c_te": 8,
    "c_free_1": 9, "c_free_2": 10, "c_free_3": 11
}


# ============================================================================
# 2. SURROGATE DEFINITION
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

    def _predict_batch(self, z_numpy, mach, reynolds, alphas):
        """
        Runs surrogate for a single geometry (z) across multiple alphas.
        Returns: tensor of shape [n_alphas, 3] -> (CL, CD, CM)
        """
        n_alphas = len(alphas)

        # Repeat Z for batching
        z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device).repeat(n_alphas, 1)

        # Prepare conditions batch
        re_norm = reynolds * 1e-6
        conds = torch.tensor([[mach, re_norm, a] for a in alphas], dtype=torch.float32, device=self.device)

        # Concat & Predict
        surr_in = torch.cat([z_tensor, conds], dim=1)
        preds = self.surrogate(surr_in)  # [n_alphas, 3]
        return preds

    def _decode_geometry(self, z_numpy):
        """ Decodes latent z to coordinates (Upper, Lower) """
        z_tensor = torch.tensor(z_numpy, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_t = z_tensor[:, :6]
        z_c = z_tensor[:, 6:]

        with torch.no_grad():
            t_out, c_out, _, _ = self.vae.decode(z_t, z_c)
            t_dist = t_out.cpu().numpy()[0]
            c_dist = c_out.cpu().numpy()[0]

        x_val = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(t_dist))))
        yu = c_dist + t_dist / 2
        yl = c_dist - t_dist / 2
        return x_val, yu, yl

    def optimize(self, initial_z, design_vars, objective="max_LD", constraints=[], bounds=(-3.0, 3.0),
                 mach=0.4, reynolds=1e6, alphas=[2.0]):
        """
        Robust Multi-Point Optimization.
        Optimizes the AVERAGE performance across the provided 'alphas' list.
        """

        active_indices = [LATENT_MAP[k] for k in design_vars]
        z_current = np.array(initial_z, dtype=np.float32)
        x0 = z_current[active_indices]

        print(f"--- Starting Robust Optimization ---")
        print(f"Objective: {objective} (Averaged over alphas: {alphas})")
        print(f"Design Vars: {design_vars}")

        # Define Objective with Gradient
        def fun(x):
            # Reconstruct Z with Gradient Support
            z_full = torch.tensor(z_current, dtype=torch.float32, device=self.device, requires_grad=False)
            z_active = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)

            # Scatter active into full (Gradient-safe way)
            z_list = []
            ptr = 0
            for i in range(12):
                if i in active_indices:
                    z_list.append(z_active[ptr])
                    ptr += 1
                else:
                    z_list.append(z_full[i])
            z_in = torch.stack(z_list).unsqueeze(0)  # [1, 12]

            # Batch Prediction for all alphas
            # We need to manually repeat z_in inside the graph to keep gradients
            n_alphas = len(alphas)
            z_batch = z_in.repeat(n_alphas, 1)

            re_norm = reynolds * 1e-6
            cond_list = [[mach, re_norm, a] for a in alphas]
            conds = torch.tensor(cond_list, dtype=torch.float32, device=self.device)

            preds = self.surrogate(torch.cat([z_batch, conds], 1))  # [n, 3]

            # Extract Metrics
            cl = preds[:, 0]
            cd = preds[:, 1]
            cm = preds[:, 2]

            # Calculate Robust Metric (Average over angles)
            if objective == "max_LD":
                # Maximize Mean L/D
                ld = cl / (cd + 1e-6)
                val = -1.0 * torch.mean(ld)
            elif objective == "min_CD":
                val = torch.mean(cd) * 100.0
            elif objective == "max_CL":
                val = -1.0 * torch.mean(cl)
            else:
                val = torch.tensor(0.0)

            val.backward()
            return val.item(), z_active.grad.cpu().numpy()

        bnds = [bounds for _ in range(len(active_indices))]

        # Constraints are also averaged for robustness
        proc_constraints = []
        for c in constraints:
            def constr_fun(x, metric=c['metric'], limit=c['value'], sign=c['sign']):
                z_temp = z_current.copy()
                z_temp[active_indices] = x
                with torch.no_grad():
                    # Check average performance across the range
                    p = self._predict_batch(z_temp, mach, reynolds, alphas)  # [n, 3]
                    p_mean = torch.mean(p, dim=0)  # [CL_avg, CD_avg, CM_avg]

                idx = {'CL': 0, 'CD': 1, 'CM': 2}[metric]
                val = p_mean[idx].item()

                if sign == '>':
                    return val - limit
                else:
                    return limit - val

            proc_constraints.append({'type': 'ineq', 'fun': constr_fun})

        start_t = time.time()
        # SLSQP is efficient for this type of constrained problem
        res = minimize(fun, x0, method='SLSQP', jac=True, bounds=bnds, constraints=proc_constraints,
                       options={'disp': True, 'maxiter': 50})

        z_final = z_current.copy()
        z_final[active_indices] = res.x

        print(f"Optimization Finished in {time.time() - start_t:.2f}s")
        print(f"Success: {res.success} | Msg: {res.message}")
        return res, z_final

    def calculate_polar_sweep(self, z_numpy, mach, reynolds):
        """ Generates full polar data for plotting """
        alphas_sweep = np.linspace(-5, 15, 30)

        with torch.no_grad():
            preds = self._predict_batch(z_numpy, mach, reynolds, alphas_sweep).cpu().numpy()

        return alphas_sweep, preds[:, 0], preds[:, 1], preds[:, 2]  # a, cl, cd, cm


# ============================================================================
# PLOTTING UTILS
# ============================================================================
def plot_results(optimizer, z_init, z_opt, mach, reynolds, design_alphas, save_name="robust_opt_result.png"):
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

    # 1. Geometry
    x, yu_i, yl_i = optimizer._decode_geometry(z_init)
    _, yu_o, yl_o = optimizer._decode_geometry(z_opt)

    # 2. Polars (Full Sweep)
    a_i, cl_i, cd_i, cm_i = optimizer.calculate_polar_sweep(z_init, mach, reynolds)
    a_o, cl_o, cd_o, cm_o = optimizer.calculate_polar_sweep(z_opt, mach, reynolds)

    # 3. Design Point Performance (Average)
    with torch.no_grad():
        p_i = optimizer._predict_batch(z_init, mach, reynolds, design_alphas).cpu().numpy()
        p_o = optimizer._predict_batch(z_opt, mach, reynolds, design_alphas).cpu().numpy()

    ld_i = np.mean(p_i[:, 0] / p_i[:, 1])
    ld_o = np.mean(p_o[:, 0] / p_o[:, 1])

    # --- PLOT ---
    fig = plt.figure(figsize=(15, 10))

    # A. Geometry
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(x, yu_i, 'k--', label=f'Initial (Mean L/D={ld_i:.1f})')
    ax1.plot(x, yl_i, 'k--')
    ax1.plot(x, yu_o, 'r-', linewidth=2, label=f'Optimized (Mean L/D={ld_o:.1f})')
    ax1.plot(x, yl_o, 'r-', linewidth=2)
    ax1.set_title(f"Robust Optimization (M={mach}, Re={reynolds:.0e}) | Alphas: {design_alphas}")
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # B. Lift Curve
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.plot(a_i, cl_i, 'k--', label='Initial')
    ax2.plot(a_o, cl_o, 'r-', label='Optimized')
    # Mark design points
    ax2.scatter(design_alphas, p_i[:, 0], c='k', s=50, zorder=5)
    ax2.scatter(design_alphas, p_o[:, 0], c='r', s=50, zorder=5, label='Design Points')
    ax2.set_xlabel("Alpha (deg)")
    ax2.set_ylabel("CL")
    ax2.set_title("Lift Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # C. Drag Polar
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.plot(cd_i, cl_i, 'k--', label='Initial')
    ax3.plot(cd_o, cl_o, 'r-', label='Optimized')
    # Mark design points
    ax3.scatter(p_i[:, 1], p_i[:, 0], c='k', s=50, zorder=5)
    ax3.scatter(p_o[:, 1], p_o[:, 0], c='r', s=50, zorder=5)
    ax3.set_xlabel("CD")
    ax3.set_ylabel("CL")
    ax3.set_title("Drag Polar")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, save_name)
    plt.savefig(path)
    print(f"Plot saved to: {path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def load_models_standalone():
    print(f"Loading VAE from {VAE_MODEL_PATH}...")
    vae = AG_VAE(num_cp=NUM_CP, seq_len=SEQ_LEN, device=DEVICE).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=DEVICE), strict=False)

    print(f"Loading Surrogate from {SURROGATE_MODEL_PATH}...")
    surrogate = AeroSurrogate(latent_dim=12).to(DEVICE)
    surrogate.load_state_dict(torch.load(SURROGATE_MODEL_PATH, map_location=DEVICE))
    return vae, surrogate


if __name__ == "__main__":
    vae, surr = load_models_standalone()
    optimizer = AirfoilOptimizer(vae, surr)

    z_init = np.zeros(12)  # Mean Airfoil

    # Define Conditions
    mach = 0.5
    reynolds = 1e6

    # ROBUST OPTIMIZATION: Optimize average L/D over 3 angles
    alphas = [0.0, 2.0, 4.0]

    # CONSTRAINTS:
    # Relaxed CM constraint to avoid infeasibility
    # We require the AVERAGE Pitching Moment to be > -0.1
    my_constraints = [{'metric': 'CM', 'sign': '>', 'value': -0.1}]

    # Variables to Optimize (Camber Only for higher lift/efficiency)
    design_vars = ["c_max", "c_pos", "t_max", "c_free_1"]

    res, z_opt = optimizer.optimize(
        initial_z=z_init,
        design_vars=design_vars,
        objective="max_LD",
        constraints=my_constraints,
        mach=mach, reynolds=reynolds, alphas=alphas
    )

    # Plot
    plot_results(optimizer, z_init, z_opt, mach, reynolds, alphas)