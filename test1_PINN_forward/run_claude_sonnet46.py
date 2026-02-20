"""
PINNs for High-Frequency Heat Equation (Forward Problem)
=========================================================
PDE:    u_t - alpha * u_xx = 0,    (x,t) in [0,1] x [0,1]
IC:     u(x,0) = sin(n*pi*x)
BC:     u(0,t) = u(1,t) = 0
Exact:  u(x,t) = sin(n*pi*x) * exp(-alpha*(n*pi)^2 * t)

Training Strategy:
  Phase 1 — Adam   : 20000 iterations  (fast global descent)
  Phase 2 — L-BFGS : 10000 iterations  (high-precision refinement)

GPU is required; falls back to CPU with a warning if unavailable.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════
SEED            = 42
ALPHA           = 0.01          # thermal diffusivity
N_FREQ          = 5             # spatial frequency n (increase for higher freq)
N_RES           = 5000          # PDE collocation points
N_IC            = 300           # initial condition points
N_BC            = 300           # boundary points (each side)
N_TEST          = 100           # test grid per dimension -> 100x100

# Adam phase
ADAM_EPOCHS     = 20000
ADAM_LR         = 1e-3
ADAM_LR_STEP    = 5000
ADAM_LR_GAMMA   = 0.5
ADAM_LOG_EVERY  = 500

# L-BFGS phase
LBFGS_MAX_ITER  = 10000         # max iterations for L-BFGS
LBFGS_LOG_EVERY = 500           # log every N L-BFGS steps

# Network
HIDDEN_DIM      = 64
N_LAYERS        = 6             # number of hidden layers

SAVE_DIR        = "output/claude_sonnet46"

# ── Device ────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[Warning] CUDA not available — running on CPU. GPU is recommended.")

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════
#  Exact Solution
# ══════════════════════════════════════════════════════════════════
def exact_np(x, t):
    """u(x,t) = sin(n*pi*x)*exp(-alpha*(n*pi)^2*t)"""
    return np.sin(N_FREQ * np.pi * x) * np.exp(-ALPHA * (N_FREQ * np.pi) ** 2 * t)

# PDE verification:
#   u_t  = -alpha*(n*pi)^2 * sin(n*pi*x)*exp(...)
#   u_xx = -(n*pi)^2       * sin(n*pi*x)*exp(...)
#   => u_t - alpha*u_xx = 0  (source term f=0)

# ══════════════════════════════════════════════════════════════════
#  Network Architecture
# ══════════════════════════════════════════════════════════════════
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, HIDDEN_DIM), nn.Tanh()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()]
        layers.append(nn.Linear(HIDDEN_DIM, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# ══════════════════════════════════════════════════════════════════
#  PDE Residual (automatic differentiation)
# ══════════════════════════════════════════════════════════════════
def compute_residual(model, x, t):
    """Returns u_t - alpha*u_xx (target = 0)."""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True)[0]
    return u_t - ALPHA * u_xx

# ══════════════════════════════════════════════════════════════════
#  Collocation Point Sampling
# ══════════════════════════════════════════════════════════════════
def sample_res(n):
    x = torch.rand(n, 1, device=DEVICE)
    t = torch.rand(n, 1, device=DEVICE)
    return x, t

def sample_ic(n):
    x  = torch.rand(n, 1, device=DEVICE)
    t  = torch.zeros(n, 1, device=DEVICE)
    u0 = torch.tensor(
        np.sin(N_FREQ * np.pi * x.cpu().numpy()).astype(np.float32),
        device=DEVICE)
    return x, t, u0

def sample_bc(n):
    t_l  = torch.rand(n, 1, device=DEVICE)
    t_r  = torch.rand(n, 1, device=DEVICE)
    x_l  = torch.zeros(n, 1, device=DEVICE)
    x_r  = torch.ones(n, 1, device=DEVICE)
    zero = torch.zeros(n, 1, device=DEVICE)
    return (x_l, t_l, zero), (x_r, t_r, zero)

# ══════════════════════════════════════════════════════════════════
#  Fixed Test Grid
# ══════════════════════════════════════════════════════════════════
x_np = np.linspace(0, 1, N_TEST)
t_np = np.linspace(0, 1, N_TEST)
XX, TT     = np.meshgrid(x_np, t_np)          # (N_TEST, N_TEST)
U_EXACT_NP = exact_np(XX, TT)

x_test = torch.tensor(XX.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
t_test = torch.tensor(TT.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
u_exact_norm = np.sqrt(np.mean(U_EXACT_NP ** 2)) + 1e-10

def compute_l2(model):
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy().reshape(N_TEST, N_TEST)
    l2 = np.sqrt(np.mean((u_pred - U_EXACT_NP) ** 2)) / u_exact_norm
    model.train()
    return l2, u_pred

# ══════════════════════════════════════════════════════════════════
#  Shared Loss Computation
# ══════════════════════════════════════════════════════════════════
def compute_loss(model):
    """Returns (total, pde, ic, bc) loss tensors."""
    xr, tr       = sample_res(N_RES)
    res          = compute_residual(model, xr, tr)
    loss_pde     = torch.mean(res ** 2)

    xi, ti, ui   = sample_ic(N_IC)
    loss_ic      = torch.mean((model(xi, ti) - ui) ** 2)

    (xl, tl, ul), (xr2, tr2, ur2) = sample_bc(N_BC)
    loss_bc      = (torch.mean((model(xl, tl) - ul) ** 2) +
                    torch.mean((model(xr2, tr2) - ur2) ** 2))

    return loss_pde + loss_ic + loss_bc, loss_pde, loss_ic, loss_bc

# ══════════════════════════════════════════════════════════════════
#  History Storage
# ══════════════════════════════════════════════════════════════════
history = {
    "phase": [], "iter": [],
    "loss_total": [], "loss_pde": [], "loss_ic": [], "loss_bc": [],
    "l2_rel": [], "lr": [],
    "phase_time": [], "total_time": [],
}

def record(phase, it, lt, lp, li, lb, l2, lr_val, phase_t):
    history["phase"].append(phase)
    history["iter"].append(it)
    history["loss_total"].append(lt)
    history["loss_pde"].append(lp)
    history["loss_ic"].append(li)
    history["loss_bc"].append(lb)
    history["l2_rel"].append(l2)
    history["lr"].append(lr_val)
    history["phase_time"].append(phase_t)
    history["total_time"].append(time.time() - T_TRAIN_START)

# ══════════════════════════════════════════════════════════════════
#  Build Model
# ══════════════════════════════════════════════════════════════════
model = PINN().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

print(f"[Model]  {N_LAYERS} hidden layers x {HIDDEN_DIM} neurons, Tanh | "
      f"Parameters: {n_params:,}")
print(f"[PDE]    u_t - {ALPHA}*u_xx = 0,  n_freq = {N_FREQ}")
print(f"[Data]   N_res={N_RES}, N_ic={N_IC}, N_bc={N_BC}x2, "
      f"N_test={N_TEST}^2\n")

# Global training clock — started once, never reset between phases
T_TRAIN_START = time.time()

HDR = (f"{'Phase':>6} {'Iter':>7} | {'Loss_total':>12} | {'Loss_pde':>12} | "
       f"{'Loss_ic':>10} | {'Loss_bc':>10} | {'L2_rel':>10} | {'LR':>10} | "
       f"{'PhaseTime':>10} | {'TotalTime':>10}")
SEP = "-" * len(HDR)

def print_row(phase, it, lt, lp, li, lb, l2, lr_val, phase_elapsed):
    total_elapsed = time.time() - T_TRAIN_START
    print(f"{phase:>6} {it:>7} | {lt:>12.4e} | {lp:>12.4e} | "
          f"{li:>10.4e} | {lb:>10.4e} | {l2:>10.4e} | {lr_val:>10.2e} | "
          f"{phase_elapsed:>9.1f}s | {total_elapsed:>9.1f}s")

# ══════════════════════════════════════════════════════════════════
#  Phase 1: Adam (20000 iterations)
# ══════════════════════════════════════════════════════════════════
print("=" * len(HDR))
print("  PHASE 1 -- Adam optimizer  (20000 iterations)")
print("=" * len(HDR))
print(HDR); print(SEP)

optimizer_adam = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
scheduler_adam = torch.optim.lr_scheduler.StepLR(
    optimizer_adam, step_size=ADAM_LR_STEP, gamma=ADAM_LR_GAMMA)

t_adam_start = time.time()
for epoch in range(1, ADAM_EPOCHS + 1):
    model.train()
    optimizer_adam.zero_grad()
    loss, lp, li, lb = compute_loss(model)
    loss.backward()
    optimizer_adam.step()
    scheduler_adam.step()

    if epoch % ADAM_LOG_EVERY == 0 or epoch == 1:
        l2, _ = compute_l2(model)
        lr_now = optimizer_adam.param_groups[0]["lr"]
        phase_elapsed = time.time() - t_adam_start
        print_row("Adam", epoch,
                  loss.item(), lp.item(), li.item(), lb.item(),
                  l2, lr_now, phase_elapsed)
        record("adam", epoch,
               loss.item(), lp.item(), li.item(), lb.item(), l2, lr_now,
               phase_elapsed)

t_adam_total = time.time() - t_adam_start
t_total_after_adam = time.time() - T_TRAIN_START
l2_after_adam, _ = compute_l2(model)
print(SEP)
print(f"  Adam finished | Phase time: {t_adam_total:.1f}s | "
      f"Cumulative: {t_total_after_adam:.1f}s | "
      f"L2 after Adam: {l2_after_adam:.6e}\n")

# ══════════════════════════════════════════════════════════════════
#  Phase 2: L-BFGS (10000 iterations)
# ══════════════════════════════════════════════════════════════════
print("=" * len(HDR))
print("  PHASE 2 -- L-BFGS optimizer  (10000 iterations, strong Wolfe)")
print("=" * len(HDR))
print(HDR); print(SEP)

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    max_eval=25,
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
    history_size=50,
    line_search_fn="strong_wolfe",
)

lbfgs_step    = [0]
lbfgs_buf     = {}   # shared buffer written by closure

def closure():
    optimizer_lbfgs.zero_grad()
    loss, lp, li, lb = compute_loss(model)
    loss.backward()
    lbfgs_buf.update({"loss": loss.item(), "lp": lp.item(),
                      "li": li.item(), "lb": lb.item()})
    return loss

t_lbfgs_start = time.time()
while lbfgs_step[0] < LBFGS_MAX_ITER:
    optimizer_lbfgs.step(closure)
    lbfgs_step[0] += 1
    s = lbfgs_step[0]

    if s % LBFGS_LOG_EVERY == 0 or s == 1:
        l2, _ = compute_l2(model)
        phase_elapsed = time.time() - t_lbfgs_start
        lt = lbfgs_buf.get("loss", float("nan"))
        lp = lbfgs_buf.get("lp",   float("nan"))
        li = lbfgs_buf.get("li",   float("nan"))
        lb = lbfgs_buf.get("lb",   float("nan"))
        print_row("LBFGS", s, lt, lp, li, lb, l2, 1.0, phase_elapsed)
        record("lbfgs", ADAM_EPOCHS + s, lt, lp, li, lb, l2, 1.0, phase_elapsed)

t_lbfgs_total = time.time() - t_lbfgs_start
t_total_final  = time.time() - T_TRAIN_START
l2_final, u_pred_final = compute_l2(model)

print(SEP)
print(f"  L-BFGS finished | Phase time: {t_lbfgs_total:.1f}s | "
      f"Cumulative: {t_total_final:.1f}s | L2 after L-BFGS: {l2_final:.6e}")
print(f"\n  Total training time : {t_total_final:.1f}s")
print(f"  Final L2 rel error  : {l2_final:.6e}\n")

# ══════════════════════════════════════════════════════════════════
#  Save Text Data
# ══════════════════════════════════════════════════════════════════
# -- Loss history (numeric)
loss_path = os.path.join(SAVE_DIR, "loss_history.txt")
np.savetxt(
    loss_path,
    np.column_stack([history["iter"],       history["loss_total"],
                     history["loss_pde"],   history["loss_ic"],
                     history["loss_bc"],    history["l2_rel"],
                     history["lr"],         history["phase_time"],
                     history["total_time"]]),
    delimiter=",",
    header="iter,loss_total,loss_pde,loss_ic,loss_bc,l2_rel,lr,phase_time_s,total_time_s",
    comments="",
)
print(f"[Saved] {loss_path}")

# -- Loss history with phase label
phase_path = os.path.join(SAVE_DIR, "loss_history_phases.txt")
with open(phase_path, "w") as f:
    f.write("phase,iter,loss_total,loss_pde,loss_ic,loss_bc,l2_rel,lr,"
            "phase_time_s,total_time_s\n")
    for i, ph in enumerate(history["phase"]):
        f.write(f"{ph},{history['iter'][i]},{history['loss_total'][i]},"
                f"{history['loss_pde'][i]},{history['loss_ic'][i]},"
                f"{history['loss_bc'][i]},{history['l2_rel'][i]},"
                f"{history['lr'][i]},{history['phase_time'][i]},"
                f"{history['total_time'][i]}\n")
print(f"[Saved] {phase_path}")

# -- Prediction / exact / error arrays
error_map = np.abs(u_pred_final - U_EXACT_NP)
for arr, name, desc in [
    (u_pred_final, "prediction.txt",     "u_pred"),
    (U_EXACT_NP,   "exact_solution.txt", "u_exact"),
    (error_map,    "error_map.txt",      "abs_error"),
]:
    p = os.path.join(SAVE_DIR, name)
    np.savetxt(p, arr, delimiter=",",
               header=f"{desc}: shape ({N_TEST}x{N_TEST}), rows=t_idx, cols=x_idx")
    print(f"[Saved] {p}")

# -- Model weights
model_path = os.path.join(SAVE_DIR, "model.pth")
torch.save(model.state_dict(), model_path)
print(f"[Saved] {model_path}")

# ══════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════
# Helper masks
adam_mask  = [p == "adam"  for p in history["phase"]]
lbfgs_mask = [p == "lbfgs" for p in history["phase"]]

def get(key, mask):
    return np.array(history[key])[mask]

iter_a  = get("iter", adam_mask);  iter_l  = get("iter", lbfgs_mask)
lt_a    = get("loss_total", adam_mask); lt_l = get("loss_total", lbfgs_mask)
lp_a    = get("loss_pde", adam_mask);  lp_l = get("loss_pde", lbfgs_mask)
li_a    = get("loss_ic",  adam_mask);  li_l = get("loss_ic",  lbfgs_mask)
lb_a    = get("loss_bc",  adam_mask);  lb_l = get("loss_bc",  lbfgs_mask)
l2_a    = get("l2_rel",   adam_mask);  l2_l = get("l2_rel",   lbfgs_mask)
lr_a    = get("lr",       adam_mask)

def _semilogy_two(ax, xa, ya, xl, yl, ylabel, title):
    if len(xa): ax.semilogy(xa, ya, color="steelblue",  lw=2, label="Adam")
    if len(xl): ax.semilogy(xl, yl, color="darkorange", lw=2, label="L-BFGS")
    if len(xa): ax.axvline(xa[-1], color="gray", ls=":", lw=1.2)
    ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Fig 1: 6-panel loss dashboard ────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
_semilogy_two(axes[0,0], iter_a, lt_a, iter_l, lt_l, "Loss", "Total Loss")
_semilogy_two(axes[0,1], iter_a, lp_a, iter_l, lp_l, "Loss", "PDE Loss")
_semilogy_two(axes[0,2], iter_a, li_a, iter_l, li_l, "Loss", "IC Loss")
_semilogy_two(axes[1,0], iter_a, lb_a, iter_l, lb_l, "Loss", "BC Loss")
_semilogy_two(axes[1,1], iter_a, l2_a, iter_l, l2_l, "L2 Rel Error", "L2 Relative Error")
if len(lr_a):
    axes[1,2].semilogy(iter_a, lr_a, color="purple", lw=2)
    axes[1,2].set_xlabel("Iteration"); axes[1,2].set_ylabel("LR")
    axes[1,2].set_title("Adam Learning Rate"); axes[1,2].grid(True, alpha=0.3)

plt.suptitle(
    f"PINNs Heat Eq. (n={N_FREQ}, alpha={ALPHA}) | "
    f"Adam {ADAM_EPOCHS} + L-BFGS {lbfgs_step[0]} iters | "
    f"L2 = {l2_final:.4e}",
    fontsize=11)
plt.tight_layout()
p = os.path.join(SAVE_DIR, "loss_curves.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"[Saved] {p}")

# ── Fig 2: Solution slices ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, ts in zip(axes, [0.1, 0.5, 1.0]):
    idx = np.argmin(np.abs(t_np - ts))
    ax.plot(x_np, U_EXACT_NP[idx],   "b-",  lw=2, label="Exact")
    ax.plot(x_np, u_pred_final[idx], "r--", lw=2, label="PINN")
    ax.fill_between(x_np, U_EXACT_NP[idx], u_pred_final[idx],
                    alpha=0.15, color="red")
    ax.set_title(f"t = {ts}"); ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle(f"PINN vs Exact (n={N_FREQ}, alpha={ALPHA}) | L2 = {l2_final:.4e}", fontsize=12)
plt.tight_layout()
p = os.path.join(SAVE_DIR, "solution_slices.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"[Saved] {p}")

# ── Fig 3: 2D contour ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin = min(U_EXACT_NP.min(), u_pred_final.min())
vmax = max(U_EXACT_NP.max(), u_pred_final.max())
for ax, data, title in [
    (axes[0], U_EXACT_NP,   "Exact Solution"),
    (axes[1], u_pred_final, "PINN Prediction"),
]:
    cf = ax.contourf(XX, TT, data, levels=60, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax); ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("t")
cf2 = axes[2].contourf(XX, TT, error_map, levels=60, cmap="hot_r")
plt.colorbar(cf2, ax=axes[2])
axes[2].set_title("Absolute Error"); axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
plt.suptitle(f"2D Fields (n={N_FREQ}, alpha={ALPHA}) | L2 = {l2_final:.4e}", fontsize=12)
plt.tight_layout()
p = os.path.join(SAVE_DIR, "solution_2d.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"[Saved] {p}")

# ── Fig 4: 3D surface ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
for sub, data, title in [(121, U_EXACT_NP, "Exact"), (122, u_pred_final, "PINN")]:
    ax = fig.add_subplot(sub, projection="3d")
    ax.plot_surface(XX, TT, data, cmap=cm.coolwarm, alpha=0.85)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("u")
plt.suptitle(f"3D Surface (n={N_FREQ}, alpha={ALPHA})", fontsize=12)
plt.tight_layout()
p = os.path.join(SAVE_DIR, "solution_3d.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"[Saved] {p}")

# ── Fig 5: L2 convergence (two-phase) ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
if len(iter_a):
    ax.semilogy(iter_a, l2_a, "steelblue",  lw=2, label=f"Adam ({ADAM_EPOCHS} iters)")
if len(iter_l):
    ax.semilogy(iter_l, l2_l, "darkorange", lw=2, label=f"L-BFGS ({lbfgs_step[0]} iters)")
if len(iter_a):
    ax.axvline(iter_a[-1], color="gray", ls=":", lw=1.5)
    ax.text(iter_a[-1] * 1.01, ax.get_ylim()[0] * 3,
            "Adam -> L-BFGS", fontsize=9, color="gray")
ax.set_xlabel("Iteration"); ax.set_ylabel("L2 Relative Error")
ax.set_title("L2 Error Convergence (Two-Phase Training)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
p = os.path.join(SAVE_DIR, "l2_convergence.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"[Saved] {p}")

# ══════════════════════════════════════════════════════════════════
#  Summary File
# ══════════════════════════════════════════════════════════════════
summary_path = os.path.join(SAVE_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("   PINNs  High-Frequency Heat Equation  Summary\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"  PDE        : u_t - {ALPHA}*u_xx = 0\n")
    f.write(f"  Domain     : x in [0,1],  t in [0,1]\n")
    f.write(f"  IC         : u(x,0) = sin({N_FREQ}*pi*x)\n")
    f.write(f"  BC         : u(0,t) = u(1,t) = 0\n")
    f.write(f"  Exact      : sin({N_FREQ}*pi*x)*exp(-{ALPHA}*({N_FREQ}*pi)^2*t)\n\n")
    f.write(f"  Network    : {N_LAYERS} hidden layers x {HIDDEN_DIM} neurons, Tanh\n")
    f.write(f"  Parameters : {n_params:,}\n\n")
    f.write(f"  Collocation: N_res={N_RES}, N_ic={N_IC}, N_bc={N_BC}x2\n")
    f.write(f"  Test grid  : {N_TEST}x{N_TEST}\n\n")
    f.write(f"  [Phase 1] Adam\n")
    f.write(f"    Iterations  : {ADAM_EPOCHS}\n")
    f.write(f"    LR (init)   : {ADAM_LR}\n")
    f.write(f"    LR schedule : StepLR(step={ADAM_LR_STEP}, gamma={ADAM_LR_GAMMA})\n")
    f.write(f"    Time        : {t_adam_total:.1f} s\n")
    f.write(f"    L2 after    : {l2_after_adam:.6e}\n\n")
    f.write(f"  [Phase 2] L-BFGS\n")
    f.write(f"    Max iters   : {LBFGS_MAX_ITER}\n")
    f.write(f"    Actual steps: {lbfgs_step[0]}\n")
    f.write(f"    Line search : strong_wolfe\n")
    f.write(f"    Time        : {t_lbfgs_total:.1f} s\n")
    f.write(f"    L2 after    : {l2_final:.6e}\n\n")
    f.write(f"  Device       : {DEVICE}\n")
    f.write(f"  Total time   : {t_total_final:.1f} s\n\n")
    f.write(f"  >>> Final L2 relative error : {l2_final:.6e} <<<\n\n")
    f.write("  Saved files\n  " + "-" * 40 + "\n")
    for fn in ["loss_history.txt", "loss_history_phases.txt",
               "prediction.txt", "exact_solution.txt", "error_map.txt",
               "model.pth", "loss_curves.png", "solution_slices.png",
               "solution_2d.png", "solution_3d.png", "l2_convergence.png",
               "summary.txt"]:
        f.write(f"  {os.path.join(SAVE_DIR, fn)}\n")

print(f"[Saved] {summary_path}")
print("\n" + "=" * 65)
print(f"  DONE")
print(f"  Phase 1 Adam   ({ADAM_EPOCHS} iters) | L2: {l2_after_adam:.6e} | "
      f"Phase time: {t_adam_total:.1f}s")
print(f"  Phase 2 L-BFGS ({lbfgs_step[0]} iters) | L2: {l2_final:.6e} | "
      f"Phase time: {t_lbfgs_total:.1f}s")
print(f"  Total training time: {t_total_final:.1f}s")
print("=" * 65)