import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# -----------------------------------------------------------------------------
# 1. Basic Setup and Hyperparameters
# -----------------------------------------------------------------------------
SAVE_DIR = "output/gemini31_pro"
os.makedirs(SAVE_DIR, exist_ok=True)

# Fix random seed
torch.manual_seed(1234)
np.random.seed(1234)

# Prefer GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==================================================")
print(f"Using device: {device}")
print(f"==================================================")

# Physical parameters (high-freq forward problem)
alpha = 0.01
k = 5.0

# Training iterations
adam_epochs = 20000
lbfgs_max_iter = 10000

# Number of data points
N_u = 1000    # IC/BC points
N_f = 20000   # Interior collocation points

# -----------------------------------------------------------------------------
# 2. Dataset Generation (Training & Testing)
# -----------------------------------------------------------------------------
def exact_solution(x, t):
    return np.sin(k * np.pi * x) * np.exp(-alpha * (k * np.pi)**2 * t)

# --- Training set ---
x_f = np.random.uniform(-1.0, 1.0, (N_f, 1))
t_f = np.random.uniform(0.0, 1.0, (N_f, 1))
X_f = torch.tensor(np.hstack((x_f, t_f)), dtype=torch.float32, requires_grad=True).to(device)

x_0 = np.random.uniform(-1.0, 1.0, (N_u, 1))
t_0 = np.zeros((N_u, 1))
X_0 = torch.tensor(np.hstack((x_0, t_0)), dtype=torch.float32).to(device)
U_0 = torch.tensor(exact_solution(x_0, t_0), dtype=torch.float32).to(device)

t_b = np.random.uniform(0.0, 1.0, (N_u, 1))
x_b1 = -np.ones((N_u // 2, 1))
x_b2 = np.ones((N_u // 2, 1))
X_b = torch.tensor(np.hstack((np.vstack((x_b1, x_b2)), np.vstack((t_b[:N_u//2], t_b[N_u//2:])))), dtype=torch.float32).to(device)
U_b = torch.tensor(np.zeros((N_u, 1)), dtype=torch.float32).to(device)

# --- Test set (for real-time L2 error computation) ---
x_test = np.linspace(-1.0, 1.0, 512)
t_test = np.linspace(0.0, 1.0, 200)
X_test_grid, T_test_grid = np.meshgrid(x_test, t_test)
X_test_flat = X_test_grid.flatten()[:, None]
T_test_flat = T_test_grid.flatten()[:, None]

u_exact_test = exact_solution(X_test_flat, T_test_flat)
exact_norm_2 = np.linalg.norm(u_exact_test, 2)

X_test_tensor = torch.tensor(np.hstack((X_test_flat, T_test_flat)), dtype=torch.float32).to(device)

# -----------------------------------------------------------------------------
# 3. Fourier Feature PINN Architecture
# -----------------------------------------------------------------------------
class FourierFeaturePINN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, sigma=3.0):
        super(FourierFeaturePINN, self).__init__()
        self.B = nn.Parameter(torch.randn(in_dim, hidden_layers[0] // 2) * sigma, requires_grad=False)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            nn.init.xavier_normal_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)
        self.out_layer = nn.Linear(hidden_layers[-1], out_dim)
        nn.init.xavier_normal_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)
        self.activation = nn.Tanh()

    def forward(self, x):
        x_proj = 2.0 * np.pi * torch.matmul(x, self.B)
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out_layer(x)
        return x

hidden_dims = [128, 128, 128, 128, 128]
model = FourierFeaturePINN(in_dim=2, out_dim=1, hidden_layers=hidden_dims, sigma=3.0).to(device)

# -----------------------------------------------------------------------------
# 4. Loss Computation and L2 Error
# -----------------------------------------------------------------------------
def compute_loss():
    u_pred_0 = model(X_0)
    loss_ic = torch.mean((u_pred_0 - U_0)**2)

    u_pred_b = model(X_b)
    loss_bc = torch.mean((u_pred_b - U_b)**2)

    u_f = model(X_f)
    du_dX = torch.autograd.grad(u_f, X_f, grad_outputs=torch.ones_like(u_f), retain_graph=True, create_graph=True)[0]
    u_x = du_dX[:, 0:1]
    u_t = du_dX[:, 1:2]
    
    du_dx_dX = torch.autograd.grad(u_x, X_f, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_xx = du_dx_dX[:, 0:1]

    f_pred = u_t - alpha * u_xx
    loss_pde = torch.mean(f_pred**2)

    loss_total = loss_pde + 20.0 * loss_ic + 20.0 * loss_bc
    return loss_pde, loss_ic, loss_bc, loss_total

def compute_l2_error():
    model.eval()
    with torch.no_grad():
        u_pred = model(X_test_tensor).cpu().numpy()
    error_l2 = np.linalg.norm(u_exact_test - u_pred, 2) / exact_norm_2
    model.train()
    return error_l2

# -----------------------------------------------------------------------------
# 5. Training Loop (Adam -> L-BFGS)
# -----------------------------------------------------------------------------
history = {'epoch': [], 'loss_pde': [], 'loss_ic': [], 'loss_bc': [], 'loss_total': [], 'l2_error': [], 'time': []}
start_time = time.time()

# --- Phase 1: Adam ---
print(">>> Starting Adam Optimizer (20,000 epochs) <<<")
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(adam_epochs):
    optimizer_adam.zero_grad()
    loss_pde, loss_ic, loss_bc, loss_total = compute_loss()
    loss_total.backward()
    optimizer_adam.step()

    if epoch % 100 == 0 or epoch == adam_epochs - 1:
        elapsed_time = time.time() - start_time
        current_l2 = compute_l2_error()
        history['epoch'].append(epoch)
        history['loss_pde'].append(loss_pde.item())
        history['loss_ic'].append(loss_ic.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_total'].append(loss_total.item())
        history['l2_error'].append(current_l2)
        history['time'].append(elapsed_time)

    if epoch % 1000 == 0:
        print(f"Adam Epoch: {epoch:05d} | Total: {loss_total.item():.4e} | PDE: {loss_pde.item():.4e} | IC: {loss_ic.item():.4e} | BC: {loss_bc.item():.4e} | L2: {current_l2:.4e} | Time: {elapsed_time:.2f}s")

# --- Phase 2: L-BFGS ---
print("\n>>> Starting L-BFGS Optimizer (up to 10,000 iterations) <<<")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), 
    lr=1.0, 
    max_iter=lbfgs_max_iter, 
    max_eval=lbfgs_max_iter, 
    tolerance_grad=1e-7, 
    tolerance_change=1e-9, 
    history_size=50, 
    line_search_fn="strong_wolfe"
)

lbfgs_iter = 0

def closure():
    global lbfgs_iter
    optimizer_lbfgs.zero_grad()
    loss_pde, loss_ic, loss_bc, loss_total = compute_loss()
    loss_total.backward()
    
    if lbfgs_iter % 100 == 0:
        elapsed_time = time.time() - start_time
        current_l2 = compute_l2_error()
        current_step = adam_epochs + lbfgs_iter
        history['epoch'].append(current_step)
        history['loss_pde'].append(loss_pde.item())
        history['loss_ic'].append(loss_ic.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_total'].append(loss_total.item())
        history['l2_error'].append(current_l2)
        history['time'].append(elapsed_time)
        
    if lbfgs_iter % 1000 == 0:
        print(f"L-BFGS Iter: {lbfgs_iter:05d} | Total: {loss_total.item():.4e} | PDE: {loss_pde.item():.4e} | IC: {loss_ic.item():.4e} | BC: {loss_bc.item():.4e} | L2: {current_l2:.4e} | Time: {elapsed_time:.2f}s")
        
    lbfgs_iter += 1
    return loss_total

optimizer_lbfgs.step(closure)

total_training_time = time.time() - start_time
print(f"\nTraining completed in {total_training_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# 6. Final L2 Error Evaluation and Data Export
# -----------------------------------------------------------------------------
final_l2_error = compute_l2_error()

print(f"==================================================")
print(f"Final Relative L2 Error: {final_l2_error:.6e}")
print(f"==================================================")

model.eval()
with torch.no_grad():
    u_pred_test = model(X_test_tensor).cpu().numpy()
u_exact_grid = u_exact_test.reshape(X_test_grid.shape)
u_pred_grid = u_pred_test.reshape(X_test_grid.shape)
error_grid = np.abs(u_exact_grid - u_pred_grid)

# -----------------------------------------------------------------------------
# 7. Save Data to txt
# -----------------------------------------------------------------------------
history_data = np.column_stack((
    history['epoch'], history['loss_total'], history['loss_pde'], 
    history['loss_ic'], history['loss_bc'], history['l2_error'], history['time']
))
np.savetxt(os.path.join(SAVE_DIR, 'loss_and_l2_history.txt'), history_data, 
           header="Epoch Total_Loss PDE_Loss IC_Loss BC_Loss L2_Error Time(s)", comments='')

np.savetxt(os.path.join(SAVE_DIR, 'exact_solution.txt'), u_exact_grid)
np.savetxt(os.path.join(SAVE_DIR, 'pred_solution.txt'), u_pred_grid)
np.savetxt(os.path.join(SAVE_DIR, 'error_absolute.txt'), error_grid)

print(f"All data successfully saved in directory: {SAVE_DIR}/")

# -----------------------------------------------------------------------------
# 8. Plot and Save Figures
# -----------------------------------------------------------------------------
plt.rcParams['font.size'] = 12

# 1. Plot all loss components and L2 error curves
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left Y-axis: all loss components
ax1.plot(history['epoch'], history['loss_total'], label='Total Loss', c='black', linewidth=2)
ax1.plot(history['epoch'], history['loss_pde'], label='PDE Loss', linestyle='--')
ax1.plot(history['epoch'], history['loss_ic'], label='IC Loss', linestyle='-.')
ax1.plot(history['epoch'], history['loss_bc'], label='BC Loss', linestyle=':')
ax1.axvline(x=adam_epochs, color='red', linestyle='-', alpha=0.5, label='Adam -> L-BFGS')

ax1.set_yscale('log')
ax1.set_xlabel('Epoch / Iterations')
ax1.set_ylabel('Loss (Log Scale)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

# Right Y-axis: L2 Error
ax2 = ax1.twinx()
ax2.plot(history['epoch'], history['l2_error'], label='Rel. L2 Error', color='green', linewidth=2, alpha=0.8)
ax2.set_yscale('log')
ax2.set_ylabel('Relative L2 Error (Log Scale)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Merge legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', bbox_to_anchor=(1.15, 0.5))

plt.title('Training Losses and L2 Error History')
plt.savefig(os.path.join(SAVE_DIR, 'loss_and_l2_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Prediction vs Exact Solution (Contour)
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
c1 = ax[0].pcolormesh(T_test_grid, X_test_grid, u_exact_grid, cmap='jet', shading='auto')
ax[0].set_title('Exact Solution')
ax[0].set_ylabel('x')
fig.colorbar(c1, ax=ax[0])

c2 = ax[1].pcolormesh(T_test_grid, X_test_grid, u_pred_grid, cmap='jet', shading='auto')
ax[1].set_title('PINN Prediction')
ax[1].set_ylabel('x')
fig.colorbar(c2, ax=ax[1])

c3 = ax[2].pcolormesh(T_test_grid, X_test_grid, error_grid, cmap='viridis', shading='auto')
ax[2].set_title(f'Absolute Error (Rel. L2 = {final_l2_error:.2e})')
ax[2].set_xlabel('t')
ax[2].set_ylabel('x')
fig.colorbar(c3, ax=ax[2])

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'solution_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Time slice comparison
t_idx = 100 
plt.figure(figsize=(8, 5))
plt.plot(x_test, u_exact_grid[t_idx, :], 'b-', label='Exact Solution', linewidth=2)
plt.plot(x_test, u_pred_grid[t_idx, :], 'r--', label='PINN Prediction', linewidth=2)
plt.title(f'Wave Profile at t = {t_test[t_idx]:.2f}')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, 'time_slice_t05.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All plots successfully saved.")