import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. PHYSICAL SYSTEM DEFINITION
# Governing Equation: d^2y/dt^2 + 2*gamma*dy/dt + w0^2*y = 0
# ==========================================
damp_gamma = 0.4    # Damping coefficient
w0 = 2.0 * np.pi    # Natural frequency (1 Hz)

def exact_analytical_solution(t: np.ndarray) -> np.ndarray:
    """Exact analytical response of an underdamped harmonic oscillator."""
    w_d = np.sqrt(w0**2 - damp_gamma**2)
    return np.exp(-damp_gamma * t) * np.cos(w_d * t)

# ==========================================
# 2. GENERATE SPARSE, NOISY SENSOR DATA
# ==========================================
t_dense = np.linspace(0, 3, 300)
y_dense_exact = exact_analytical_solution(t_dense)

# Simulate small data regime: Only 12 noisy sensor readings available
num_sensor_points = 12
t_sensor_np = np.sort(np.random.choice(t_dense[:150], size=num_sensor_points, replace=False))
y_sensor_np = exact_analytical_solution(t_sensor_np) + np.random.normal(0, 0.05, size=num_sensor_points)

# Convert arrays to PyTorch Tensors
t_sensor = torch.tensor(t_sensor_np, dtype=torch.float32).reshape(-1, 1)
y_sensor = torch.tensor(y_sensor_np, dtype=torch.float32).reshape(-1, 1)

# Physics Collocation Points (evaluating differential equation residual across domain)
t_physics = torch.linspace(0, 3, 100, requires_grad=True).reshape(-1, 1)


# ==========================================
# 3. PHYSICS-INFORMED NEURAL NETWORK (PINN)
# ==========================================
class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return self.net(t)

# Instantiate models for comparison: Pure Data Driven vs. Physics-Informed
pure_data_model = PhysicsInformedNN()
pinn_model = PhysicsInformedNN()

optimizer_data = torch.optim.Adam(pure_data_model.parameters(), lr=1e-3)
optimizer_pinn = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)


# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
epochs = 4000
physics_loss_weight = 1e-3

# --- Train Pure Data-Driven Model ---
for epoch in range(epochs):
    optimizer_data.zero_grad()
    y_pred_sensor = pure_data_model(t_sensor)
    loss_data_only = nn.MSELoss()(y_pred_sensor, y_sensor)
    loss_data_only.backward()
    optimizer_data.step()

# --- Train Physics-Informed Model (PINN) ---
for epoch in range(epochs):
    optimizer_pinn.zero_grad()
    
    # Data Loss component
    y_pred_sensor = pinn_model(t_sensor)
    loss_data = nn.MSELoss()(y_pred_sensor, y_sensor)
    
    # Physics Loss component using Autograd derivatives
    y_phys = pinn_model(t_physics)
    
    # First Derivative dy/dt
    dy_dt = torch.autograd.grad(
        outputs=y_phys, inputs=t_physics,
        grad_outputs=torch.ones_like(y_phys),
        create_graph=True
    )[0]
    
    # Second Derivative d^2y/dt^2
    d2y_dt2 = torch.autograd.grad(
        outputs=dy_dt, inputs=t_physics,
        grad_outputs=torch.ones_like(dy_dt),
        create_graph=True
    )[0]
    
    # Damped Oscillator Differential Equation Residual
    physics_residual = d2y_dt2 + 2 * damp_gamma * dy_dt + (w0**2) * y_phys
    loss_physics = torch.mean(physics_residual**2)
    
    # Total Combined Loss
    total_loss = loss_data + physics_loss_weight * loss_physics
    total_loss.backward()
    optimizer_pinn.step()


# ==========================================
# 5. EVALUATION & GRAPHICAL VISUALIZATION
# ==========================================
t_eval = torch.tensor(t_dense, dtype=torch.float32).reshape(-1, 1)
y_pred_pure = pure_data_model(t_eval).detach().numpy()
y_pred_pinn = pinn_model(t_eval).detach().numpy()

plt.figure(figsize=(11, 5))

# Plot Exact Physics vs Models
plt.plot(t_dense, y_dense_exact, 'k--', label='True Physical System (Ground Truth)', alpha=0.75, lw=2)
plt.scatter(t_sensor_np, y_sensor_np, color='crimson', s=60, zorder=5, label='Sparse Sensor Data (12 samples)')

plt.plot(t_dense, y_pred_pure, color='orange', lw=2, label='Standard Data-Driven Neural Net (Overfits/Drifts)')
plt.plot(t_dense, y_pred_pinn, color='#1f77b4', lw=2.5, label='Physics-Informed Neural Net (PINN)')

plt.title("Physics-Informed ML: Reconstructing Continuous Signal Dynamics from Sparse Data", fontsize=12, fontweight='bold')
plt.xlabel("Time (s)")
plt.ylabel("Sensor Signal Amplitude")
plt.axvspan(1.5, 3.0, color='gray', alpha=0.1, label='Extrapolation Region (No Data)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()