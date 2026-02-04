import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- Parameters ---
g = 9.81
l1, l2 = 1.0, 1.0
m1, m2 = 1.0, 1.0

# --- Equations of motion ---
def double_pendulum(t, y):
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2
    denom = 2*m1 + m2 - m2*np.cos(2*delta)
    
    dydt = np.zeros_like(y)
    dydt[0] = omega1
    dydt[1] = (-g*(2*m1 + m2)*np.sin(theta1) 
               - m2*g*np.sin(theta1 - 2*theta2)
               - 2*np.sin(delta)*m2*(omega2**2*l2 + omega1**2*l1*np.cos(delta))
              ) / (l1 * denom)
    dydt[2] = omega2
    dydt[3] = (2*np.sin(delta) * 
               (omega1**2*l1*(m1 + m2) + g*(m1 + m2)*np.cos(theta1) 
                + omega2**2*l2*m2*np.cos(delta))
              ) / (l2 * denom)
    return dydt

# --- Initial conditions ---
y0 = [np.pi/2, 0.0, np.pi/2 + 0.1, 0.0]

# --- Integrate ---
t_span = (0, 50)
t_eval = np.linspace(*t_span, 5000)
sol = solve_ivp(double_pendulum, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

theta1, omega1, theta2, omega2 = sol.y
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# --- Set up figure ---
fig, (ax_pend, ax_poincare) = plt.subplots(1, 2, figsize=(12,6))

# Pendulum plot
line1, = ax_pend.plot([], [], 'o-', lw=2, color='red', label='Bob 1')
line2, = ax_pend.plot([], [], 'o-', lw=2, color='blue', label='Bob 2')
ax_pend.set_xlim(-2.2, 2.2)
ax_pend.set_ylim(-2.2, 0.5)
ax_pend.set_aspect('equal')
ax_pend.grid(True)
ax_pend.set_title('Double Pendulum')
ax_pend.legend()

# Poincare plot
poincare_points, = ax_poincare.plot([], [], 'k.', markersize=2)
ax_poincare.set_xlim(-np.pi, np.pi)
ax_poincare.set_ylim(-10, 10)
ax_poincare.set_title('Poincare Map (theta_2=0)')
ax_poincare.set_xlabel('theta_1 [rad]')
ax_poincare.set_ylabel('omega_1 [rad/s]')
ax_poincare.grid(True)

# --- Animation ---
theta1_vals, omega1_vals, theta2_vals, omega2_vals = theta1, omega1, theta2, omega2
poincare_theta1, poincare_omega1 = [], []

def update(frame):
    # Update pendulum
    x = [0, x1[frame], x2[frame]]
    y = [0, y1[frame], y2[frame]]
    line1.set_data([0, x[1]], [0, y[1]])
    line2.set_data([x[1], x[2]], [y[1], y[2]])
    
    # Update Poincare
    if theta2[frame-1] < 0 <= theta2[frame]:
        poincare_theta1.append(theta1[frame])
        poincare_omega1.append(omega1[frame])
        poincare_points.set_data(poincare_theta1, poincare_omega1)
    
    return line1, line2, poincare_points

ani = FuncAnimation(fig, update, frames=len(t_eval), blit=True, interval=10)
plt.show()
