import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters
g = 9.81
l1, l2 = 1.0, 1.0
m1, m2 = 1.0, 1.0

def double_pendulum(t, y):
    theta1, omega1, theta2, omega2 = y
    delta = theta1 - theta2
    
    denom1 = (2*m1 + m2 - m2*np.cos(2*delta))
    denom2 = (l2/ l1) * denom1
    
    dydt = np.zeros_like(y)
    dydt[0] = omega1
    dydt[1] = (-g*(2*m1 + m2)*np.sin(theta1) 
               - m2*g*np.sin(theta1 - 2*theta2)
               - 2*np.sin(delta)*m2*(omega2**2*l2 + omega1**2*l1*np.cos(delta))
              ) / (l1 * denom1)
    dydt[2] = omega2
    dydt[3] = (2*np.sin(delta) * 
               (omega1**2*l1*(m1 + m2) + g*(m1 + m2)*np.cos(theta1) 
                + omega2**2*l2*m2*np.cos(delta))
              ) / (l2 * denom1)
    return dydt

# initial conditions
y0 = [np.pi/2, 0.0, np.pi/2 + 0.1, 0.0]

# integrate
t_span = (0, 200)
t_eval = np.linspace(*t_span, 50000)
sol = solve_ivp(double_pendulum, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

theta1, omega1, theta2, omega2 = sol.y

# unwrap angles
theta1 = np.mod(theta1 + np.pi, 2*np.pi) - np.pi
theta2 = np.mod(theta2 + np.pi, 2*np.pi) - np.pi

# detect crossings: theta2 = 0 and omega2 > 0
cross_indices = np.where((theta2[:-1] < 0) & (theta2[1:] >= 0))[0]

poincare_theta1 = theta1[cross_indices]
poincare_omega1 = omega1[cross_indices]

fig, ax = plt.subplots(2)

# plot Poincaré map
ax[0].scatter(poincare_theta1, poincare_omega1, s=1, color='black')
ax[0].set_title("Poincaré Map of Double Pendulum (theta_2=0)")
ax[0].set_xlabel("theta_1 [rad]")
ax[0].set_ylabel("omega_1 [rad/s]")
ax[0].grid(True)

# positions of bobs
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# plot pendulum path
ax[1].plot(x2, y2, color='blue', lw=0.5, label='Bob 2 path')
ax[1].plot(x1, y1, color='red', lw=0.5, label='Bob 1 path')
ax[1].set_title("Double Pendulum Trajectory")
ax[1].set_xlabel("x [m]")
ax[1].set_ylabel("y [m]")
ax[1].axis('equal')
ax[1].grid(True)
fig.legend()
plt.show()
