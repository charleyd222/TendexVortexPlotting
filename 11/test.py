import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

def f(omega1, omega2, r, t):
    return (32/ 105) * (omega1*omega2)**4 * np.pi * (21 + 10 * r**4) * np.cos(2 * (omega1 - omega2) * (r-t))

def f(omega1, omega2, r, t):
    return omega1

R = 1
omega1 = 1
omega2 = 1

t = np.linspace(0,30,200)

plt.plot(t, f(omega1, omega2, R, t))
plt.show()