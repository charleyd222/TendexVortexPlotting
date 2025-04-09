import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sech(x):
    return 1/np.cosh(x)

# Define the functions for magnitudes (f1, f2, f3) and directions (e1, e2, e3)
def f1(x, y, z, R, S):
    r = np.sqrt((z*z) + (y*y) + (x*x))
    
    DY = ((S*y * sech(S*(R+r)) * sech(S*(R+r))) - (S*y * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)
    DZ = ((S*z * sech(S*(R+r)) * sech(S*(R+r))) - (S*z * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)
    DY2 = DY * DY
    DZ2 = DZ * DZ

    return (1/12) * (-DY2 - DZ2)

def f3(x, y, z, R, S):
    r = np.sqrt((z*z) + (y*y) + (x*x))
    
    DY = ((S*y * sech(S*(R+r)) * sech(S*(R+r))) - (S*y * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)
    DZ = ((S*z * sech(S*(R+r)) * sech(S*(R+r))) - (S*z * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)
    DY2 = DY * DY
    DZ2 = DZ * DZ

    return (1/6) * (DY2 + DZ2)

# Direction functions (these return 3D direction vectors for each point)
def e1(x, y, z, R, S):
    r = np.sqrt((z*z) + (y*y) + (x*x))
    
    DY = ((S*y * sech(S*(R+r)) * sech(S*(R+r))) - (S*y * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)
    DZ = ((S*z * sech(S*(R+r)) * sech(S*(R+r))) - (S*z * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)

    return 0, -1 * DZ, DY

def e2(x, y, z, R, S):
    return 1,0,0

def e3(x, y, z, R, S):
    r = np.sqrt((z*z) + (y*y) + (x*x))
    
    DY = ((S*y * sech(S*(R+r)) * sech(S*(R+r))) - (S*y * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)
    DZ = ((S*z * sech(S*(R+r)) * sech(S*(R+r))) - (S*z * sech(S*(R-r)) * sech(S*(R-r)))) / (r * np.tanh(R*S) * -2.0)

    return 0, DY, DZ

N=15
# Define the grid for x, y, z
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
z = np.linspace(-5, 5, N)
X, Y, Z = np.meshgrid(x, y, z)
R = 5  # Example value for R
S = 5  # Example value for S

factor = 10
# Compute the magnitudes and directions
F1 = f1(X, Y, Z, R, S) * factor
F2 = f1(X, Y, Z, R, S) * factor
F3 = f3(X, Y, Z, R, S) * factor

E1_x, E1_y, E1_z = e1(X, Y, Z, R, S)
E2_x, E2_y, E2_z = e2(X, Y, Z, R, S)
E3_x, E3_y, E3_z = e3(X, Y, Z, R, S)

# Normalize direction vectors (e1, e2, e3)
magnitude_e1 = np.sqrt(E1_x**2 + E1_y**2 + E1_z**2)
E1_x, E1_y, E1_z = E1_x / magnitude_e1, E1_y / magnitude_e1, E1_z / magnitude_e1

magnitude_e2 = np.sqrt(E2_x**2 + E2_y**2 + E2_z**2)
E2_x, E2_y, E2_z = E2_x / magnitude_e2, E2_y / magnitude_e2, E2_z / magnitude_e2

magnitude_e3 = np.sqrt(E3_x**2 + E3_y**2 + E3_z**2)
E3_x, E3_y, E3_z = E3_x / magnitude_e3, E3_y / magnitude_e3, E3_z / magnitude_e3

# Scale the direction vectors by the magnitudes (f1, f2, f3)
E1_x *= F1
E1_y *= F1
E1_z *= F1

E2_x *= F2
E2_y *= F2
E2_z *= F2

E3_x *= F3
E3_y *= F3
E3_z *= F3

# Create the 3D quiver plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each quiver (arrow) for the three components of direction vectors (E1, E2, E3)
ax.quiver(X, Y, Z, E1_x, E1_y, E1_z, length=0.1, color='r', linewidth=0.5)
ax.quiver(X, Y, Z, E2_x, E2_y, E2_z, length=0.1, color='r', linewidth=0.5)
ax.quiver(X, Y, Z, E3_x, E3_y, E3_z, length=0.1, color='b', linewidth=0.5)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Quiver Plot of f1, f2, f3 with Normalized Directions e1, e2, e3')

plt.show()

fig.savefig('warpDriveQuiver.png', dpi = 500)