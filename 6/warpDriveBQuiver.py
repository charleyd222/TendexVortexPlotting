import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sech(x):
    return 1/np.cosh(x)
def coth(x):
    return 1/np.tanh(x)

def e_val1(x,y,z,R,S): # Negative E Val
    r = np.sqrt((z*z) + (y*y) + (x*x))

    sechPlus2 = sech(S*(r + R)) * sech(S*(r + R))
    sechMin2 = sech(S*(r - R)) * sech(S*(r - R))
    sechMinPlus = sechMin2 - sechPlus2
    sechMinPlusTanh = sechMin2*np.tanh(S*(r-R)) - sechPlus2*np.tanh(S*(r+R))

    DzDyB = -.5 * coth(R*S) * S * y * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))
    DyDyB = -.5 * coth(R*S) * S * (y * y * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))
    DzDzB = -.5 * coth(R*S) * S * (z * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))

    return -.5 * np.sqrt(np.abs((DzDyB*DzDyB) - DyDyB*DzDzB))

#E val 2 = - e val 1

# Direction functions
def e_vec1(x, y, z, R, S):
    r = np.sqrt((z*z) + (y*y) + (x*x))

    sechPlus2 = sech(S*(r + R)) * sech(S*(r + R))
    sechMin2 = sech(S*(r - R)) * sech(S*(r - R))
    sechMinPlus = sechMin2 - sechPlus2
    sechMinPlusTanh = sechMin2*np.tanh(S*(r-R)) - sechPlus2*np.tanh(S*(r+R))

    DzDyB = -.5 * coth(R*S) * S * y * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))
    DyDyB = -.5 * coth(R*S) * S * (y * y * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))
    DzDzB = -.5 * coth(R*S) * S * (z * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))
    DyDxB = -.5 * coth(R*S) * S * y * x * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))
    DzDxB = -.5 * coth(R*S) * S * x * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))

    v1 = (DyDxB * DzDzB + DzDxB * DzDyB - DzDxB * np.sqrt(np.abs(DzDyB * DzDyB - DyDyB*DzDzB))) / (DzDzB * np.sqrt(np.abs(DzDyB * DzDyB - DyDyB*DzDzB)))
    v2 = -DzDyB - np.sqrt(np.abs(DzDyB * DzDyB - DyDyB*DzDzB)) / DzDzB

    return -1 * v1, -1 * v2, 1

def e_vec2(x, y, z, R, S):
    r = np.sqrt((z*z) + (y*y) + (x*x))

    sechPlus2 = sech(S*(r + R)) * sech(S*(r + R))
    sechMin2 = sech(S*(r - R)) * sech(S*(r - R))
    sechMinPlus = sechMin2 - sechPlus2
    sechMinPlusTanh = sechMin2*np.tanh(S*(r-R)) - sechPlus2*np.tanh(S*(r+R))

    DzDyB = -.5 * coth(R*S) * S * y * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))
    DyDyB = -.5 * coth(R*S) * S * (y * y * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))
    DzDzB = -.5 * coth(R*S) * S * (z * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2)) + (S * -1 * sechMinPlus / r))
    DyDxB = -.5 * coth(R*S) * S * y * x * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))
    DzDxB = -.5 * coth(R*S) * S * x * z * (sechMinPlus / (r**3) + (2 * S * sechMinPlusTanh / r**2))

    v1 = (-1 * DyDxB * DzDzB - DzDxB * DzDyB - DzDxB * np.sqrt(np.abs(DzDyB * DzDyB - DyDyB*DzDzB))) / (DzDzB * np.sqrt(np.abs(DzDyB * DzDyB - DyDyB*DzDzB)))
    v2 = -DzDyB + np.sqrt(np.abs(DzDyB * DzDyB - DyDyB*DzDzB)) / DzDzB

    return -1 * v1, -1 * v2, 1

N=8
# Define the grid for x, y, z
x = np.linspace(-5, 5, N)
y = np.linspace(-5, 5, N)
z = np.linspace(-5, 5, N)
X, Y, Z = np.meshgrid(x, y, z)
R = 5  # Example value for R
S = 5  # Example value for S

factor = 10
# Compute the magnitudes and directions
F1 = e_val1(X, Y, Z, R, S) * factor
F2 = -1 * e_val1(X, Y, Z, R, S) * factor

E1_x, E1_y, E1_z = e_vec1(X, Y, Z, R, S)
E2_x, E2_y, E2_z = e_vec1(X, Y, Z, R, S)

# Normalize direction vectors (e1, e2, e3)
magnitude_e1 = np.sqrt(E1_x**2 + E1_y**2 + E1_z**2)
E1_x, E1_y, E1_z = E1_x / magnitude_e1, E1_y / magnitude_e1, E1_z / magnitude_e1

magnitude_e2 = np.sqrt(E2_x**2 + E2_y**2 + E2_z**2)
E2_x, E2_y, E2_z = E2_x / magnitude_e2, E2_y / magnitude_e2, E2_z / magnitude_e2

# Scale the direction vectors by the magnitudes (f1, f2, f3)
E1_x *= F1
E1_y *= F1
E1_z *= F1

E2_x *= F2
E2_y *= F2
E2_z *= F2

# Create the 3D quiver plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each quiver (arrow) for the three components of direction vectors (E1, E2, E3)
ax.quiver(X, Y, Z, E1_x, E1_y, E1_z, length=0.1, color='r', linewidth=0.5)
ax.quiver(X, Y, Z, E2_x, E2_y, E2_z, length=0.1, color='b', linewidth=0.5)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('B Field warp drive quiver plot. Blue positive, red negative')

plt.show()

fig.savefig('warpDriveBQuiver.png', dpi = 500)