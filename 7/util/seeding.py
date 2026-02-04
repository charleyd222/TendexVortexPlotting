import numpy as np
import matplotlib.pyplot as plt

# Funcs to generate seeds
def plane_seeds(xSeeds, ySeeds, zSeeds, seed_distance): #0
    totX_dist = (xSeeds - 1) / 2
    totY_dist = (ySeeds - 1) / 2
    totZ_dist = (zSeeds - 1) / 2

    x0 = []
    y0 = []
    z0 = []

    for x in range(xSeeds):
        for y in range(ySeeds):
            for z in range(zSeeds):
                if (x - totX_dist) * seed_distance != 0 or (y - totY_dist) * seed_distance != 0 or (z - totZ_dist) * seed_distance != 0:
                    x0 += [(x - totX_dist) * (seed_distance / xSeeds)]
                    y0 += [(y - totY_dist) * (seed_distance / ySeeds)]
                    z0 += [(z - totZ_dist) * (seed_distance / zSeeds)]

    return x0, y0, z0

def circular_seeds(seeds, starting_radius, radius_steps, radius_increment): #1
    x_list = []
    y_list = []
    z_list = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    for n in range(radius_steps):
        radius = starting_radius + (radius_increment * n)
        for i in range(seeds):
            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            if x != 0 or z != 0:
                x_list += [x]
                y_list += [0]
                z_list += [z]

    return x_list, y_list, z_list

def spherical_seeds(seeds, spacing): #2
    x_list = []
    y_list = []
    z_list = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(seeds):
        y = 1 - (i / float(seeds - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        if x != 0 or y != 0 or z != 0:
            x_list += [x * spacing]
            y_list += [y * spacing]
            z_list += [z * spacing]

    return x_list, y_list, z_list

def helical_seeds(start_rad, spacing, seeds): #3
    x_list = []
    y_list = []
    z_list = []
 
    for i in range(1,seeds+1):
        a = (0.78643418 * i) + 3.76697152
        x_list += [a * np.cos(i)]
        y_list += [a * np.sin(i)]
        z_list += [0]

    return x_list, y_list, z_list

def random_seeds(seeds, distance):
    x_list = []
    y_list = []
    z_list = []

    for i in range(seeds):
        # Quadrant 1
        x_list += [np.random.random() * distance - distance/2]
        y_list += [np.random.random() * distance - distance/2]
        z_list += [np.random.random() * distance - distance/2]

    return x_list, y_list, z_list

def frandom_seeds(seeds, distance):
    x_list = []
    y_list = []
    z_list = []

    for i in range(seeds//4):
        x = np.random.random() * distance
        y = np.random.random() * distance
        z = np.random.random() * distance

        # Quadrant 1
        x_list += [x]
        y_list += [y]
        z_list += [z]#[np.random.random() * distance - distance / 2]

        # Quadrant 2
        x_list += [-1 * x]
        y_list += [y]
        z_list += [z]#[np.random.random() * distance - distance / 2]

        # Quadrant 3
        x_list += [-1 * x]
        y_list += [-1 * y]
        z_list += [z]#[np.random.random() * distance - distance / 2]

        # Quadrant 4
        x_list += [x]
        y_list += [-1 * y]
        z_list += [z]#[np.random.random() * distance - distance / 2]
        # Quadrant 5
        x_list += [x]
        y_list += [y]
        z_list += [-1 * z]#[np.random.random() * distance - distance / 2]

        # Quadrant 6
        x_list += [-1 * x]
        y_list += [y]
        z_list += [-1 * z]#[np.random.random() * distance - distance / 2]

        # Quadrant 7
        x_list += [-1 * x]
        y_list += [-1 * y]
        z_list += [-1 * z]#[np.random.random() * distance - distance / 2]

        # Quadrant 8
        x_list += [x]
        y_list += [-1 * y]
        z_list += [-1 * z]#[np.random.random() * distance - distance / 2]

    return x_list, y_list, z_list

def rect(xseeds, radius):
    l = np.linspace(-radius, radius, xseeds)
    x0 = []
    y0 = []
    z0 = []

    l = l[1:-1]

    x0 += [radius, radius, -radius, -radius]
    y0 += [radius, -radius, radius, -radius]
    z0 += [0.,0.,0.,0.]

    for i in l:
        x0 += [i]
        y0 += [radius]

        x0 += [i]
        y0 += [-radius]

        y0 += [i]
        x0 += [radius]

        y0 += [i]
        x0 += [-radius]

        z0 += [0.,0.,0.,0.]

    return x0, y0, z0

def sphericalRand(seeds, radius):
    x = []
    y = []
    z = []
    for i in range(seeds//8):

        theta = np.random.rand() * np.pi/2
        phi = np.random.rand() * np.pi/2

        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

        phi += np.pi/2
        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

        theta += np.pi/2
        x += [radius * np.cos(theta) * np.sin(phi)]
        y += [radius * np.sin(theta) * np.sin(phi)]
        z += [radius * np.cos(phi)]

    return(x,y,z)

def seed(seed_type,distance, num = 10):
    rotate = False
    if seed_type == 0: # Plane
            
        xSeeds = 6
        ySeeds = 6
        zSeeds = 6

        if rotate:
            title = "Plane_x%sy%sz%s_Spaced:%s_RKA" % (xSeeds, ySeeds, zSeeds, distance)
        else:
            title = "Rot_Plane_x%sy%sz%s_Spaced:%s_RKA" % (xSeeds, ySeeds, zSeeds, distance)

        x0, y0, z0 = plane_seeds(xSeeds, ySeeds, zSeeds, distance)
    elif seed_type == 1: # Circular
        starting_radius = 2
        radius_steps = 2
        radius_increment = 5
        seeds = 5
        if rotate:
            title = "Circular seeding. Spacing: %s. RK Adaptive" % (radius_increment)
        else:
            title = "Circular seeding. Spacing: %s. RK Adaptive" % (radius_increment)
        
        x0, z0, y0 = circular_seeds(seeds, starting_radius, radius_steps, radius_increment)
    elif seed_type == 2: # Spherical
        radius = 10
        seeds = 15
        if rotate:
            title = "Spherical seeding. Radius: %s. RK Adaptive" % (radius)
        else:
            title = "Spherical seeding. Radius: %s. RK Adaptive" % (radius)

        x0, y0, z0 = spherical_seeds(seeds, radius)
    elif seed_type == 3: # Helical
        start_rad = 0
        spacing = 2
        seeds = 5

        if rotate:
            title = "Rotating dipole. Helical seeding. RK Adaptive"
        else:
            title = "Two dipole. Helical seeding. RK Adaptive"

        x0, y0, z0 = helical_seeds(start_rad, spacing, seeds)
    elif seed_type == 4:
        x0, y0, z0 = random_seeds(num, distance)
        title = "Random_RKA"
    elif seed_type == 5:
        x0, y0, z0 = rect(10,5)
        title = "Outline seeding. RK Adaptive"
    elif seed_type == 6:
        x0, y0, z0 = sphericalRand(num, distance)

        title = "Spherical rand seeding"

    return title, x0, y0, z0

if __name__ == '__main__':
    x,y,z = sphericalRand(100, 1)

    lim = 1.5

    pos_fig = plt.figure(1)
    pos_ax = pos_fig.add_subplot(projection='3d')   

    pos_ax.set_xlim(-lim,lim)
    pos_ax.set_ylim(-lim,lim)
    pos_ax.set_zlim(-lim,lim)

    pos_ax.scatter(x,y,z)
    plt.show()