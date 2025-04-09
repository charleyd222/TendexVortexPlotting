import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.path as mpath

def colorline(ax, x, y, z, mag, norm, width = 1, widths = None, cmap=plt.get_cmap('copper')):
    if widths is None:
        widths = np.zeros(len(z)) + width

    # Turn x, y into segments of (x,y)
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, array=mag, cmap=cmap,linewidth=widths, norm = norm)
    ax.add_collection(lc)

    return lc
