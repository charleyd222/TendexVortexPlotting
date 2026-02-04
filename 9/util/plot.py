import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath

def colorline(ax, x, y, z, norm, width = 2, widths = None, cmap=plt.get_cmap('copper')):
    if widths is None:
        widths = np.zeros(len(z)) + width

    # Turn x, y into segments of (x,y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = mcoll.LineCollection(segments, array=z, cmap=cmap,linewidth=widths, norm = norm, zorder=0)
    ax.add_collection(lc)

    return lc
