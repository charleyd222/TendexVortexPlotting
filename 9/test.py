# squirrel_dihedral.py
# Visualize a Dihedral group D_n arranged along a squirrel silhouette.
# Requirements: numpy, matplotlib, pillow (PIL)
# Usage: put a 'squirrel.png' silhouette in same folder (optional), then run.

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def load_or_make_silhouette(filename='squirrel.png', size=(600,600)):
    if os.path.exists(filename):
        im = Image.open(filename).convert('L')  # grayscale
        im = im.resize(size, Image.LANCZOS)
        # threshold to binary mask
        arr = np.array(im)
        thresh = np.percentile(arr, 30)  # try to pick silhouette pixels
        mask = (arr < thresh)  # assume dark silhouette on light bg
        return mask
    else:
        # make an approximate squirrel-ish silhouette using ellipses and circles
        w,h = size
        img = Image.new('L', size, 255)
        draw = ImageDraw.Draw(img)
        # tail (big ellipse)
        draw.ellipse([w*0.05,h*0.15,w*0.6,h*0.8], fill=0)
        # body
        draw.ellipse([w*0.4,h*0.3,w*0.9,h*0.75], fill=0)
        # head
        draw.ellipse([w*0.75,h*0.18,w*0.95,h*0.36], fill=0)
        # ear
        draw.polygon([(w*0.86,h*0.18),(w*0.82,h*0.06),(w*0.9,h*0.14)], fill=0)
        # leg / foot
        draw.ellipse([w*0.7,h*0.6,w*0.78,h*0.7], fill=0)
        # smooth a bit by resizing
        img = img.filter(Image.Image.filter) if False else img  # no-op to avoid imports
        mask = np.array(img) < 128
        return mask

def boundary_ordered_points(mask, num_points=64):
    # mask: boolean 2D array where True = silhouette pixel
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No silhouette pixels found.")
    # centroid (mean)
    cx = xs.mean(); cy = ys.mean()
    # compute angle of each mask pixel about centroid
    angles = np.arctan2(ys - cy, xs - cx)  # range [-pi, pi]
    # For each unique angle bin, pick the furthest-out pixel (approx boundary)
    # Build array of (angle, radius, x, y)
    rs = np.hypot(xs - cx, ys - cy)
    data = np.vstack([angles, rs, xs, ys]).T
    # sort by angle then choose max radius per small angle bin
    bins = np.linspace(-math.pi, math.pi, 1000)  # fine bins
    chosen = []
    idx = np.digitize(data[:,0], bins)
    for b in np.unique(idx):
        block = data[idx == b]
        if block.size == 0: 
            continue
        # pick point with largest radius (outermost)
        far = block[np.argmax(block[:,1])]
        chosen.append(far)
    chosen = np.array(chosen)
    # sort chosen by angle so they follow outline
    chosen = chosen[np.argsort(chosen[:,0])]
    angs = chosen[:,0]; xs_b = chosen[:,2]; ys_b = chosen[:,3]
    # now sample num_points evenly along the ordered boundary
    m = len(xs_b)
    if m == 0:
        raise ValueError("No boundary found.")
    indices = np.linspace(0, m-1, num_points, dtype=int)
    pts = np.column_stack([xs_b[indices], ys_b[indices]]).astype(float)
    # convert to Cartesian coordinates with origin at centroid for nicer plotting
    cx = xs.mean(); cy = ys.mean()
    pts[:,0] = pts[:,0] - cx
    pts[:,1] = -(pts[:,1] - cy)  # flip y for usual math coordinates (up is +)
    return pts

def plot_dihedral_on_shape(pts, n=None, show_edges=True, figsize=(8,8)):
    if n is None: 
        n = len(pts)
    else:
        pts = pts[:n]
    # nodes indices 0..n-1
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    # draw silhouette outline by connecting pts (for context)
    xs = pts[:,0]; ys = pts[:,1]
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color='lightgray', linewidth=1.2, zorder=0)
    # draw nodes
    ax.scatter(xs, ys, s=100, c='tab:blue', zorder=3)
    for i,(x,y) in enumerate(pts):
        ax.text(x, y, str(i), color='white', ha='center', va='center', fontsize=8, zorder=4)
    if show_edges:
        # rotation generator r: i -> i+1 (mod n)
        for i in range(n):
            j = (i+1) % n
            ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], linestyle='-', linewidth=1.2, color='tab:orange', alpha=0.9, zorder=1)
        # reflection generator s: i -> (-i mod n). Choose reflection axis through index 0 and midpoint of opposite arc
        for i in range(n):
            j = (-i) % n
            ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], linestyle='--', linewidth=0.9, color='tab:green', alpha=0.8, zorder=1)
    ax.set_title(f"Dihedral group D_{n} placed on a squirrel silhouette (nodes labeled 0..{n-1})")
    ax.axis('off')
    plt.show()

def main(filename='squirrel.png', n=48):
    mask = load_or_make_silhouette(filename)
    pts = boundary_ordered_points(mask, num_points=n)
    plot_dihedral_on_shape(pts, n=n)

if __name__ == '__main__':
    # default: attempt to use squirrel.png; if not present, uses built-in approximation
    main(filename='squirrel.png', n=60)
