import numpy as np


def get_np_subgrid_from_xy_coords(x, y, w=3):
    x = int(x)
    y = int(y)
    xx = np.arange(x - w, x + w + 1)
    yy = np.arange(y - w, y + w + 1)
    the_x = []
    the_y = []
    for xxx in xx:
        for yyy in yy:
            the_x.append(xxx)
            the_y.append(yyy)
    return the_x, the_y
