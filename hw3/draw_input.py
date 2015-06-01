import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import sys

def __main__():
    minx = 101
    maxx = -1
    miny = 101
    maxy = -1
    xs = []
    ys = []
    size = int(sys.stdin.readline())
    data = sys.stdin.readlines()
    for i in range(size):
        t = data[i].split(' ')
        x = float(t[0])
        y = float(t[1])
        if (x < minx):
            minx = x
        if (x > maxx):
            maxx = x
        if (y < miny):
            miny = y
        if (y > maxy):
            maxy = y
        xs.append(x)
        ys.append(y)

    plt.plot(xs, ys, 'ro')
    plt.axis([minx, maxx, miny, maxy])
    plt.savefig(sys.stdout)
    plt.show()

__main__()
