import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def __main__():
    fin = open("cycle.in", "r")
    solin = open("cycle.out", "r")

    minx = 101
    maxx = -1
    miny = 101
    maxy = -1
    verts = []
    ordered_verts = []
    size = int(fin.readline())
    for i in range(size):
        t = fin.readline().split(' ')
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
        verts.append( (x, y) )
    
    t = solin.readline().split(' ')
    for i in t:
      idx = int(i) - 1
      ordered_verts.append( verts[idx] )

    ordered_verts.append( (0,0) )
    codes = [Path.MOVETO] + [Path.LINETO]*(size - 1) + [Path.CLOSEPOLY]

    path = Path(ordered_verts, codes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(minx - 1, maxx + 1)
    ax.set_ylim(miny - 1, maxy + 1)
    plt.savefig("simple_path.png")

__main__()
