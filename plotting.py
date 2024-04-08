import pandas as pd
import time
from matplotlib import pyplot as plt

def plotDataLoc():
    """
    Plots the datapoints (ATL06, ATL08, ATL08QL) to see what area they cover.
    Does not include ATL03 because it took forever to load...
    """
    start_time = time.time()
    # read data
    atl06 = pd.read_csv('data/atl06.csv')
    print('atl06 loaded', time.time() - start_time)
    atl08 = pd.read_csv('data/atl08.csv')
    print('atl08 loaded', time.time() - start_time)
    atl08ql = pd.read_csv('data/atl08ql.csv')
    print('atl08ql loaded', time.time() - start_time)

    # visualize where they are
    plt.subplots(2, 2)

    print('plotting atl06', time.time() - start_time)
    plt.subplot(2, 2, 2)
    plt.title('ATL06')
    plt.scatter(atl06['longitude'], atl06['latitude'], c=atl06['h'])

    print('plotting atl08', time.time() - start_time)
    plt.subplot(2, 2, 3)
    plt.title('ATL08')
    plt.scatter(atl08['longitude'], atl08['latitude'], c=atl08['h'])

    print('plotting atl08ql', time.time() - start_time)
    plt.subplot(2, 2, 4)
    plt.title('ATL08QL')
    plt.scatter(atl08ql['longitude'], atl08ql['latitude'], c=atl08ql['h'])
    print(time.time() - start_time)

    plt.tight_layout()
    plt.show()


def plotSBBB(sb06dh, bb06dh, sb08dh, bb08dh, sb08qldh, bb08qldh):
    # plot
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    plt.title('Scheelebreen')
    plt.scatter(sb06dh.h, sb06dh.dh, s=2, c='brown', label='atl06')
    plt.scatter(sb08dh.h, sb08dh.dh, s=2, c='darkblue', label='atl08')
    plt.scatter(sb08qldh.h, sb08qldh.dh, s=2, c='orange', label='atl08ql')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Bakaninbreen')
    plt.scatter(bb06dh.h, bb06dh.dh, s=2, c='brown', label='atl06')
    plt.scatter(bb08dh.h, bb08dh.dh, s=2, c='darkblue', label='atl08')
    plt.scatter(bb08qldh.h, bb08qldh.dh, s=2, c='orange', label='atl08ql')
    plt.legend()

    plt.show()

