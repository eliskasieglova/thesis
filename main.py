import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import read_icesat, analysis, preprocessing, plotting
import rasterio as rio
import numpy as np
from sklearn import linear_model
from pyproj import Proj


important_ids = [('Scheelebreen', 'G016964E77694N'), ('Bakaninbreen', 'G017525E77773N')]

# so now i plotted the ATL06, ATL08 and ATL08QL data over each other to see what differences there
# are regarding amount of data points, similarity of them etc.
# ATL08QL data --> less data points than ATL08
# ATL06 data --> really really stupid noise, didn't really figure out what that is, maybe have to ask

# todo:
#  1) create a system in the folders on how to save the files etc.
#  2) make a def preprocessing() and run the whole thing on heerland
#       --> that means that as a result we will have
#       1) for the area: dh pts
#       2) for each glacier: filtered dh
#  3) caching
#  4) group by years
#  5) start analysis (threshold method by years)

# todo later:
#  investigate options with ATL03









