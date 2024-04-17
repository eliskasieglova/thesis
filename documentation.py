# documentation of what i did
import pandas as pd
import geopandas as gpd
import read_icesat, analysis, preprocessing, plotting
from shapely import wkt

# READING DATA
# read ATL03, ATL06, ATL08, ATL08QL from .h5 files using the library h5py (.h5 -> df -> .csv in '/data')
# what the .csv contains:
#   - selected variables extracted from the .h5 files
#   - added a column 'product' with the product name if i wanted to merge the datasets later
#   - converted lat, lon (EPSG:4326) to easting, northing (EPSG:32633)

products = ['ATL03', 'ATL06', 'ATL08', 'ATL08QL']

# convert .h5 files to .csv
read_icesat.readATL06()
read_icesat.readATL08()
read_icesat.readATL08QL()

# load the data as df
atl06 = pd.read_csv('data/atl06.csv')
atl08 = pd.read_csv('data/atl08.csv')
atl08ql = pd.read_csv('data/atl08ql.csv')

# PREPROCESSING
# - create dh from h using a vrt DEM
#   - copied np_dems from svalbardsurges
#   - adapted icesatDEMdifference() from svalbardsurges
# - converted dataframe to geodataframe and saved as .gpkg ('/data')
# - merged data

# convert h to dh
atl06dh = preprocessing.dh(atl06)
atl08dh = preprocessing.dh(atl08)
atl08qldh = preprocessing.dh(atl08ql)

# correct geometry and convert to geodataframe
atl06 = preprocessing.xy2geom(atl06dh, 'easting', 'northing', 'geometry')
atl08 = preprocessing.xy2geom(atl08dh, 'easting', 'northing', 'geometry')
atl08ql = preprocessing.xy2geom(atl08qldh, 'easting', 'northing', 'geometry')

atl06 = gpd.GeoDataFrame(atl06dh, geometry='geometry', crs='EPSG:32633')
atl08 = gpd.GeoDataFrame(atl08dh, geometry='geometry', crs='EPSG:32633')
atl08ql = gpd.GeoDataFrame(atl08qldh, geometry='geometry', crs='EPSG:32633')

# save geodataframes
atl06.to_file('data/atl06dh.gpkg')
atl08.to_file('data/atl08dh.gpkg')
atl08ql.to_file('data/atl08qldh.gpkg')

# merge the data
merged = pd.concat([atl06, atl08])
merged = pd.concat([merged, atl08ql])
merged.to_file('data/icesat.gpkg')

# COMPARING DATASETS
# before beginning the analysis we want to visualize the data we have from the different datasets
# we merge the different products to one big dataset and create a subset for heerland (a lot of surges there)
# and then we clip the data to two glaciers: Scheelebreen (surging) and Bakaninbreen (not surging)
# then we visualize these two glaciers with all the datasets

# create subset for the area
heerland_bbox = [18.5, 15.5, 78, 77.4]
subset_heerland = analysis.subset([atl06, atl08, atl08ql], heerland_bbox)

# clip to glacier extents and save files
sb = gpd.read_file('data/temp/scheelebreen_dh.gpkg').to_crs('EPSG:32633')
bb = gpd.read_file('data/temp/bakaninbreen_dh.gpkg').to_crs('EPSG:32633')
sb_pts = analysis.clip(subset_heerland, sb)
bb_pts = analysis.clip(subset_heerland, bb)
sb_pts.to_file('data/temp/scheelebreen_pts.gpkg')
bb_pts.to_file('data/temp/bakaninbreen_pts.gpkg')

# create subsets for the two glaciers (sb = scheelebreen, bb = bakaninbreen)
bb06 = bb_pts[bb_pts['product'] == 'atl06']
bb08 = bb_pts[bb_pts['product'] == 'atl08']
bb08ql = bb_pts[bb_pts['product'] == 'atl08ql']
sb06 = sb_pts[sb_pts['product'] == 'atl06']
sb08 = sb_pts[sb_pts['product'] == 'atl08']
sb08ql = sb_pts[sb_pts['product'] == 'atl08ql']

# so now i plot the ATL06, ATL08 and ATL08QL data over each other to see what differences there
# are regarding amount of data points, similarity of them etc.
# - ATL08QL data --> less data points than ATL08
# - ATL06 data --> really really stupid noise, didn't really figure out what that is, maybe have to ask
# the result of this is stored in 'figs/comparison.png'
plotting.plotSBBB(sb06, bb06, sb08, bb08, sb08ql, bb08ql)

# now i have the files saves so i can just run this:
sb_pts = gpd.read_file('data/temp/scheelebreen_dh.gpkg').to_crs('EPSG:32633')
bb_pts = gpd.read_file('data/temp/bakaninbreen_dh.gpkg').to_crs('EPSG:32633')
bb06 = bb_pts[bb_pts['product'] == 'atl06']
bb08 = bb_pts[bb_pts['product'] == 'atl08']
bb08ql = bb_pts[bb_pts['product'] == 'atl08ql']
sb06 = sb_pts[sb_pts['product'] == 'atl06']
sb08 = sb_pts[sb_pts['product'] == 'atl08']
sb08ql = sb_pts[sb_pts['product'] == 'atl08ql']
plotting.plotSBBB(sb06, bb06, sb08, bb08, sb08ql, bb08ql)

# filtering ATL06 points
# so far I filtered points on Scheelebreen based on the ATL08 and ATL08QL data like this:
#  1) created geom from (h, dh)
#  2) created buffer around ATL08 and ATL08QL pts
#  3) extracted ATL06 data within these points
# not sure this is the best way to do it....... todo: maybe try some ransac thing instead??

# filter the atl06 data
filtered_sb = preprocessing.filterATL06('data/temp/scheelebreen_dh.gpkg', 'data/temp/scheelebreen_filtered.gpkg')
filtered_bb = preprocessing.filterATL06('data/temp/bakaninbreen_dh.gpkg', 'data/temp/bakaninbreen_filtered.gpkg')

# i tried RANSAC filtering but was not much use:
filtered_ransac = preprocessing.filterATL06Ransac('data/temp/scheelebreen_dh.gpkg')

# let's stick with the upper form of filtering then and head on to the analysis!!!




# TESTING IT OUT ON A WHOLE AREA
# of Heerland

# select the glaciers from RGI within given bounds
analysis.selectGlaciers([18.5, 15.5, 78, 77.4])


# times of opening datasets:
#ATL08 csv: 1.5722465515136719, ATL08 gpkg: 114.23847365379333
#ATL06 csv: 4.699980974197388, ATL06 gpkg: 447.89306473731995
#.... using gpkg is really really bad for the big data, should only do it for counting dh


