from shapely.geometry import Point
import geopandas as gpd
from pyproj import Proj
import rasterio as rio
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from shapely.ops import unary_union
import pandas as pd


def latlon2UTM(df):
    """
    Convert latitude and longitude to easting and northing.
    :param df: input df with 'lat', 'lon'
    :return: new df with columns 'easting', 'northing'
    """

    # set projection parameters
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    # convert
    easting, northing = myproj(df[f'longitude'], df[f'latitude'])

    # add columns to df
    df['easting'] = easting
    df['northing'] = northing

    return df


def xy2geom(df, x_column, y_column, geom_column):
    """
    Makes new geometry column from easting and northing and converts df to gdf in crs UTM zone 33.

    :param df: input dataframe (ICESat-2 data with columns easting and northing)

    :return: geodataframe in EPSG:32633
    """

    # create geometry from easting, northing
    df[geom_column] = [Point(xy) for xy in zip(df[x_column], df[y_column])]

    # convert df to df
    gdf = gpd.GeoDataFrame(df).set_crs('EPSG:32633')

    return gdf


def dh(data):
    # path to DEM mosaic vrt in SvalbardSurges (had to copy folder cache/NP_DEMs to current folder for this to work)
    dem_path = 'C:/Users/eliss/Documents/SvalbardSurges/data/npi_vrts/npi_mosaic.vrt'

    with rio.open(dem_path) as raster:
        data["dem_elevation"] = list(np.fromiter(
            raster.sample(
                np.transpose([data.easting.values, data.northing.values]),
                masked=True
            ),
            dtype=raster.dtypes[0],
            count=data.easting.shape[0]
        ))

    # subtract ICESat-2 elevation from DEM elevation (with elevation correction)
    data["dh"] = data["h"] - data["dem_elevation"] - 31.55
    data["dh_uncorr"] = data["h"] - data["dem_elevation"]
    # todo a bit better correction

    # get rid of nan values in ATL06
    data = data[data['h'] < 3000]

    return data


def filterATL06(datapath, outpath):

    data = gpd.read_file(datapath)

    # create new geometry (h, dh) so that i can make buffer etc.
    data = xy2geom(data, 'h', 'dh', 'computing_geom')

    # split the data into reference data and filter data
    ref_data = data[data['product'] != 'atl06']
    ref_data = gpd.GeoDataFrame(ref_data, geometry='computing_geom')
    filter_data = data[data['product'] == 'atl06']
    filter_data = gpd.GeoDataFrame(filter_data, geometry='computing_geom')

    # create buffer around the reference data
    buffers = ref_data.buffer(20)
    buff = gpd.GeoSeries(unary_union(buffers))

    # loop through points and create mask list of True/False (in/out)
    mask = []
    for i, row in filter_data.iterrows():
        if row['computing_geom'].intersects(buff)[0]:
            mask.append(True)
        else:
            mask.append(False)

    # filter data based on mask
    filter_data['mask'] = mask
    filtered_data = filter_data[filter_data['mask'] == True]

    # merge the datasets
    merge_result = pd.concat([ref_data, filtered_data])

    # reset geometries
    outdata = gpd.GeoDataFrame(merge_result, geometry='geometry', crs='EPGS:32633')

    # save
    outdata.to_file(outpath)

    return outdata


def filterATL06Ransac(inpath):

    # read data
    data = gpd.read_file(inpath)

    # only select atl06
    data = data[data['product'] == 'atl06']

    lw = 2  # linewidth for plots

    # reshape arrays
    X = data.h.values.reshape(-1, 1)
    y = data.dh.values.reshape(-1, 1)

    # Robustly fit data with ransac algorithm
    ransac = linear_model.RANSACRegressor(max_trials=100)
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    # PLOT
    plt.scatter(X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
    plt.scatter(X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers")
    plt.plot(line_X, line_y_ransac, color="cornflowerblue", linewidth=lw, label="RANSAC regressor")
    plt.title('ATL06: RANSAC filtering')
    plt.legend()
    plt.show()

    # ransac coefficient
    coef = ransac.estimator_.coef_[0][0]

    # plt.title(f'{str(year - 1)[:4]}-{str(year)[:4]}, {glacier_name}, {algorithm}, {coef}')
    # plt.xlabel("Input")
    # plt.ylabel("Response")

    return

