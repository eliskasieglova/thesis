import pandas as pd
import preprocessing
import geopandas as gpd
import rasterio as rio
import numpy as np
from pyproj import Proj


def subset(dfs, bbox):
    """
    Create subset of data points.

    :param bbox: bbox of subset area in format list[east, west, north, south]
    :return:
    """

    # extract bounds from bounding box
    east = bbox[0]
    west = bbox[1]
    north = bbox[2]
    south = bbox[3]

    subset = pd.DataFrame()

    for df in dfs:
        cond1 = df['longitude'] < east
        cond2 = df['longitude'] > west
        cond3 = df['latitude'] < north
        cond4 = df['latitude'] > south

        df = df.where(cond1 & cond2 & cond3 & cond4).dropna()
        subset = pd.concat([subset, df])

    return subset


def clip(df, shp):

    # convert df to gdf
    gdf = preprocessing.xy2geom(df)

    # clip
    clipped = gpd.clip(gdf, shp)

    return clipped


def selectGlaciers(bbox):
    """
    Selects glaciers from Randolph Glacier Inventory (RGI) inside given bounding box.

    :param bbox: Bounding box of subset area in format list[east, west, north, south].

    :returns: Subset of RGI.
    """

    # load Randolph Glacier Inventory
    rgi = gpd.read_file('data/rgi.gpkg').to_crs('EPSG:32633')

    # create subset by bbox
    subset = rgi.cx[bbox[1]:bbox[0], bbox[3]:bbox[2]]

    # convert bbox lat lon to easting northing
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")  # assign projection
    eastings, northings = myproj((bbox[0], bbox[1]), (bbox[2], bbox[3]))

    # convert
    from shapely.geometry import Polygon
    polygon = Polygon([(eastings[0], northings[0]), (eastings[1], northings[0]), (eastings[1], northings[1]), (eastings[0], northings[1]), (eastings[0], northings[0])])
    bbox = gpd.GeoDataFrame(gpd.GeoSeries(polygon), columns=['geometry'], crs='EPSG:32633')

    # go glacier by glacier and determine if it's in bbox or not
    mask = []
    for i, row in rgi.iterrows():
        if row['geometry'].within(bbox)['geometry'][0]:
            mask.append(True)
        else:
            mask.append(False)

    # mask it out!!!
    rgi['mask'] = mask
    subset = rgi[rgi['mask'] == True]

    # if glaciers are too small just throw them out
    subset = subset[subset['geometry'].area / 1e6 > 15]

    # save it
    subset.to_file('data/temp/rgi_heerland.gpkg')

    return subset



