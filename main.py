import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import read_icesat, analysis, preprocessing, plotting, download, management
import rasterio as rio
import numpy as np
from sklearn import linear_model
from pyproj import Proj
import xarray as xr
import os
from pathlib import Path
from vars import label, spatial_extent

important_ids = [('Scheelebreen', 'G016964E77694N'), ('Bakaninbreen', 'G017525E77773N')]

# so now i plotted the ATL06, ATL08 and ATL08QL data over each other to see what differences there
# are regarding amount of data points, similarity of them etc.
# ATL08QL data --> less data points than ATL08
# ATL06 data --> really really stupid noise, didn't really figure out what that is, maybe have to ask

# todo:
#  - start analysis (threshold method by years)

# todo later:
#  investigate options with ATL03

#download.downloadSvalbard()

rerun = False
if rerun:
    # read icesat data (one .csv file or pd dataframe with all the data products for the whole of Svalbard)
    print('reading icesat')
    data = read_icesat.readICESat(['ATL06', 'ATL08'])

    # crop icesat data to Heerland
    print(f'subsetting icesat to {label}')
    icesat = analysis.subsetICESat(data, spatial_extent)

    # select glaciers from RGI for heerland
    print('subsetting rgi')
    rgi = analysis.selectGlaciers(spatial_extent)

    # list glacier ids to loop through
    glacier_ids = management.listGlacierIDs(rgi)
    total = len(glacier_ids)

    # convert icesat to gdf
    print('converting icesat to gdf')
    icesat = preprocessing.pointsToGeoDataFrame(icesat)

    # 1) create glacier subsets
    i = 1
    for glacier_id in glacier_ids:

        print(f'{i}/{total} ({glacier_id})')
        i = i+1

        # load shapefile for glacier
        glacier = management.loadGlacierShapefile(glacier_id)

        # subset icesat by glacier (and normalize the data)
        print(f'clipping {glacier_id}')
        clipped = analysis.clip(icesat, glacier)

        if clipped == 'nodata':
            print('nodata')
            glacier_ids.remove(glacier_id)
            continue

        # filter ATL06 data
        print(f'filtering ATL06 {glacier_id}')
        preprocessing.filterATL06(glacier_id)

    # 2) group data by hydrological years
    # list years i have in data
    # create date value in data
    print('grouping by hydro years')
    data['date'] = [i[:10] for i in data['acquisition_time'].values]
    years = list(np.unique([i[:4] for i in data['date'].values]))

    for glacier_id in glacier_ids:

        print(glacier_id)

        # open pts for glacier
        try:
            data = pd.read_csv(f'data/temp/glaciers/{glacier_id}_filtered.csv')
        except:
            continue
        data['date'] = [i[:10] for i in data['acquisition_time'].values]

        # loop through years
        for year in years:
            print(year)
            # select subset of data for given year (hydrological year)
            analysis.groupByHydroYear(data, year, glacier_id)

#analysis.runFeatureExtraction()
#analysis.decisionTree()

# todo:
#  - gather information about glacier surges (std, lower max, lin coef)
#  - make training dataset from surging/nonsurging glaciers
#  - threshold analysis based on known characteristics
#  - supervised RF based on dataset

def RF():
    """
    Create training dataset, so far for the southern part.

    :param data: dataframe with extracted features, glacier_id, geometry

    :return:
    """
    # import my data
    data = gpd.read_file(f'data/temp/{label}_features.gpkg')
    data['id'] = data['glacier_id'] + '_' + data['year'].astype(str)

    # read training data
    training_data = pd.read_csv('data/data/trainingdata.csv')

    # drop features
    training_data = training_data.drop(columns=['max_dh', 'max_binned', 'slope_binned', 'slope_lower', 'slope'])

    # select the data subset
    training_data = pd.merge(data, training_data, on=['glacier_id', 'year'])
    training_data['id'] = training_data['glacier_id'] + '_' + training_data['year'].astype(str)
    training_data = training_data[['id', 'dh_max', 'dh_min', 'dh_mean', 'dh_std', 'lin_coef',
                                   'dh_max_lower', 'dh_mean_lower', 'dh_min_lower', 'dh_std_lower', 'surging']]
    input_data = data[['id', 'dh_max', 'dh_min', 'dh_mean', 'dh_std', 'lin_coef',
                                   'dh_max_lower', 'dh_mean_lower', 'dh_min_lower', 'dh_std_lower',
                 'glacier_id', 'year']]

    result = analysis.classifyRF(input_data, training_data)

    result = pd.merge(data, result, on=['glacier_id', 'year'])
    result = result[['glacier_id', 'year', 'surging', 'geometry']]
    result = gpd.GeoDataFrame(result)
    result.to_file('data/temp/RFresults.gpkg')

    return result


RF()



