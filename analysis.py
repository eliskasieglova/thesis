import pandas as pd
import preprocessing
import geopandas as gpd
import rasterio as rio
import numpy as np
from shapely import wkt
from pathlib import Path
import time
from pyproj import Proj
from vars import label, spatial_extent
from sklearn import linear_model
import management


def subsetICESat(df, bbox):
    """
    Create subset of data points.

    :param dfs: dataframes in list
    :param bbox: bbox of subset area in format list[east, west, north, south]

    :return: DataFrame of points within the given bounding box. Merged from all the input DataFrames.
    """

    outpath = Path(f'data/data/ICESat_{label}.csv')

    # cache
    if outpath.is_file():
        return pd.read_csv(outpath)

    # extract bounds from bounding box
    west = bbox[0]
    south = bbox[1]
    east = bbox[2]
    north = bbox[3]

    # create subset based on conditions
    subset = df[df['longitude'] < east]
    subset = subset[subset['longitude'] > west]
    subset = subset[subset['latitude'] < north]
    subset = subset[subset['latitude'] > south]

    # save as .csv file
    subset.to_csv(outpath)

    return subset

glaciers_nodata = []

def clip(pts, glacier):

    glacier_id = list(glacier['glims_id'].values)[0]
    outpath = Path(f'data/temp/glaciers/{glacier_id}_icesat.gpkg')

    # cache
    if outpath.is_file():
        return

    # clip the points to the glacier outline
    clipped = gpd.clip(pts, glacier)

    if len(clipped) < 15:
        glaciers_nodata.append(glacier_id)
        return 'nodata'

    from matplotlib import pyplot as plt
    plt.scatter(clipped.h, clipped.dh)
    plt.title(glacier_id)
    plt.savefig(f'data/temp/figs/{glacier_id}_clipped.png')
    plt.close()

    # normalize
    data = normalize(clipped)

    # save as .csv file
    data.to_file(outpath)
    return outpath


def selectGlaciers(bbox):
    """
    Selects glaciers from Randolph Glacier Inventory (RGI) inside given bounding box.

    :param bbox: Bounding box of subset area in format list[east, west, north, south].

    :returns: Subset of RGI.
    """

    outpath = Path(f'data/data/rgi_{label}.gpkg')

    # cache
    if outpath.is_file():
        subset = gpd.read_file(outpath)
        return subset

    # load Randolph Glacier Inventory
    rgi = gpd.read_file('data/data/rgi.gpkg').to_crs('EPSG:32633')

    # create subset by bbox (not used bc it included glaciers that only overlapped and were not completely within)
    #subset = rgi.cx[bbox[1]:bbox[0], bbox[3]:bbox[2]]

    # convert bbox lat lon to easting northing
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")  # assign projection
    eastings, northings = myproj((bbox[0], bbox[2]), (bbox[1], bbox[3]))

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
    subset.to_file(outpath)

    return subset



def groupByHydroYear(data, year, glacier_id):
    """
    Groups data by given hydrological year. Saves the data as .csv to temp folder.
    """

    year = int(year)
    output_path = Path(f'data/temp/glaciers/{glacier_id}_{year}.csv')
    output_gpkg = Path(f'data/temp/glaciers/{glacier_id}_{year}.gpkg')

    # cache
    if output_gpkg.is_file():
        return

    # split data into day, month, year
    data['day'] = [int(i[8:10]) for i in data['date'].values]
    data['month'] = [int(i[5:7]) for i in data['date'].values]
    data['year'] = [int(i[0:4]) for i in data['date'].values]

    # select data that belong in hydrological year
    subset = data[((data['year'] == year - 1) & (data['month'] > 10)) |
                  ((data['year'] == year) & (data['month'] < 11))]

    # save as csv and gpkg
    subset.to_csv(output_path)
    subset['geometry'] = subset['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(subset, geometry='geometry', crs='EPSG:32633')
    gdf.to_file(Path(f'data/temp/glaciers/{glacier_id}_{year}.gpkg'))

    return


def normalize(data):
    """
    Normalize h and dh in ICESat-2 data.

    params
    ------
    - inpath
        input path (Path object)
    - outpath
        output path (Path object)

    returns
    -------
    output path
    """

    # normalize dh
    min_dh = min(data.dh.values)
    max_dh = max(data.dh.values)
    data['dh_norm'] = (data['dh'] - min_dh) / (max_dh - min_dh)

    # normalize h
    min_h = min(data.h.values)
    max_h = max(data.h.values)
    data['h_norm'] = (data['h'] - min_h) / (max_h - min_h)

    return data



def countTimeOfReadingData():
    start = time.time()
    csv = pd.read_csv('C:/Users/eliss/Documents/diplomka/data/temp/ATL08.csv')
    csv_duration = time.time() - start

    start2 = time.time()
    gpkg = gpd.read_file('C:/Users/eliss/Documents/diplomka/data/data/ATL08.gpkg')
    gpkg_duration = time.time() - start2

    print('ATL08 csv:')
    print(csv_duration)
    print('ATL08 gpkg:')
    print(gpkg_duration)

    start = time.time()
    csv = pd.read_csv('C:/Users/eliss/Documents/diplomka/data/temp/ATL06.csv')
    csv_duration = time.time() - start

    start2 = time.time()
    gpkg = gpd.read_file('C:/Users/eliss/Documents/diplomka/data/data/ATL06.gpkg')
    gpkg_duration = time.time() - start2

    print('ATL06 csv:')
    print(csv_duration)
    print('ATL06 gpkg:')
    print(gpkg_duration)



def linRegAlg(data, glacier_name):

    if data.index.size == 0:
        return 'nodata'

    # reshape arrays
    X = data.h.values.reshape(-1, 1)
    y = data.dh.values.reshape(-1, 1)

    # Fit line
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)

    # plot
    # plt.scatter(X, y, color="yellowgreen", marker='.')
    # plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")

    # coefficient
    coef = lr.coef_[0][0]
    return X, y, line_X, line_y, coef


def linreg(data):

    if data.index.size == 0:
        return np.nan

    # reshape arrays
    X = data.h.values.reshape(-1, 1)
    y = data.dh.values.reshape(-1, 1)

    # Fit line
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    #line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    #line_y = lr.predict(line_X)

    coef = lr.coef_[0][0]

    #plt.scatter(X, y, color="orange", marker='.', s=2)
    #plt.plot(line_X, line_y, color="navy", linewidth=2, label="Linear regressor")
    #ax.invert_xaxis()

    return coef


def featureExtraction(glacier_ids, years):
    """
    Extract features relevant for detecting surging glaciers.
    :param data:
    :return:
    """

    glacier_ids_result = []
    years_result = []
    geometries = []
    dh_means = []
    dh_mins = []
    dh_maxs = []
    dh_stds = []
    lin_coefs = []
    dh_means_lower = []
    dh_mins_lower = []
    dh_maxs_lower = []
    dh_stds_lower = []
    lin_coefs_lower = []

    for glacier_id in glacier_ids:
        for year in years:
            # append geometry
            try:
                data = gpd.read_file(f'data/temp/glaciers/{glacier_id}_{year}.gpkg')
            except:
                continue
            print(glacier_id, year)
            # append glacier id and year
            glacier_ids_result.append(glacier_id)
            years_result.append(year)
            from shapely.geometry import MultiPoint
            icesat_multipoint = MultiPoint(data['geometry']).wkt
            geometries.append(icesat_multipoint)

            # count all the mins and maxes etc. on the whole glacier
            dh_mins.append(data['dh'].min())
            dh_maxs.append(data['dh'].max())
            dh_means.append(data['dh'].mean())
            dh_stds.append(data['dh'].std())

            # lower part
            data_lower = data[data['h_norm'] > 0.5]
            dh_mins_lower.append(data['dh'].min())
            dh_maxs_lower.append(data['dh'].max())
            dh_means_lower.append(data['dh'].mean())
            dh_stds_lower.append(data['dh'].std())

            # linear regression
            try:
                lin_coefs.append(linreg(data))
            except:
                lin_coefs.append(999)
            try:
                lin_coefs_lower.append(linreg(data_lower))
            except:
                lin_coefs_lower.append(999)

    variables = {
        'glacier_id': glacier_ids_result,
        'year': years_result,
        'geometry': geometries,
        'dh_max': dh_maxs,
        'dh_min': dh_mins,
        'dh_mean': dh_means,
        'dh_std': dh_stds,
        'dh_mean_lower': dh_means_lower,
        'dh_min_lower': dh_mins_lower,
        'dh_max_lower': dh_maxs_lower,
        'dh_std_lower': dh_stds_lower,
        'lin_coef': lin_coefs,
        'lin_coef_lower': lin_coefs_lower
    }

    results = pd.DataFrame.from_dict(variables)

    return results




def runFeatureExtraction():
    rgi = selectGlaciers(spatial_extent)
    # list glacier ids to loop through
    glacier_ids = management.listGlacierIDs(rgi)

    years = [2018, 2019, 2020, 2021, 2022, 2023]

    # extract features (min, max, linreg coefs etc. for each glacier)
    print('extracting features')
    glacier_features = featureExtraction(glacier_ids, years)

    print('exporting')
    rgi = rgi.rename(columns={"glims_id" : "glacier_id"})
    c = glacier_features.merge(rgi, how='left', on='glacier_id')
    c = c.drop(columns=['lin_coef_lower', 'geometry_x'])
    c = c.dropna()
    gdf = gpd.GeoDataFrame(c, geometry='geometry_y')
    gdf.to_file(f'data/temp/{label}_features.gpkg', geometry='geometry_y')





def decisionTree():
    # so now i extracted features from the datasets and have them all collected in the dataset 'glacier_features'
    data = gpd.read_file(f'data/temp/{label}_features.gpkg')

    # new geodataframe with glacier_id, geometry
    gdf = gpd.GeoDataFrame()
    gdf['glacier_id'] = list(data['glacier_id'])
    gdf['year'] = list(data['year'])
    gdf['geometry'] = list(data['geometry'])

    dh_max_lower = [1 if i > 25 else 0 for i in data['dh_max_lower']]
    lin_coef = [1 if i < 0.025 else 0 for i in data['lin_coef']]
    dh_std = [1 if i > 10 else 0 for i in data['dh_std']]

    gdf['dh_max_lower'] = dh_max_lower
    gdf['lin_coef'] = lin_coef
    gdf['dh_std'] = dh_std

    gdf['surging'] = gdf['dh_max_lower'] * gdf['lin_coef'] * gdf['dh_std']
    gdf['surging_sum'] = gdf['dh_max_lower'] + gdf['lin_coef'] + gdf['dh_std']

    gdf = gdf.set_geometry('geometry')
    gdf = gdf.set_crs(crs='EPSG:32633')

    # split by years
    y1 = gdf[gdf['year'] == 2018]
    y2 = gdf[gdf['year'] == 2019]
    y3 = gdf[gdf['year'] == 2020]
    y4 = gdf[gdf['year'] == 2021]
    y5 = gdf[gdf['year'] == 2022]
    y6 = gdf[gdf['year'] == 2023]

    y1.to_file(f'data/temp/{label}_2020_thresholds.gpkg', geometry='geometry')
    y2.to_file(f'data/temp/{label}_2021_thresholds.gpkg', geometry='geometry')
    y3.to_file(f'data/temp/{label}_2022_thresholds.gpkg', geometry='geometry')



def classifyRF(data, training_data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from scipy.stats import randint
    from sklearn.tree import export_graphviz

    # remove nans from datasets
    data = data.dropna(axis='index')
    training_data = training_data.dropna(axis='index')

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['id', 'glacier_id', 'year'])
    #y = df.surging_rf.replace({True: 1, False: 0})

    # split training dataset into features and target
    X_train = training_data.drop(columns=['id', 'surging'])
    y_train = training_data.surging

    # fitting and evaluating the model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # evaluate the model by comparison with actual data
    y_pred = rf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(accuracy)

    X['surging'] = rf.predict(X)
    result = pd.concat([data, X], axis=1, join='inner')

    # only return surge/not surge
    subset = result[['id', 'surging', 'glacier_id', 'year']]

    return subset



