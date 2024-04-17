import os
from vars import label
import geopandas as gpd
import pandas as pd
from pathlib import Path

def createFolder(path):
    """create new folder"""

    # check if folder already exists
    if not os.path.exists(path):
        # if not, make folder
        os.mkdir(path)

    return


# get glacier ids
def listGlacierIDs(glaciers):
    glacier_ids = []

    for i in range(len(glaciers)):
        row = glaciers.iloc[i]
        glacier_id = row['glims_id']
        glacier_ids.append(glacier_id)

    return glacier_ids


def loadGlacierShapefile(glacier_id):
    outpath = Path(f'data/temp/glaciers/{glacier_id}.gpkg')

    # cache
    if outpath.is_file():
        return gpd.read_file(outpath)

    # load the RGI shapefile
    rgi = gpd.read_file(f'data/data/rgi_{label}.gpkg')

    # select glacier with input glacier_id
    glacier = rgi[rgi['glims_id'] == glacier_id]

    # save this glacier
    glacier.to_file(outpath)

    return glacier


