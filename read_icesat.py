from pathlib import Path
import h5py
import os
import pandas as pd
from datetime import timedelta, datetime
from pyproj import Proj


def readICESat(icesat_product):

    if icesat_product == 'ATL03':
        readATL03()
    elif icesat_product == 'ATL06':
        readATL06()
    elif icesat_product == 'ATL08':
        readATL08()
    elif icesat_product == 'ATL08QL':
        readATL08QL()
    else:
        print('invalid product')


def readATL06():
    path = Path('C:/Users/eliss/Documents/SvalbardSurges/is2_ATL06')

    data = pd.DataFrame()

    # list through directory with downloaded ATL06 data
    for filepath in os.listdir(path):

        # open file
        with h5py.File(path/filepath, 'r') as file:

            # create temporary data dictionary
            temp = pd.DataFrame()

            # loop through beams
            beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
            for beam in beams:
                temp_beam = pd.DataFrame()

                # skip if the group doesn't exist
                if f'{beam}/land_ice_segments' not in file:
                    continue

                # save the measured data
                temp_beam["h"] = list(file[beam]['land_ice_segments']['h_li'])
                temp_beam["longitude"] = list(file[beam]['land_ice_segments']['longitude'])
                temp_beam["latitude"] = list(file[beam]['land_ice_segments']['latitude'])
                temp_beam["quality_summary"] = list(file[beam]['land_ice_segments']['atl06_quality_summary'])
                temp_beam["delta_time"] = list(file[beam]['land_ice_segments']['delta_time'])

                # save the beam and pair
                temp_beam['beam'] = [beam] * len(temp_beam['h'])

                # set start time to 2018-01-01 according to ICESat-2 websites
                starttime = datetime(2018, 1, 1)

                # convert second doubles to time deltas and add start time
                timedeltas = []
                for i in temp_beam['delta_time']:
                    timedeltas.append(timedelta(seconds=i) + starttime)

                temp_beam["acquisition_time"] = timedeltas

                # at the end of each beam append to temp file df
                temp = temp.append(temp_beam)

            # at the end of each file append to the data
            data = data.append(temp)

    # convert latitude and longitude to UTM
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    easting, northing = myproj(data[f'longitude'], data[f'latitude'])
    data['easting'] = easting
    data['northing'] = northing

    # assign product name
    data['product'] = 'atl06'

    # save data as csv
    data.to_csv('data/atl06.csv')

    # todo save as netcdf
    #ds = xr.Dataset.from_dataframe(data)
    #ds.to_netcdf('data/atl06.nc')

    return


def readATL08():
    path = Path('C:/Users/eliss/Documents/SvalbardSurges/is2_ATL08')

    data = pd.DataFrame()

    # list through directory with downloaded ATL06 data
    for filepath in os.listdir(path):

        # open file
        with h5py.File(path/filepath, 'r') as file:

            # create temporary data dictionary
            temp = pd.DataFrame()

            # loop through beams
            beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
            for beam in beams:
                temp_beam = pd.DataFrame()

                # skip if the group doesn't exist
                if f'{beam}/land_segments' not in file:
                    continue

                # save the measured data
                temp_beam["h"] = list(file[beam]['land_segments']['terrain']['h_te_best_fit'])
                temp_beam["longitude"] = list(file[beam]['land_segments']['longitude'])
                temp_beam["latitude"] = list(file[beam]['land_segments']['latitude'])
                temp_beam["delta_time"] = list(file[beam]['land_segments']['delta_time'])

                # save the beam and pair
                temp_beam['beam'] = [beam] * len(temp_beam['h'])

                # set start time to 2018-01-01 according to ICESat-2 websites
                starttime = datetime(2018, 1, 1)

                # convert second doubles to time deltas and add start time
                timedeltas = []
                for i in temp_beam['delta_time']:
                    timedeltas.append(timedelta(seconds=i) + starttime)

                temp_beam["acquisition_time"] = timedeltas

                # at the end of each beam append to temp file df
                temp = temp.append(temp_beam)

            # at the end of each file append to the data
            data = data.append(temp)

    # convert latitude and longitude to UTM
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    easting, northing = myproj(data[f'longitude'], data[f'latitude'])
    data['easting'] = easting
    data['northing'] = northing

    # assign product name
    data['product'] = 'atl08'

    # save data as csv
    data.to_csv('data/atl08.csv')

    # todo save as netcdf
    #ds = xr.Dataset.from_dataframe(data)
    #ds.to_netcdf('data/atl06.nc')

    return


def readATL03():
    path = Path('C:/Users/eliss/Documents/SvalbardSurges/is2_ATL03')

    data = pd.DataFrame()

    # list through directory with downloaded ATL06 data
    for filepath in os.listdir(path):

        # open file
        with h5py.File(path/filepath, 'r') as file:

            # create temporary data dictionary
            temp = pd.DataFrame()

            # loop through beams
            beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
            for beam in beams:
                temp_beam = pd.DataFrame()

                # skip if the group doesn't exist
                if f'{beam}/heights' not in file:
                    continue

                # save the measured data
                temp_beam["h"] = list(file[beam]['heights']['h_ph'])
                temp_beam["longitude"] = list(file[beam]['heights']['lon_ph'])
                temp_beam["latitude"] = list(file[beam]['heights']['lat_ph'])
                temp_beam["delta_time"] = list(file[beam]['heights']['delta_time'])
                temp_beam["quality"] = list(file[beam]['heights']['quality_ph'])


                # save the beam and pair
                temp_beam['beam'] = [beam] * len(temp_beam['h'])

                # set start time to 2018-01-01 according to ICESat-2 websites
                starttime = datetime(2018, 1, 1)

                # convert second doubles to time deltas and add start time
                timedeltas = []
                for i in temp_beam['delta_time']:
                    timedeltas.append(timedelta(seconds=i) + starttime)

                temp_beam["acquisition_time"] = timedeltas

                # at the end of each beam append to temp file df
                temp = temp.append(temp_beam)

            # at the end of each file append to the data
            data = data.append(temp)

    # convert latitude and longitude to UTM
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    easting, northing = myproj(data[f'longitude'], data[f'latitude'])
    data['easting'] = easting
    data['northing'] = northing

    # assign product name
    data['product'] = 'atl03'

    # save data as csv
    data.to_csv('data/atl03.csv')

    # todo save as netcdf
    #ds = xr.Dataset.from_dataframe(data)
    #ds.to_netcdf('data/atl06.nc')

    return


def readATL08QL():
    path = Path('C:/Users/eliss/Documents/SvalbardSurges/QLdata')

    data = pd.DataFrame()

    # list through directory with downloaded ATL06 data
    for filename in os.listdir(path):

        if filename.endswith('.xml'):
            continue

        # open file
        with h5py.File(path / filename, 'r') as file:

            # create temporary data dictionary
            temp = pd.DataFrame()

            # loop through beams
            beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
            for beam in beams:
                temp_beam = pd.DataFrame()

                # skip if the group doesn't exist
                if f'{beam}/land_segments' not in file:
                    continue

                # variables that i want
                land_segments = ['longitude', 'longitude_20m', 'latitude', 'latitude_20m',
                                 'delta_time', 'msw_flag', 'sigma_h', 'cloud_flag_atm']
                terrain = ['h_te_best_fit', 'h_te_best_fit_20m']
                #signal_photons = ['ph_h', 'classed_pc_flag', 'd_flag']

                # save the chosen variables to the temp file
                for var in land_segments:
                    temp_beam[var] = list(file[beam]['land_segments'][var])

                for var in terrain:
                    if var == 'h_te_best_fit':  # rename elevation variable to 'h'
                        temp_beam['h'] = list(file[beam]['land_segments']['terrain'][var])
                    else:
                        temp_beam[var] = list(file[beam]['land_segments']['terrain'][var])

                #for var in signal_photons:
                #    temp_beam[var] = list(file[beam]['signal_photons'][var])

                # save the beam and pair
                temp_beam['beam'] = [beam] * len(temp_beam['h'])

                # set start time to 2018-01-01 according to ICESat-2 websites
                starttime = datetime(2018, 1, 1)

                # convert second doubles to time deltas and add start time
                timedeltas = []
                for i in temp_beam['delta_time']:
                    timedeltas.append(timedelta(seconds=i) + starttime)

                temp_beam["acquisition_time"] = timedeltas

                # at the end of each beam append to temp file df
                temp = temp.append(temp_beam)

            # at the end of each file append to the data
            data = data.append(temp)

    # convert latitude and longitude to UTM
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    easting, northing = myproj(data[f'longitude'], data[f'latitude'])
    data['easting'] = easting
    data['northing'] = northing

    # save data as csv
    data.to_csv('data/atl08ql.csv')

    # assign product name
    data['product'] = 'atl08ql'

    # todo save as netcdf
    # ds = xr.Dataset.from_dataframe(data)
    # ds.to_netcdf('data/atl08ql.nc')

    return

