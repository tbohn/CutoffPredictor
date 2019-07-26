# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
import requests
from . import table_columns as tc

# Integrity check on occupant table
def occupant_integrity(df_occupant):

    past = pd.Timestamp('1800-01-01')
    future = pd.Timestamp('2099-12-31')
    k = 0
    for loc in df_occupant['location_id'].unique():
        dfo = df_occupant.loc[df_occupant['location_id'] == loc]
        cis_occupant_id_list = np.asarray(dfo['cis_occupant_id'].unique())
        nOccs = len(cis_occupant_id_list)
        for i in range(nOccs):
            df = dfo.loc[dfo['cis_occupant_id'] == cis_occupant_id_list[i]] \
                .copy()
            if len(df) > 1:
                df = df.loc[~pd.isnull(df['move_out_date'])]
            df.loc[pd.isnull(df['move_in_date']), 'move_in_date'] = \
                str(past.date())
            df.loc[pd.isnull(df['move_out_date']), 'move_out_date'] = \
                str(future.date())
            df['move_in_date'] = pd.to_datetime(df['move_in_date'])
            df['move_out_date'] = pd.to_datetime(df['move_out_date'])
            if k == 0:
                df_occupant_new = df.copy()
            else:
                df_occupant_new = df_occupant_new.append(df, ignore_index=True,
                                                         sort=False)
        k += 1
    df_occupant = df_occupant_new

    return df_occupant


# Ensure consistent date formats in all tables
def date_consistency(df_occupant, df_charge, df_volume, df_cutoffs):

    df_occupant['move_in_date'] = pd.to_datetime(df_occupant['move_in_date'])
    df_occupant['move_out_date'] = pd.to_datetime(df_occupant['move_out_date'])
    df_charge['billing_period_end'] = \
        pd.to_datetime(df_charge['billing_period_end'])
    df_volume['meter_read_at'] = pd.to_datetime(df_volume['meter_read_at'])
    df_cutoffs['cutoff_at'] = pd.to_datetime(df_cutoffs['cutoff_at'])

    return [df_occupant, df_charge, df_volume, df_cutoffs]


# Create unique occupant_id strings
def make_occupant_ids(df_cutoffs, df_occupant, df_charge, df_volume):

    # Create compound key to identify occupants
    df_cutoffs['occupant_id'] = df_cutoffs['location_id'].astype(str) + \
        '-' + df_cutoffs['cis_occupant_id'].astype(int).map('{:03d}'.format)
    df_occupant['occupant_id'] = df_occupant['location_id'].astype(str) + \
        '-' + df_occupant['cis_occupant_id'].astype(int).map('{:03d}'.format)

    # For timeseries, for each location and occupant, label records at the
    # location that fall within occupant move-in and move-out date range
    # with the compound occupant_id - this will make later processing easier

    # Charge dataframe
    nRecs = len(df_charge['location_id'])
    df_charge['occupant_id'] = np.empty([nRecs], dtype=str)
    locations_charged = df_charge['location_id'].unique()
    nLocs = len(locations_charged)
    k = 0
    for loc in locations_charged:
        dfc = df_charge.loc[df_charge['location_id'] == loc].copy()
        bill_first = dfc.loc[dfc['location_id'] == loc, 'billing_period_end'] \
            .min()
        bill_last = dfc.loc[dfc['location_id'] == loc, 'billing_period_end'] \
            .max()
        if loc in df_occupant['location_id'].unique():
            dfo = df_occupant.loc[df_occupant['location_id'] == loc]
            cis_occupant_id_list = np.asarray(dfo['cis_occupant_id'].values)
            move_in_date_list = np.asarray(dfo['move_in_date'].values)
            move_out_date_list = np.asarray(dfo['move_out_date'].values)
        else:
            cis_occupant_id_list = np.asarray([0])
            move_in_date_list = np.asarray([bill_first])
            move_out_date_list = np.asarray([bill_last])
        nOccs = len(cis_occupant_id_list)
        for i in range(nOccs):
            cis_occupant_id = cis_occupant_id_list[i]
            move_in_date = move_in_date_list[i]
            move_out_date = move_out_date_list[i]
            occupant_id = '{:d}-{:03d}'.format(loc,cis_occupant_id)
            dfc.loc[dfc['billing_period_end'].between(move_in_date,
                                                      move_out_date),
                    'occupant_id'] = occupant_id
        if k == 0:
            df_charge_new = dfc.copy()
        else:
            df_charge_new = df_charge_new.append(dfc, ignore_index=True,
                                                 sort=False)
        k += 1
    df_charge = df_charge_new

    # Volume dataframe
    nRecs = len(df_volume['location_id'])
    df_volume['occupant_id'] = np.empty([nRecs], dtype=str)
    locations_read = df_volume['location_id'].unique()
    nLocs = len(locations_read)
    k = 0
    for loc in locations_read:
        dfv = df_volume.loc[df_volume['location_id'] == loc].copy()
        read_first = dfv.loc[dfv['location_id'] == loc, 'meter_read_at'].min()
        read_last = dfv.loc[dfv['location_id'] == loc, 'meter_read_at'].max()
        if loc in df_occupant['location_id'].unique():
            dfo = df_occupant.loc[df_occupant['location_id'] == loc]
            cis_occupant_id_list = np.asarray(dfo['cis_occupant_id'].values)
            move_in_date_list = np.asarray(dfo['move_in_date'].values)
            move_out_date_list = np.asarray(dfo['move_out_date'].values)
        else:
            cis_occupant_id_list = np.asarray([0])
            move_in_date_list = np.asarray([read_first])
            move_out_date_list = np.asarray([read_last])
        nOccs = len(cis_occupant_id_list)
        for i in range(nOccs):
            cis_occupant_id = cis_occupant_id_list[i]
            move_in_date = move_in_date_list[i]
            move_out_date = move_out_date_list[i]
            occupant_id = '{:d}-{:03d}'.format(loc,cis_occupant_id)
            dfv.loc[dfv['meter_read_at'].between(move_in_date, move_out_date),
                    'occupant_id'] = occupant_id
        if k == 0:
            df_volume_new = dfv.copy()
        else:
            df_volume_new = df_volume_new.append(dfv, ignore_index=True,
                                                 sort=False)
        k += 1
    df_volume = df_volume_new

    return [df_cutoffs, df_occupant, df_charge, df_volume]


# Geocode the latlons from the customer addresses
def address_to_latlon(config, df_location):

    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {
        'sensor': 'false',
        'key': config['MAPPING']['GOOGLE_MAPS_API_KEY'],
    }
    nLocs = len(df_location['location_id'].values)
    print('nLocs',nLocs)
    lat = np.empty([nLocs])
    lng = np.empty([nLocs])
    i = 0
    for loc in df_location['location_id'].values:
        address = np.asscalar(df_location.loc[df_location['location_id'] == loc,
                                              'meter_address'])
        address = address[:-5]
        params['address'] = address
        response = requests.get(url, params=params)
        resp_json_payload = response.json()
        lat[i] = resp_json_payload['results'][0]['geometry']['location']['lat']
        lng[i] = resp_json_payload['results'][0]['geometry']['location']['lng']
        i += 1
    df_location['latitude'] = lat
    df_location['longitude'] = lng

    return df_location


# Assemble a date from year, month, and day
def assemble_date(row):
    maxdays = [31,28,31,30,31,30,31,31,30,31,30,31]
    if row.name.year % 4 == 0:
        maxdays[1] = 29
    day = int(max(min(row['day'], maxdays[row.name.month - 1]), 1))
    date = dt.datetime(row.name.year, row.name.month, day)
        
    return date


# Utility function to divide a column by the 'nRecs' column
def divide_by_nrecs(row, col_name, dtype):
    if not np.isnan(row[col_name]) and row['nRecs'] > 0:
        if row['nRecs'] == 1:
            x = row[col_name]
        else:
            x = row[col_name] / row['nRecs']
    else:
        x = row[col_name]
    if dtype == 'int':
        x = int(x)

    return x


# Ensure that there are no missing months in the timeseries
def align_dates_monthly(df, date_col_name, col_names_data, align_mode='pad'):

    df['date_align'] = df[date_col_name].map(lambda x: dt.datetime(x.year,
                                                                   x.month, 1))
    df['year'] = pd.DatetimeIndex(df[date_col_name]).year
    df['month'] = pd.DatetimeIndex(df[date_col_name]).month
    df['day'] = pd.DatetimeIndex(df[date_col_name]).day
    df.set_index(df['date_align'], drop=False, inplace=True)
    df = df.loc[~df.index.duplicated(keep='first')].copy()
    if len(df) > 1:
        if align_mode == 'sum' or align_mode == 'mean':
            df['nRecs'] = np.ones([len(df)])
            df = df.resample('M').sum()
            for col in ['year','month','day']:
                df[col] = df.apply(divide_by_nrecs, args=(col, 'int'), axis=1)
            if align_mode == 'mean':
                for col in col_names_data:
                    df[col] = df[col].apply(divide_by_nrecs,
                                            args=(col, 'float'), axis=1)
            df.drop(columns=['nRecs'], inplace=True)
        else:
            df = df.resample('M').pad()
        df.loc[df['year'] != df.index.year, col_names_data] = np.nan
        df.loc[df['month'] != df.index.month, col_names_data] = np.nan
        df[date_col_name] = df.apply(assemble_date, axis=1)
    else:
        df[date_col_name] = df['date_align'].copy()
    if 'date_align' in df.columns:
        df.drop(columns=['date_align'], inplace=True)
    df['year'] = pd.DatetimeIndex(df[date_col_name]).year
    df['month'] = pd.DatetimeIndex(df[date_col_name]).month
    df['day'] = pd.DatetimeIndex(df[date_col_name]).day

    return df


# Wrapper for align_dates()
def wrap_align_dates(df_charge, df_volume):

    # Charge table
    print('aligning dates: charge table')
    col_names_data_charge = tc.tables_and_columns['charge'].copy()
    for col in ['charge_id','cis_bill_id','billing_period_end']:
        col_names_data_charge.remove(col)
    i = 0
    for occ in df_charge['occupant_id'].unique():
        df = df_charge.loc[df_charge['occupant_id'] == occ] \
            .sort_values(by='billing_period_end').copy()
        df_new = align_dates_monthly(df, 'billing_period_end',
                                     col_names_data_charge,
                                     align_mode='pad')
        if i == 0:
            df_charge_align = df_new.copy()
        else:
            df_charge_align = df_charge_align.append(df_new, ignore_index=True,
                                                     sort=False)
        i += 1

    # Volume table
    print('aligning dates: volume table')
    col_names_data_volume = tc.tables_and_columns['volume'].copy()
    for col in ['meter_id','charge_id','meter_read_at']:
        col_names_data_volume.remove(col)
    i = 0
    for occ in df_volume['occupant_id'].unique():
        df = df_volume.loc[df_volume['occupant_id'] == occ] \
            .sort_values(by='meter_read_at').copy()
        df_new = align_dates_monthly(df, 'meter_read_at',
                                     col_names_data_volume,
                                     align_mode='pad')
        if i == 0:
            df_volume_align = df_new.copy()
        else:
            df_volume_align = df_volume_align.append(df_new, ignore_index=True,
                                                     sort=False)
        i += 1

    return [df_charge, df_volume]


# Null out bad values
def null_out_bad(df_volume):

    df_volume.loc[df_volume['volume_kgals'] < 0, 'volume_kgals'] = np.nan
    df_volume.loc[df_volume['location_id'] <= 0, 'location_id'] = np.nan
    df_volume.loc[df_volume['meter_id'] <= 0, 'meter_id'] = np.nan
    i = 0
    for occ in df_volume['occupant_id'].unique():
        df = df_volume.loc[df_volume['occupant_id'] == occ].sort_values(by='meter_read_at').copy()
        mean = df['volume_kgals'].mean(skipna=True)
        std = df['volume_kgals'].std(skipna=True)
        if std > 0:
            df['ztrans'] = (df['volume_kgals'] - mean) / std
            df.loc[df['ztrans'] > 5, 'volume_kgals'] = np.nan
            df.drop(columns=['ztrans'], inplace=True)
        if i == 0:
            df_volume_clean = df.copy()
        else:
            df_volume_clean = df_volume_clean.append(df, ignore_index=True, sort=False)
        i += 1

    return df_volume_clean


# Fill gaps
def fill_gaps(df_charge, df_volume):
    df_charge_fill = df_charge.copy()
    for col in ['location_id']:
        df_charge_fill[col].interpolate(inplace=True)
    for col in ['total_charge','late_charge']:
        df_charge_fill.loc[df_charge_fill[col].isna(), col] = 0
    df_volume_fill = df_volume.copy()
    for col in ['location_id', 'meter_id', 'volume_kgals']:
        df_volume_fill[col].interpolate(inplace=True)
    for col in ['volume_kgals']:
        df_volume_fill.loc[df_volume_fill[col].isna(), col] = 0

    return [df_charge_fill, df_volume_fill]


# Create binary late column
def create_late(df_charge):
    df_charge['late'] = df_charge['late_charge']
    df_charge.loc[df_charge['late'].isna(), 'late'] = 0
    df_charge.loc[df_charge['late'] > 0, 'late'] = 1
    df_charge.loc[df_charge['late'] <= 0, 'late'] = 0

    return df_charge


# Prepare data for use
def prep_data(config, df_meter, df_location, df_occupant,
              df_volume, df_charge, df_cutoffs):

    # Integrity check on occupant table
    print('integrity check on occupant table')
    df_occupant = occupant_integrity(df_occupant)

    # Ensure date consistency in other tables
    print('converting date columns to datetime objects')
    [df_occupant, df_charge,
     df_volume, df_cutoffs] = date_consistency(df_occupant, df_charge,
                                               df_volume, df_cutoffs)

    # Create unique occupant_id strings
    print('creating occupant_id')
    [df_cutoffs, df_occupant,
     df_charge, df_volume] = make_occupant_ids(df_cutoffs, df_occupant,
                                               df_charge, df_volume)

    # Geocode the lat/lons of the customer addresses
    print('geocoding the lat/lons of customer addresses')
    df_location = address_to_latlon(config, df_location)

    # Ensure there are no missing months in the time series
    print('ensuring that there is a record for every month')
    [df_charge, df_volume] = wrap_align_dates(df_charge, df_volume)

    # Null out bad values
    print('nulling out bad values')
    df_volume = null_out_bad(df_volume)

    # Fill gaps
    print('filling gaps')
    [df_charge, df_volume] = fill_gaps(df_charge, df_volume)

    # Create binary late column
    print('creating late column')
    df_charge = create_late(df_charge)

    return [df_meter, df_location, df_occupant,
            df_volume, df_charge, df_cutoffs]


