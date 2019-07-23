# -*- coding: utf-8 -*-
import psycopg2
import pandas as pd
import numpy as np
from . import table_columns as tc

def create_conn(*args,**kwargs):
    config = kwargs['config']
    try:
        con=psycopg2.connect(dbname=config['DATABASE']['DBNAME'],
                             host=config['DATABASE']['HOST'], 
                             port=config['DATABASE']['PORT'],
                             user=config['DATABASE']['USER'], 
                             password=config['DATABASE']['PWD'])
        return con
    except Exception as err:
        print(err)

def fetch_data(config):

    conn = create_conn(config=config)
    schema = config['DATABASE']['SCHEMA']
    data_set_id = config['DATABASE']['DATA_SET_ID']

    # Construct meter dataframe, keyed by meter_id
    tables_full = [schema + '.meter']
    columns = tc.tables_and_columns['meter']
    columnliststr = ','.join(columns)
    query = ("SELECT " + columnliststr +
             " FROM " + tables_full[0] +
             " WHERE data_set_id=" + str(data_set_id))
    df_meter = pd.read_sql(query, con=conn)

    # Construct location dataframe, keyed by location_id
    tables_full = [schema + '.meter_location']
    columns = tc.tables_and_columns['meter']
    columns = [
        'location_id',
#        'data_set_id',
        'meter_address',
        'municipality',
        'subdivision',
        'route',
        'aggregate_cust_type_code',
    ]
    columnliststr = ','.join(columns)
    query = ("SELECT " + columnliststr +
             " FROM " + tables_full[0] +
             " WHERE data_set_id=" + str(data_set_id))
    df_location = pd.read_sql(query, con=conn)

    # Construct occupant dataframe, keyed by id
    tables_full = [schema + '.occupant']
    columns = tc.tables_and_columns['meter']
    columns = [
#        'id',
        'location_id',
        'cis_occupant_id',
#        'data_set_id',
        'move_in_date',
        'move_out_date',
    ]
    columnliststr = ','.join(columns)
    query = ("SELECT " + columnliststr +
             " FROM " + tables_full[0] +
             " WHERE data_set_id=" + str(data_set_id))
    df_occupant = pd.read_sql(query, con=conn)

    # Construct volume dataframe, keyed by location_id
    tables = ['meter_location','meter','volume']
    tables_full = []
    for name in tables:
        tables_full.append(schema + '.' + name)
        columns = tc.tables_and_columns['meter']
        columns = [
        tables_full[0] + '.' + 'location_id',
        tables_full[1] + '.' + 'meter_id',
        'charge_id',
#        tables_full[0] + '.' + 'data_set_id',
        'meter_read_at',
        'volume_kgals',
        'volume_unit',
        'unbilled',
    ]
    columnliststr = ','.join(columns)
    query = ("SELECT " + columnliststr +
             " FROM " + tables_full[0] +
             " INNER JOIN " + tables_full[1] +
             " ON " + tables_full[0] + ".location_id=" +
             tables_full[1] + ".location_id" +
             " INNER JOIN " + tables_full[2] +
             " ON " + tables_full[1] + ".meter_id=" +
             tables_full[2] + ".meter_id" +
             " WHERE " + tables_full[0] + ".data_set_id=" +
             str(data_set_id))
    df_volume = pd.read_sql(query, con=conn)

    # Construct charge dataframe, keyed by location_id
    tables = ['meter_location','charge']
    tables_full = []
    for name in tables:
        tables_full.append(schema + '.' + name)
    columns = tc.tables_and_columns['meter_location']
    columns['location_id'] = tables_full[0] + '.' + columns['location_id']
    columns['data_set_id'] = tables_full[0] + '.' + columns['data_set_id']
    columnliststr = ','.join(columns)
    query = ("SELECT " + columnliststr +
             " FROM " + tables_full[0] +
             " INNER JOIN " + tables_full[1] +
             " ON " + tables_full[0] + ".location_id=" +
             tables_full[1] + ".location_id" +
             " WHERE " + tables_full[0] + ".data_set_id=" +
             str(data_set_id))
    df_charge = pd.read_sql(query, con=conn)

    # Construct cutoffs dataframe, keyed by location_id and id
    tables = ['cutoffs','meter_location']
    tables_full = []
    for name in tables:
        tables_full.append(schema + '.' + name)
    columns = tc.tables_and_columns['cutoffs']
    columns['location_id'] = tables_full[0] + '.' + columns['location_id']
    columnliststr = ','.join(columns)
    query = ("SELECT " + columnliststr +
             " FROM " + tables_full[0] +
             " INNER JOIN " + tables_full[1] +
             " ON " + tables_full[0] + ".location_id=" +
             tables_full[1] + ".location_id" +
             " WHERE " + tables_full[0] + ".data_set_id=" +
             str(data_set_id))
    df_cutoffs = pd.read_sql(query, con=conn)

#    query = ("SELECT * FROM clayton_county.flagged_meters")
#    df_flagged_meters = pd.read_sql(query, con=conn)

    return [df_meter, df_location, df_occupant, df_volume, df_charge,
#        df_cutoffs, df_flagged_meters]
        df_cutoffs]


