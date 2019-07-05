import psycopg2
import pandas as pd
import numpy as np

owners = {
    'clayton_county': 'broostaei',
    'clayton_county_unfiltered_raw': 'mokun',
}

config = {
    'broostaei': {
        'dbname':'aquifer_development_broostaei',
        'user':'aquifer_development_tbohn',
        'pwd':'jbSo5WL9J1XwbvUvtECduA==',
        'host':'aquifer-e.caxvpfnlw4wu.us-west-2.redshift.amazonaws.com',
        'port':'5439'
    },
    'mokun': {
        'dbname':'aquifer_development_mokun',
        'user':'aquifer_development_tbohn',
        'pwd':'jbSo5WL9J1XwbvUvtECduA==',
        'host':'aquifer-e.caxvpfnlw4wu.us-west-2.redshift.amazonaws.com',
        'port':'5439'
    }
}

table_names_and_columns = {
    'broostaei': {
        'meter': [
            'meter_id',
            'cis_cust_id',
            'cis_location_id',
            'location_id',
            'data_set_id',
            'cust_type',
            'cust_type_code',
            'meter_type',
            'meter_size',
            'meter_size_units',
            'meter_size_lowflow',
            'num_units',
        ],
        'meter_location': [
            'location_id',
            'cis_location_id',
            'data_set_id',
#            'latitude',
#            'longitude',
            'meter_address',
            'municipality',
            'subdivision',
            'route',
#            'experimental_group',
#            'shared_premise_identifier',
            'aggregate_cust_type_code',
        ],
        'occupant': [
            'id',
            'location_id',
            'cis_occupant_id',
            'data_set_id',
            'move_in_date',
            'move_out_date',
        ],
        'cutoffs': [
            'id',
            'location_id',
            'cis_occupant_id',
            'data_set_id',
            'cutoff_at',
        ],
        'volume': [
            'meter_id',
            'charge_id',
            'data_set_id',
            'meter_read_at',
#            'fiscal_year',
            'volume_kgals',
            'volume_unit',
            'unbilled',
            'days_diff',
        ],
        'rate_table_vol': [
            'rate_code',
            'min_volume_kgals',
            'price_per_kgal',
            'max_volume_kgals',
            'rate_started_at',
            'rate_ended_at',
        ],
        'charge': [
            'charge_id',
            'cis_bill_id',
            'location_id',
            'data_set_id',
            'billing_period_end',
#            'fiscal_year',
            'days_diff',
            'base_charge',
            'vol_charge',
            'total_charge',
            'late_charge',
            'tampering_charge',
            'special_rate_account',
            'zero_bill_account',
            'other_charge',
            'rate_code',
            'payment_arrangement',
            'prorated',
            'estimated',
        ],
    },
    'mokun': {
        'meter',
        'volume',
        'rate_table_vol',
        'charge',
    },
}

def create_conn(*args,**kwargs):
    config = kwargs['config']
    try:
        con=psycopg2.connect(dbname=config['dbname'], host=config['host'], 
                             port=config['port'], user=config['user'], 
                             password=config['pwd'])
        return con
    except Exception as err:
        print(err)

def fetch_data(schema):

    owner = owners[schema]
    conn = create_conn(config=config[owner])
    if schema == 'clayton_county':
        data_set_id = 28

        # Construct meter dataframe, keyed by meter_id
        tables_full = [schema + '.meter']
        columns = tables_and_columns[owner['meter']]
        columnliststr = ','.join(columns)
        query = ("SELECT " + columnliststr +
                 " FROM " + tables_full[0] +
                 " WHERE data_set_id=" + str(data_set_id))
        df_meter = pd.read_sql(query, con=conn)

        # Construct location dataframe, keyed by location_id
        tables_full = [schema + '.meter_location']
        columns = tables_and_columns[owner['meter']]
        columns = [
            'location_id',
#            'data_set_id',
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
        columns = tables_and_columns[owner['meter']]
        columns = [
#            'id',
            'location_id',
            'cis_occupant_id',
#            'data_set_id',
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
        columns = tables_and_columns[owner['meter']]
        columns = [
            tables_full[0] + '.' + 'location_id',
            tables_full[1] + '.' + 'meter_id',
            'charge_id',
#            tables_full[0] + '.' + 'data_set_id',
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
        columns = tables_and_columns[owner['meter_location']]
        columns['location_id'] = tables_full[0] + . + columns['location_id']
        columns['data_set_id'] = tables_full[0] + . + columns['data_set_id']
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
        columns = tables_and_columns[owner['cutoffs']]
        columns['location_id'] = tables_full[0] + . + columns['location_id']
        columnliststr = ','.join(columns)
        query = ("SELECT " + columnliststr +
                 " FROM " + tables_full[0] +
                 " INNER JOIN " + tables_full[1] +
                 " ON " + tables_full[0] + ".location_id=" +
                 tables_full[1] + ".location_id" +
                 " WHERE " + tables_full[0] + ".data_set_id=" +
                 str(data_set_id))
        df_cutoffs = pd.read_sql(query, con=conn)

#        query = ("SELECT * FROM clayton_county.flagged_meters")
#        df_flagged_meters = pd.read_sql(query, con=conn)

    return [df_meter, df_location, df_occupant, df_volume, df_charge,
#            df_cutoffs, df_flagged_meters]
            df_cutoffs]


