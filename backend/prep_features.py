# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import skew
import math
from . import prep_data as prda


# Z-transform of timeseries
def ztrans(x):
    
    N = len(x)
    ztrans = np.zeros([N], dtype=np.float)
    if N > 2:
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std > 0:
            ztrans = (x - mean) / std

    return ztrans


# Z-transform of timeseries, separately for each of 12 months
def ztrans_bymon(x, default=0.0):
    
    N = len(x)
    ztrans = np.full([N], default, dtype=np.float)
    if N > 24:
        if N % 12 > 0:
            ntmp = (int(N / 12) + 1) * 12
        else:
            ntmp = N
        nYears = int(ntmp / 12)
        tmp = np.empty([ntmp])
        tmp[:N] = x.values
        mmean = np.nanmean(tmp.reshape(nYears, 12))
        mstd = np.nanstd(tmp.reshape(nYears, 12))
        anom = (tmp.reshape(nYears, 12) - mmean) / mstd
        anom[np.isnan(anom)] = default
        anom[np.isinf(anom)] = default
        ztrans = anom.reshape(nYears * 12)[:N]

    return ztrans


def clip_time_series_by_cutoffs(df, date_colname, date_first, date_last,
                                cutoff_dates):

    nDates = len(cutoff_dates)
    df_list = []
    lengths = []

    # cutoff_trinary = 0 if no cutoffs occur after or within this segment
    #                  1 if a cutoff occurs after this segment but not within
    #                  2 if a cutoff occurs on the final time step of the
    #                      segment
    cutoff_trinary = []

    # Number of prior cutoffs for each segment
    n_cut_prior = []

    date0 = date_first
    # allow for cutoff to occur 1 month after final billing date
    date_last = date_last + pd.DateOffset(months=1)

    # Get all the segments within the time series that are followed by a cutoff
    c = 0
    while c < nDates and cutoff_dates[c] <= date_last:
        if cutoff_dates[c] > date_first and cutoff_dates[c] <= date_last:
            # The cutoff occurred within the given time series
            date1 = cutoff_dates[c]
            df_tmp = df.loc[df[date_colname].between(date0, date1)]
            date0 = date1 + pd.DateOffset(months=1)
            df_list.append(df_tmp)
            lengths.append(len(df_tmp))
            cutoff_trinary.append(2)
            n_cut_prior.append(c)
        c += 1
    # Get any final segment that doesn't end in a cutoff
    if c < nDates and date0 < date_last:
        date1 = date_last
        df_tmp = df.loc[df[date_colname].between(date0, date1)]
        df_list.append(df_tmp)
        lengths.append(len(df_tmp))
        if cutoff_dates[c] > date_last:
            cutoff_trinary.append(1)
        else:
            cutoff_trinary.append(0)
        n_cut_prior.append(c)

    return df_list, lengths, cutoff_trinary, n_cut_prior


# Create table of predictors and predictands
def create_feature_table(df_occupant, df_location, df_meter, df_cutoffs,
                         df_charge, df_volume, today, mode='train',
                         nSamples=6):

    occlist = df_occupant['occupant_id'].unique()
    i = 0
    k = 0
    kcut = 0
    knocut = 0

    for occ in occlist:

        # Determine whether this customer is within our timeframe
        dfo = df_occupant.loc[df_occupant['occupant_id'] == occ].copy()
        if len(dfo) > 0:
            move_in_date = pd.to_datetime(dfo['move_in_date'].values[0])
            move_out_date = pd.to_datetime(dfo['move_out_date'].values[0])
            if mode == 'train' and move_in_date > today:
                continue
            elif mode == 'predict' and move_out_date < today:
                continue
        
        # Get customer metadata
        if occ in df_cutoffs['occupant_id'].unique():
            cutoff = 1
        else:
            cutoff = 0
        try:
            location_id = df_occupant.loc[df_occupant['occupant_id'] == occ,
                                          'location_id'].values[0]
        except:
            location_id = -1
        try:
            latitude = df_location.loc[df_location['location_id'] == \
                location_id, 'latitude'].values[0]
        except:
            latitude = 0.0
        try:
            longitude = df_location.loc[df_location['location_id'] == \
                location_id, 'longitude'].values[0]
        except:
            longitude = 0.0
        try:
            meter_address = df_location.loc[df_location['location_id'] == \
                location_id, 'meter_address'].values[0]
        except:
            meter_address = 'unknown'
        try:
            municipality = df_location.loc[df_location['location_id'] == \
                location_id, 'municipality'].values[0]
        except:
            municipality = -1
        try:
            meter_id = df_meter.loc[df_meter['location_id'] == location_id,
                                    'meter_id'].values[0]
        except:
            meter_id = -1
        try:
            cust_type = df_meter.loc[df_meter['meter_id'] == meter_id,
                                     'cust_type'].values[0]
        except:
            cust_type = 'unknown'
        try:
            cust_type_code = df_meter.loc[df_meter['meter_id'] == meter_id,
                                          'cust_type_code'].values[0]
        except:
            cust_type_code = -1
        try:
            meter_size = df_meter.loc[df_meter['meter_id'] == meter_id,
                                      'meter_size'].values[0]
        except:
            meter_size = -1

        # Compute time series stats, for those customers that have them
        
        # First, get cutoff dates, if any
        if cutoff:
            cutoff_dates = pd.to_datetime(np.asarray(df_cutoffs \
                .loc[df_cutoffs['occupant_id'] == occ, 'cutoff_at'] \
                .sort_values()))
            cutoff_dates = cutoff_dates[cutoff_dates <= today]
            nCut = len(cutoff_dates)

        # Next, identify boundaries of time series of billing and usage;
        # grab the time series, and do some whole-timeseries normalization
        # (standardized anomalies)
        if occ in df_charge['occupant_id'].unique():
            dfc = df_charge.loc[df_charge['occupant_id'] == occ].copy()
            bill_first = pd.to_datetime(dfc['billing_period_end'].min())
            bill_last = pd.to_datetime(dfc['billing_period_end'].max())
            bill_last = min(bill_last, today)
            dfc = dfc.loc[dfc['billing_period_end'].between(bill_first,
                                                            bill_last)]
            nCharge = len(dfc)
        else:
            nCharge = 0
        if occ in df_volume['occupant_id'].unique():
            dfv = df_volume.loc[df_volume['occupant_id'] == occ].copy()
            read_first = dfv['meter_read_at'].min()
            read_last = dfv['meter_read_at'].max()
            read_last = min(read_last, today)
            dfv = dfv.loc[dfv['meter_read_at'].between(read_first, read_last)]
            dfv['vol_log'] = dfv['volume_kgals']
            dfv.loc[dfv['vol_log'] == 0, 'vol_log'] = 0.1
            dfv['vol_log'] = dfv['vol_log'].apply(math.log)
            nVolume = len(dfv)
            for col in ['volume_kgals','vol_log']:
                # Anomalies without accounting for seasonal cycle
                col_anom = col + '_anom'
                dfv[col_anom] = ztrans(dfv[col])
                # Anomalies relative to seasonal cycle
                col_manom = col + '_manom'
                dfv[col_manom] = ztrans_bymon(dfv[col])
        else:
            nVolume = 0

        # Skip occupants with no usable records
        if nCharge == 0 or nVolume == 0:
            continue

        # Clip sample windows out of time series

        # segments = divide time series into "independent" portions between
        # cutoffs (if no cutoff, entire time series is a single segment)
        # windows = divide each segment into consecutive windows, starting
        # from end and moving backwards

        # dfc_list and dfv_list are the charge and volume timeseries for each
        # segment
        # dfc_lens and dfv_lens are the lengths of those segments
        # dfc_ and dfv_cut_tri are trinary cutoff labels of those segments
        #     where 0 = no subsequent cutoff
        #           1 = cutoff subsequent but not immediate
        #           2 = cutoff immediately subsequent
        # dfc_ and dfv_nprior are numbers of prior cutoffs for each segment
        dfc_list = []
        dfv_list = []
        dfc_lens = []
        dfv_lens = []
        dfc_cut_tri = []
        dfv_cut_tri = []
        dfc_nprior = []
        dfv_nprior = []
        # nWindows_list = number of windows in each segment
        nWindows_list = []

        # label_list, t0c_list, and t1c_list are the trinary labels and start
        # times for each window in each segment
        label_list = []
        t0c_list = []
        t0v_list = []

        # Different numbers and positions of windows for 'train' and 'pred'
        # periods
        if mode == 'train':

            # For training period cutoff time series, clip time series into
            # segments between cutoffs (if any)
            if cutoff:
                dfc_list, dfc_lens, dfc_cut_tri, dfc_nprior = \
                    clip_time_series_by_cutoffs(dfc, 'billing_period_end',
                                                bill_first, bill_last,
                                                cutoff_dates)
                dfv_list, dfv_lens, dfv_cut_tri, dfv_nprior = \
                    clip_time_series_by_cutoffs(dfv, 'meter_read_at',
                                                read_first, read_last,
                                                cutoff_dates)
                nSegc = len(dfc_lens)
                nSegv = len(dfv_lens)
                nSegments = min(nSegc, nSegv)
                nCP_array = np.empty([nSegments])
                nCP_array[:] = dfc_nprior[:nSegments]
            # For training period nocut time series, there is just 1 segment
            else:
                dfc_list.append(dfc.copy())
                dfv_list.append(dfv.copy())
                dfc_lens.append(len(dfc.index))
                dfv_lens.append(len(dfv.index))
                nSegments = 1
                nCP_array = np.arange(0, 1).astype(int)
            CP_array = np.where(nCP_array > 0, 1, 0).astype(int)

            # Clip out consecutive sample windows from the end of the segment
            # backwards as far as possible
            for g in range(nSegments):
                t0c_list_tmp = []
                t0v_list_tmp = []
                label_list_tmp = []
                nWindows = 0
                # Assign value of the trinary label for final window
                # (which we start with)
                if cutoff:
                    # Final window of segment, whose value will be the
                    # trinary label computed in clip_time_series...()
                    label_tri = dfc_cut_tri[g]
                else:
                    # Final window in a non-cutoff segment
                    # (all windows get same label)
                    label_tri = 0
                t0c = dfc_lens[g] - nSamples
                t0v = dfv_lens[g] - nSamples
                while t0c >= 0 and t0v >= 0:
                    t0c_list_tmp.append(t0c)
                    t0v_list_tmp.append(t0v)
                    label_list_tmp.append(label_tri)
                    t0c -= nSamples
                    t0v -= nSamples
                    if cutoff == 1:
                        # Prior windows in a cutoff segment
                        if dfc_cut_tri[g] >= 1:
                            label_tri = 1
                        else:
                            label_tri = 0
                    nWindows += 1
                nWindows_list.append(nWindows)
                t0c_list.append(t0c_list_tmp)
                t0v_list.append(t0v_list_tmp)
                label_list.append(label_list_tmp)

        # For predictions from current conditions, clip out sample window
        # from nSamples months prior to today
        else:

            nSegments = 1
            dfc_list.append(dfc.copy())
            dfv_list.append(dfv.copy())
            dfc_lens.append(len(dfc.index))
            dfv_lens.append(len(dfv.index))
            for g in range(nSegments):
                t0c_list_tmp = []
                t0v_list_tmp = []
                label_list_tmp = []
                nWindows = 1
                t0c = dfc_lens[g] - nSamples
                t0v = dfv_lens[g] - nSamples
                nCutPost = len(df_cutoffs \
                    .loc[((df_cutoffs['occupant_id'] == occ) &
                          (df_cutoffs['cutoff_at'] > today)), 'cutoff_at'] \
                    .index)
                if nCutPost == 0:
                    label_tri = 0
                else:
                    date_first_cut = df_cutoffs \
                        .loc[((df_cutoffs['occupant_id'] == occ) &
                              (df_cutoffs['cutoff_at'] > today)), 'cutoff_at'] \
                        .values()[0]
                    date_first_cut = pd.to_datetime(date_first_cut)
                    date_diff = int(np.round(((date_first_cut - today) / \
                                              np.timedelta64(1, 'M')))
                    if date_diff <= nSamples:
                        label_tri = 2
                    else:
                        label_tri = 1
                t0c_list_tmp.append(t0c)
                t0v_list_tmp.append(t0v)
                label_list_tmp.append(label_tri)
                nWindows_list.append(nWindows)
                t0c_list.append(t0c_list_tmp)
                t0v_list.append(t0v_list_tmp)
                label_list.append(label_list_tmp)

            nCP_array = np.empty([1])
            if cutoff:
                nCutPrior = len(df_cutoffs \
                    .loc[((df_cutoffs['occupant_id'] == occ) &
                          (df_cutoffs['cutoff_at'] <= today)), 'cutoff_at'] \
                    .index)
                nCP_array[0] = nCutPrior
            else:
                nCP_array[0] = 0
            CP_array = np.where(nCP_array > 0, 1, 0).astype(int)

        # Compute statistics over all windows in all segments
        for g in range(nSegments):

            nWindows = nWindows_list[g]
            if nWindows == 0:
                continue
            dfc_seg = dfc_list[g]
            dfv_seg = dfv_list[g]
            nCutPrior = nCP_array[g]
            cut_prior = CP_array[g]

            for w in range(nWindows):
            
                t0c = t0c_list[g][w]
                t1c = t0c + nSamples
                t0v = t0v_list[g][w]
                t1v = t0v + nSamples
                label_tri = label_list[g][w]

                # Default values of window statistics
                f_late = np.nan
                mean_charge_tot = np.nan
                mean_charge_late = np.nan
                mean_vol = np.nan
                f_zero_vol = np.nan
                skew_vol = np.nan
                f_anom3_vol = np.nan
                f_anom4_vol = np.nan
                f_anom5_vol = np.nan
                max_anom_vol = np.nan
                f_anom3_vol_log = np.nan
                f_anom4_vol_log = np.nan
                f_anom5_vol_log = np.nan
                max_anom_vol_log = np.nan
                f_manom3_vol = np.nan
                f_manom4_vol = np.nan
                f_manom5_vol = np.nan
                max_manom_vol = np.nan
                f_manom3_vol_log = np.nan
                f_manom4_vol_log = np.nan
                f_manom5_vol_log = np.nan
                max_manom_vol_log = np.nan

                # Compute statistics of sample window
                # Stats of charge
                if t0c >= 0 and t1c > t0c:
                    df_tmp = dfc_seg.iloc[t0c:t1c].copy()
                    f_late = df_tmp['late'].mean(skipna=True)
                    mean_charge_tot = df_tmp['total_charge'].mean(skipna=True)
                    mean_charge_late = df_tmp['late_charge'].mean(skipna=True)
                else:
                    print('WARNING: i', i, 'g', g, 'w', w, 'nCharge', nCharge,
                          't0c', t0c, 't1c', t1c)
                # Stats of volume
                if t0v >= 0 and t1v > t0v:
                    df_tmp = dfv_seg.iloc[t0v:t1v].copy()
                    mean_vol = df_tmp['volume_kgals'].mean(skipna=True)
                    f_zero_vol = \
                        df_tmp.loc[df_tmp['volume_kgals'] == 0,
                                   'volume_kgals'].count() / (t1v - t0v)
                    skew_vol = abs(skew(np.asarray(df_tmp['volume_kgals'])))
                    nAnom = df_tmp['volume_kgals_anom'].count()
                    nMAnom = df_tmp['volume_kgals_manom'].count()
                    if nAnom < nSamples or nMAnom < nSamples:
                        print('WARNING: i', i, 'g', g, 'w', w,
                              'nSamples', nSamples, 'nAnom', nAnom,
                              'nMAnom', nMAnom)
                    if nAnom > 0:
                        f_anom3_vol = \
                            df_tmp.loc[abs(df_tmp['volume_kgals_anom']) > 3,
                                       'volume_kgals_anom'].count() / nAnom
                        f_anom4_vol = \
                            df_tmp.loc[abs(df_tmp['volume_kgals_anom']) > 4,
                                       'volume_kgals_anom'].count() / nAnom
                        f_anom5_vol = \
                            df_tmp.loc[abs(df_tmp['volume_kgals_anom']) > 5,
                                       'volume_kgals_anom'].count() / nAnom
                        max_anom_vol = \
                            np.max(np.abs(df_tmp['volume_kgals_anom']))
                        f_anom3_vol_log = \
                            df_tmp.loc[abs(df_tmp['vol_log_anom']) > 3,
                                       'vol_log_anom'].count() / nAnom
                        f_anom4_vol_log = \
                            df_tmp.loc[abs(df_tmp['vol_log_anom']) > 4,
                                       'vol_log_anom'].count() / nAnom
                        f_anom5_vol_log = \
                            df_tmp.loc[abs(df_tmp['vol_log_anom']) > 5,
                                       'vol_log_anom'].count() / nAnom
                        max_anom_vol_log = \
                            np.max(np.abs(df_tmp['vol_log_anom']))

                    if nMAnom > 0:
                        f_manom3_vol = \
                            df_tmp.loc[abs(df_tmp['volume_kgals_manom']) > 3,
                                       'volume_kgals_manom'].count() / nMAnom
                        f_manom4_vol = \
                            df_tmp.loc[abs(df_tmp['volume_kgals_manom']) > 4,
                                       'volume_kgals_manom'].count() / nMAnom
                        f_manom5_vol = \
                            df_tmp.loc[abs(df_tmp['volume_kgals_manom']) > 5,
                                       'volume_kgals_manom'].count() / nMAnom
                        max_manom_vol = \
                            np.max(np.abs(df_tmp['volume_kgals_manom']))
                        f_manom3_vol_log = \
                            df_tmp.loc[abs(df_tmp['vol_log_manom']) > 3,
                                       'vol_log_manom'].count() / nMAnom
                        f_manom4_vol_log = \
                            df_tmp.loc[abs(df_tmp['vol_log_manom']) > 4,
                                       'vol_log_manom'].count() / nMAnom
                        f_manom5_vol_log = \
                            df_tmp.loc[abs(df_tmp['vol_log_manom']) > 5,
                                       'vol_log_manom'].count() / nMAnom
                        max_manom_vol_log = \
                            np.max(np.abs(df_tmp['vol_log_manom']))
                else:
                    print('WARNING: i', i, 'g', g, 'w', w, 'nVolume', nVolume,
                          't0v', t0v, 't1v', t1v)

                # Define binary cutoff label
                if label_tri < 2: 
                    cutoff_strict = 0
                else:
                    cutoff_strict = 1

                # Save feature values in output dataframe
                df = pd.DataFrame(
                    data={'occupant_id': occ,
                          'location_id': location_id,
                          'latitude': latitude,
                          'longitude': longitude,
                          'meter_address': meter_address,
                          'municipality': municipality,
                          'cust_type': cust_type,
                          'cust_type_code': cust_type_code,
                          'meter_size': meter_size,
                          'segment': g,
                          'window': w,
                          'label': label_tri,
                          'cutoff_strict': cutoff_strict,
                          'cutoff': cutoff,
                          'nCutPrior': nCutPrior,
                          'cut_prior': cut_prior,
                          'f_late': f_late,
                          'mean_charge_tot': mean_charge_tot,
                          'mean_charge_late': mean_charge_late,
                          'skew_vol': skew_vol,
                          'f_zero_vol': f_zero_vol,
                          'f_anom3_vol': f_anom3_vol,
                          'f_anom4_vol': f_anom4_vol,
                          'f_anom5_vol': f_anom5_vol,
                          'max_anom_vol': max_anom_vol,
                          'f_anom3_vol_log': f_anom3_vol_log,
                          'f_anom4_vol_log': f_anom4_vol_log,
                          'f_anom5_vol_log': f_anom5_vol_log,
                          'max_anom_vol_log': max_anom_vol_log,
                          'f_manom3_vol': f_manom3_vol,
                          'f_manom4_vol': f_manom4_vol,
                          'f_manom5_vol': f_manom5_vol,
                          'max_manom_vol': max_manom_vol,
                          'f_manom3_vol_log': f_manom3_vol_log,
                          'f_manom4_vol_log': f_manom4_vol_log,
                          'f_manom5_vol_log': f_manom5_vol_log,
                          'max_manom_vol_log': max_manom_vol_log,
                          'mean_vol': mean_vol,
                         },
                   index=[k])

                if k == 0:
                    feature_table = df.copy()
                else:
                    feature_table = feature_table.append(df, ignore_index=True,
                                                         sort=False)
                k += 1
                if cutoff:
                    kcut += 1
                else:
                    knocut += 1
        i += 1

    return feature_table
        
        
def create_and_prep_feature_table(df_occupant, df_location, df_meter,
                                  df_cutoffs, df_charge, df_volume,
                                  ref_date, mode, nSamples, strict=False):

    today = pd.to_datetime(ref_date)

    # Create feature table
    feature_table = create_feature_table(df_occupant, df_location,
                                         df_meter, df_cutoffs,
                                         df_charge, df_volume,
                                         today, mode, nSamples)

    collist = ['f_late', 'mean_charge_tot', 'mean_charge_late',
               'mean_vol', 'skew_vol',
               'f_anom3_vol','f_anom4_vol',
               'f_anom5_vol','max_anom_vol',
               'f_anom3_vol_log','f_anom4_vol_log',
               'f_anom5_vol_log','max_anom_vol_log',
               'f_manom3_vol','f_manom4_vol',
               'f_manom5_vol','max_manom_vol',
               'f_manom3_vol_log','f_manom4_vol_log',
               'f_manom5_vol_log','max_manom_vol_log',
              ]

    if strict:
        label = 'cutoff_strict'
    else:
        label = 'cutoff'

    # Handle customers that are missing stats
    if mode == 'train':
        # Populate stats of cutoff customers who have no stats
        # with random samples from those with stats
        for col in collist:
            df = feature_table.loc[feature_table[col].notna(),
                                   ['occupant_id', label]].copy()
            occs_cut_stats = df.loc[df[label] == 1, 'occupant_id'].values
            nOccsCutStats = len(occs_cut_stats)
            df = feature_table.loc[feature_table[col].isna(),
                                   ['occupant_id', label]].copy()
            occs_cut_no_stats = df.loc[df[label] == 1, 'occupant_id'].values
            nOccsCutNoStats = len(occs_cut_no_stats)
            if nOccsCutNoStats > 0:
                print('WARNING: col:', col,
                      'number of cutoffs with stats:', nOccsCutStats,
                      'number of cutoffs without stats:', nOccsCutNoStats,
                      'imputing missing stats with random samples from those',
                      'with stats')
                print(df)
                df = feature_table.loc[feature_table[col].notna()].copy()
                df_cut_stats = df.loc[df[label] == 1].copy()
                for occ in occs_cut_no_stats:
                    i = int(np.random.uniform(0, nOccsCutStats, 1))
                    feature_table.loc[feature_table['occupant_id'] == occ,
                                      col] = df_cut_stats.iloc[i][col]

        # Remove nocut customers who don't have stats
        for col in collist:
            df = feature_table.loc[feature_table[col].isna(),
                                   ['occupant_id',label]].copy()
            occs_nocut_no_stats = df.loc[df[label] == 0, 'occupant_id'].values
            for occ in occs_nocut_no_stats:
                feature_table = \
                    feature_table.loc[feature_table['occupant_id'] != occ]

    else:
        
        # Set missing data to innocuous values
        for col in collist:
            feature_table.loc[feature_table[col].isna(), col] = 0
        
    return feature_table


# Prepare features for model input
def prep_features(config, df_meter, df_location, df_occupant,
                  df_volume, df_charge, df_cutoffs, mode):

    # Ensure date consistency
    print('converting date columns to datetime objects')
    [df_occupant, df_charge,
     df_volume, df_cutoffs] = prda.date_consistency(df_occupant, df_charge,
                                                    df_volume, df_cutoffs)

    if mode == 'train':
        nSamples_list = config['TRAINING']['N_SAMPLE_LIST']
        nSamples_list = list(map(int, nSamples_list))
        ref_date = config['TRAINING']['REF_DATE']
        outdir = config['PATHS']['FEATURE_TABLE_DIR_TRAIN']
    else:
        model_save_dir = config['PATHS']['MODEL_SAVE_DIR']
        ref_date_train = config['TRAINING']['REF_DATE']
        option_cut_prior = config['PREDICTION']['FEATURES_CUT_PRIOR']
        option_metadata = config['PREDICTION']['FEATURES_METADATA']
        option_anom = config['PREDICTION']['FEATURES_ANOM']
        opt_str_train = '{:s}.{:s}.{:s}.{:s}.{:s}'.format('train',
                                                          ref_date_train,
                                                          option_cut_prior,
                                                          option_metadata,
                                                          option_anom)
#        best_model_info_file = config['PATHS']['BEST_MODEL_INFO_FILE']
        best_model_info_file = model_save_dir + '/best_model_info.' + \
            opt_str_train + '.csv'
        df_best_model_info = pd.read_csv(best_model_info_file)
        nSamples_list = [int(df_best_model_info['nSamples'].values[0])]
        ref_date = config['PREDICTION']['REF_DATE']
        outdir = config['PATHS']['FEATURE_TABLE_DIR_PRED']

    print('creating feature tables for specified values of nSamples')
    np.random.seed(0)
    for nSamples in nSamples_list:
        print('....nSamples',nSamples)
        feature_table = create_and_prep_feature_table(df_occupant,
                                                      df_location,
                                                      df_meter,
                                                      df_cutoffs,
                                                      df_charge,
                                                      df_volume,
                                                      ref_date,
                                                      mode,
                                                      nSamples,
                                                      strict=True)

        outfile = outdir + '/feature_table.' + \
            '{:s}.{:s}.N{:02d}.csv'.format(mode, ref_date, nSamples)
        print('....writing to', outfile)
        feature_table.to_csv(outfile)

    if mode == 'train':
        return 0
    else:
        return feature_table


