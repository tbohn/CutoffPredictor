# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew
import math

# Apply exponentially-decaying weights to x (backwards from end of array)
def exp_decay_wts(x, tau):
    N = len(x)
    w = np.empty([N])
    w[:] = [math.exp(-t/tau) for t in range(N-1,-1,-1)]
    return x * w, w

# Linear Regression
def linreg(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

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

# Z-transform of a point relative to the distribution of all points behind it
def ztrans_expanding_window(x, t0, t1):
    
    maxlen = t1 - t0
    ztrans_expanding = np.zeros([maxlen], dtype=np.float)
    
    for t in range(2, maxlen):
        mean = np.nanmean(x[t0:t0+t])
        std = np.nanstd(x[t0:t0+t])
        if std > 0:
            ztrans_expanding[t] = (x[t0+t] - mean) / std

    return ztrans_expanding

def clip_time_series_by_cutoffs(df, date_colname, date_first, date_last, cutoff_dates):
    nCut = len(cutoff_dates)
    df_list = []
    lengths = []
    date0 = date_first
    # allow for cutoff to occur 1 month after final billing date
    date_last = date_last + pd.DateOffset(months=1)
    for c in range(nCut):
        if cutoff_dates[c] > date_first and cutoff_dates[c] <= date_last:
            date1 = cutoff_dates[c]
            if date1 > date0:
                df_tmp = df.loc[df[date_colname].between(date0, date1)]
            else:
                df_tmp = df.loc[df[date_colname].isna()]
            date0 = date1
        else:
            df_tmp = df.loc[df[date_colname].isna()]
        df_list.append(df_tmp)
        lengths.append(len(df_tmp))
    return df_list, lengths

def select_segment(df_lens, N):
    nplengths = np.asarray(df_lens)
    long_idxs = np.asarray(np.where(nplengths >= N))
    long_lengths = nplengths[long_idxs]
    # If at least one segment is long enough to contain a complete sample window,
    # choose the most recent one
    if long_idxs.shape[-1] > 0:
        if long_idxs.ndim == 2:
            i = np.asscalar(long_idxs[0,-1])
        else:
            i = np.asscalar(long_idxs[-1])
    # Otherwise, choose the longest segment
    elif len(nplengths) > 0:
        i = np.argmax(nplengths)
    else:
        i = None
    return i


# Create table of predictors and predictands
def create_feature_table(df_occupant, df_location, df_meter, df_cutoffs,
                         df_charge, df_volume, today, mode='train',
                         N_sample=6):

    occlist = df_occupant['occupant_id'].unique()
    i = 0
    icut = 0
    inocut = 0
    k = 0

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
            location_id = df_occupant.loc[df_occupant['occupant_id'] == occ, 'location_id'].values[0]
        except:
            location_id = -1
        try:
            lat = df_location.loc[df_location['location_id'] == location_id, 'lat'].values[0]
        except:
            lat = 0.0
        try:
            lng = df_location.loc[df_location['location_id'] == location_id, 'lng'].values[0]
        except:
            lng = 0.0
        try:
            meter_address = df_location.loc[df_location['location_id'] == location_id, 'meter_address'].values[0]
        except:
            meter_address = 'unknown'
        try:
            municipality = df_location.loc[df_location['location_id'] == location_id, 'municipality'].values[0]
        except:
            municipality = '-1'
        try:
            meter_id = df_meter.loc[df_meter['location_id'] == location_id, 'meter_id'].values[0]
        except:
            meter_id = -1
        try:
            cust_type = df_meter.loc[df_meter['meter_id'] == meter_id, 'cust_type'].values[0]
        except:
            cust_type = 'unknown'
        try:
            cust_type_code = df_meter.loc[df_meter['meter_id'] == meter_id, 'cust_type_code'].values[0]
        except:
            cust_type_code = -1
        try:
            meter_size = df_meter.loc[df_meter['meter_id'] == meter_id, 'meter_size'].values[0]
        except:
            meter_size = -1

        # Compute time series stats, for those customers that have them
        
        # First, get cutoff dates, if any
        if cutoff:
            cutoff_dates = pd.to_datetime(np.asarray(df_cutoffs.loc[df_cutoffs['occupant_id'] == occ, 'cutoff_at'].sort_values()))
            cutoff_dates = cutoff_dates[cutoff_dates <= today]
            nCut = len(cutoff_dates)

        # Next, identify boundaries of time series of billing and usage; grab the time series, and do some
        # whole-timeseries normalization (standardized anomalies)
        if occ in df_charge['occupant_id'].unique():
            inCharge = True
            dfc = df_charge.loc[df_charge['occupant_id'] == occ].copy()
            bill_first = pd.to_datetime(dfc['billing_period_end'].min())
            bill_last = pd.to_datetime(dfc['billing_period_end'].max())
            bill_last = min(bill_last, today)
            dfc = dfc.loc[dfc['billing_period_end'].between(bill_first, bill_last)]
            nCharge = len(dfc)
        else:
            inCharge = False
            nCharge = 0
        if occ in df_volume['occupant_id'].unique():
            inVolume = True
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
#            for col in ['volume_kgals']:
                col_anom = col + '_anom'
                if nVolume > 24:
                    if nVolume % 12 > 0:
                        ntmp = (int(nVolume / 12) + 1) * 12
                    else:
                        ntmp = nVolume
                    nYears = int(ntmp / 12)
                    tmp = np.empty([ntmp])
                    tmp[:nVolume] = dfv[col].values
                    mmean = np.nanmean(tmp.reshape(nYears,12))
                    mstd = np.nanstd(tmp.reshape(nYears,12))
                    anom = (tmp.reshape(nYears,12) - mmean) / mstd
                    anom[np.isnan(anom)] = 0
                    anom[np.isinf(anom)] = 0
                    dfv[col_anom] = anom.reshape(nYears*12)[:nVolume]
                elif nVolume >= 3:
                    mean = np.nanmean(dfv[col])
                    std = np.nanstd(dfv[col])
                    if std > 0:
                        dfv[col_anom] = (dfv[col] - mean) / std
                    else:
                        dfv[col_anom] = dfv[col] - mean
                    dfv.loc[dfv[col_anom].isna()] = 0
                else:
                    dfv[col_anom] = np.nan
        else:
            inVolume = False
            nVolume = 0

        # Clip sample windows out of time series
        dfc_list = []
        dfv_list = []
        t0c_list = []
        t1c_list = []
        t0v_list = []
        t1v_list = []
        N_samples = 0
        if mode == 'train':
            # For train cutoff time series, clip time series into segments in between cutoffs (if any) and
            # select the most recent segment that can accomodate a sample window; window's end is immediately
            # before the cutoff
            if cutoff:
                nCP_array = np.arange(nCut).astype(int)
                CP_array = np.where(nCP_array > 0, 1, 0).astype(int)
                dfc_list, dfc_lens = clip_time_series_by_cutoffs(dfc, 'billing_period_end',
                                                                 bill_first, bill_last, cutoff_dates)
                dfv_list, dfv_lens = clip_time_series_by_cutoffs(dfv, 'meter_read_at',
                                                                 read_first, read_last, cutoff_dates)
                for c in range(nCut):
                    t0c_list.append(max(len(dfc_list[c]) - N_sample, 0))
                    t1c_list.append(len(dfc_list[c]))
                    t0v_list.append(max(len(dfv_list[c]) - N_sample, 0))
                    t1v_list.append(len(dfv_list[c]))
                N_samples = nCut
            # For train nocut time series, clip out sample window from random location anywhere in history
            else:
                dfc_list.append(dfc.copy())
                t0c = np.asscalar(np.random.uniform(0, (nCharge - N_sample), 1).astype(int))
                t0c_list.append(t0c)
                t1c_list.append(t0c + N_sample)
                dfv_list.append(dfv.copy())
                t0v = np.asscalar(np.random.uniform(0, (nVolume - N_sample), 1).astype(int))
                t0v_list.append(t0v)
                t1v_list.append(t0v + N_sample)
                nCP_array = np.arange(0,1).astype(int)
                CP_array = np.where(nCP_array > 0, 1, 0).astype(int)
                N_samples = 1
        # For predictions from current conditions, clip out sample window from N months prior to today
        else:
            dfc_list.append(dfc.copy())
            t1c = nCharge
            t1c_list.append(t1c)
            t0c_list.append(max(t1c - N_sample, 0))
            dfv_list.append(dfv.copy())
            t1v = nVolume
            t1v_list.append(t1v)
            t0v_list.append(max(t1v - N_sample, 0))
            if cutoff:
                nCP_array = np.arange(nCut,nCut+1).astype(int)
                CP_array = np.where(nCP_array > 0, 1, 0).astype(int)
            else:
                nCP_array = np.arange(0,1)
                CP_array = np.where(nCP_array > 0, 1, 0).astype(int)
            N_samples = 1

        for s in range(N_samples):
            
            t0c = t0c_list[s]
            t1c = t1c_list[s]
            t0v = t0v_list[s]
            t1v = t1v_list[s]
            dfc_samp = dfc_list[s]
            dfv_samp = dfv_list[s]
            nCutPrior = nCP_array[s]
            cut_prior = CP_array[s]

            # Compute statistics of sample window
            if nCharge > 0 and t1c > t0c:
                if t1c > t0c:
                    late_frac = dfc_samp['late'][t0c:t1c].mean(skipna=True)
                    mean_charge_tot = dfc_samp['total_charge'][t0c:t1c].mean(skipna=True)
                    mean_charge_late = dfc_samp['late_charge'][t0c:t1c].mean(skipna=True)
                else:
                    late_frac = np.nan
                    mean_charge_tot = np.nan
                    mean_charge_late = np.nan
            if nVolume > 0 and t1v - t0v:
                df_tmp = dfv_samp.iloc[t0v:t1v].copy()
                mean_vol = df_tmp['volume_kgals'].mean(skipna=True)
                if t1v > t0v:
                    zero_frac_vol = df_tmp.loc[df_tmp['volume_kgals'] == 0, 'volume_kgals'].count() / (t1v - t0v)
                else:
                    zero_frac_vol = 0
                if t1v - t0v >= 3:
                    skew_vol = abs(skew(np.asarray(df_tmp['volume_kgals'])))
                    n_anom3_vol = df_tmp.loc[abs(df_tmp['volume_kgals_anom']) > 3, 'volume_kgals_anom'].count()
                    n_anom4_vol = df_tmp.loc[abs(df_tmp['volume_kgals_anom']) > 4, 'volume_kgals_anom'].count()
                    n_anom5_vol = df_tmp.loc[abs(df_tmp['volume_kgals_anom']) > 5, 'volume_kgals_anom'].count()
                    max_anom_vol = np.max(np.abs(df_tmp['volume_kgals_anom']))
                    n_anom3_vol_log = df_tmp.loc[abs(df_tmp['vol_log_anom']) > 3, 'vol_log'].count()
                    n_anom4_vol_log = df_tmp.loc[abs(df_tmp['vol_log_anom']) > 4, 'vol_log'].count()
                    n_anom5_vol_log = df_tmp.loc[abs(df_tmp['vol_log_anom']) > 5, 'vol_log'].count()
                    max_anom_vol_log = np.max(np.abs(df_tmp['vol_log_anom']))
                    vol_dev = ztrans(np.asarray(df_tmp['volume_kgals']))
                    vol_dev_abs = np.abs(vol_dev)
                    n_anom3_local_vol = len(vol_dev_abs[vol_dev_abs > 3])
                    n_anom4_local_vol = len(vol_dev_abs[vol_dev_abs > 4])
                    n_anom5_local_vol = len(vol_dev_abs[vol_dev_abs > 5])
                    max_anom_local_vol = np.asscalar(np.max(vol_dev_abs))
                    vol_log_dev = ztrans(np.asarray(df_tmp['vol_log']))
                    vol_log_dev_abs = np.abs(vol_log_dev)
                    n_anom3_local_vol_log = len(vol_log_dev_abs[vol_log_dev_abs > 3])
                    n_anom4_local_vol_log = len(vol_log_dev_abs[vol_log_dev_abs > 4])
                    n_anom5_local_vol_log = len(vol_log_dev_abs[vol_log_dev_abs > 5])
                    max_anom_local_vol_log = np.asscalar(np.max(vol_log_dev_abs))
                else:
                    skew_vol = np.nan
                    n_anom3_vol = np.nan
                    n_anom4_vol = np.nan
                    n_anom5_vol = np.nan
                    max_anom_vol = np.nan
                    n_anom3_vol_log = np.nan
                    n_anom4_vol_log = np.nan
                    n_anom5_vol_log = np.nan
                    max_anom_vol_log = np.nan
                    n_anom3_local_vol = np.nan
                    n_anom4_local_vol = np.nan
                    n_anom5_local_vol = np.nan
                    max_anom_local_vol = np.nan
                    n_anom3_local_vol_log = np.nan
                    n_anom4_local_vol_log = np.nan
                    n_anom5_local_vol_log = np.nan
                    max_anom_local_vol_log = np.nan
            
            df = pd.DataFrame(data={'occupant_id': occ,
                                    'location_id': location_id,
                                    'lat': lat,
                                    'lng': lng,
                                    'meter_address': meter_address,
                                    'municipality': municipality,
                                    'cust_type': cust_type,
                                    'cust_type_code': cust_type_code,
                                    'meter_size': meter_size,
                                    'sample': s,
                                    'cutoff': cutoff,
                                    'nCutPrior': nCutPrior,
                                    'cut_prior': cut_prior,
                                    'late_frac': late_frac,
                                    'mean_charge_tot': mean_charge_tot,
                                    'mean_charge_late': mean_charge_late,
                                    'skew_vol': skew_vol,
                                    'zero_frac_vol': zero_frac_vol,
                                    'n_anom3_vol': n_anom3_vol,
                                    'n_anom4_vol': n_anom4_vol,
                                    'n_anom5_vol': n_anom5_vol,
                                    'max_anom_vol': max_anom_vol,
                                    'n_anom3_vol_log': n_anom3_vol_log,
                                    'n_anom4_vol_log': n_anom4_vol_log,
                                    'n_anom5_vol_log': n_anom5_vol_log,
                                    'max_anom_vol_log': max_anom_vol_log,
                                    'n_anom3_local_vol': n_anom3_local_vol,
                                    'n_anom4_local_vol': n_anom4_local_vol,
                                    'n_anom5_local_vol': n_anom5_local_vol,
                                    'max_anom_local_vol': max_anom_local_vol,
                                    'n_anom3_local_vol_log': n_anom3_local_vol_log,
                                    'n_anom4_local_vol_log': n_anom4_local_vol_log,
                                    'n_anom5_local_vol_log': n_anom5_local_vol_log,
                                    'max_anom_local_vol_log': max_anom_local_vol_log,
                                    'mean_vol': mean_vol,
                                   },
                             index=[k])
            if k == 0:
                feature_table = df.copy()
            else:
                feature_table = feature_table.append(df, ignore_index=True, sort=False)
            k += 1
            if cutoff:
                icut += 1
            else:
                inocut += 1
            i += 1

    return feature_table
        
        
def create_and_prep_feature_table(df_occupant, df_location, df_meter, df_cutoffs,
                                  df_charge_align_clean, df_volume_align_clean,
                                  today, mode, N_sample):

    # Create feature table
    feature_table = create_feature_table(df_occupant, df_location, df_meter, df_cutoffs,
                                         df_charge_align_clean, df_volume_align_clean,
                                         today, mode, N_sample)

    collist = ['late_frac','mean_charge_tot','mean_charge_late','mean_vol','skew_vol',
               'n_anom3_vol','n_anom4_vol','n_anom5_vol','max_anom_vol',
               'n_anom3_vol_log','n_anom4_vol_log','n_anom5_vol_log','max_anom_vol_log',
               'n_anom3_local_vol','n_anom4_local_vol','n_anom5_local_vol','max_anom_local_vol',
               'n_anom3_local_vol_log','n_anom4_local_vol_log','n_anom5_local_vol_log','max_anom_local_vol_log',
              ]

    # Handle customers that are missing stats
    if mode == 'train':
        # Populate stats of cutoff customers who have no stats with random samples from those with stats
        for col in collist:
            df = feature_table.loc[feature_table[col].isna(), ['occupant_id','cutoff']].copy()
            occs_cut_no_stats = df.loc[df['cutoff'] == 1, 'occupant_id'].values
            df = feature_table.loc[feature_table[col].notna(), ['occupant_id','cutoff']].copy()
            occs_cut_stats = df.loc[df['cutoff'] == 1, 'occupant_id'].values
            nOccsCutStats = len(occs_cut_stats)
            df = feature_table.loc[feature_table[col].notna()].copy()
            df_cut_stats = df.loc[df['cutoff'] == 1].copy()
            for occ in occs_cut_no_stats:
                i = int(np.random.uniform(0, nOccsCutStats, 1))
                feature_table.loc[feature_table['occupant_id'] == occ, col] = df_cut_stats.iloc[i][col]

        # Remove nocut customers who don't have stats
        for col in collist:
            df = feature_table.loc[feature_table[col].isna(), ['occupant_id','cutoff']].copy()
            occs_nocut_no_stats = df.loc[df['cutoff'] == 0, 'occupant_id'].values
            for occ in occs_nocut_no_stats:
                feature_table = feature_table.loc[feature_table['occupant_id'] != occ]

    else:
        
        # Set missing data to innocuous values
        for col in collist:
            feature_table.loc[feature_table[col].isna(), col] = 0
        
    return feature_table


# Prepare features for model training
def prep_features(config, df_meter, df_location, df_occupant,
                  df_volume, df_charge, df_cutoffs, mode):

    if mode == 'train':
        N_sample_list = config['N_sample_list']
        N_realizations = config['N_realizations']
        ref_day = config['train_day']
        outdir = config['feature_table_hist_dir']
    else:
# read N_sample from a file!!!!!!!!!
# filename must be derivable from config dict: config['best_model_info_file']
        N_sample_list = [N_sample]
        N_realizations = 1
        ref_day = config['pred_day']
        outdir = config['feature_table_curr_dir']

    np.random.seed(0)
    for N_sample in N_sample_list:
        for r in range(N_realizations):
            feature_table = create_and_prep_feature_table(df_occupant,
                                                          df_location,
                                                          df_meter,
                                                          df_cutoffs,
                                                          df_charge,
                                                          df_volume,
                                                          ref_day,
                                                          mode, N_sample)

            if mode == 'train':
                rstr = '.r{:d}'.format(r)
            else:
                rstr = ''
            outfile = outdir + '/feature_table.{:s}.N{:02d}.{:s}{:s}.csv'.format(config['train_day'], N_sample, mode, rstr)
            print('writing to', outfile)
            feature_table.to_csv(outfile)

    if mode == 'train':
        return 0
    else:
        return feature_table


