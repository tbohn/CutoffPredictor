# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def save_tables(outdir, df_meter, df_location, df_occupant,
                df_volume, df_charge, df_cutoffs):

    df_meter.to_csv(outdir + '/meter.csv')
    df_location.to_csv(outdir + '/location.csv')
    df_occupant.to_csv(outdir + '/occupant.csv')
    df_volume.to_csv(outdir + '/volume.csv')
    df_charge.to_csv(outdir + '/charge.csv')
    df_cutoffs.to_csv(outdir + '/cutoffs.csv')

def read_tables(indir)

    df_meter = read_csv(indir + '/meter.csv')
    df_location = read_csv(indir + '/location.csv')
    df_occupant = read_csv(indir + '/occupant.csv')
    df_volume = read_csv(indir + '/volume.csv')
    df_charge = read_csv(indir + '/charge.csv')
    df_cutoffs = read_csv(indir + '/cutoffs.csv')

    return [df_meter, df_location, df_occupant,
            df_volume, df_charge, df_cutoffs]
