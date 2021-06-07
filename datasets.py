"""Dataset registrations."""
import os

import numpy as np

import common

import pandas as pd

def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def LoadCover(filename="cover.csv"):
    filepath = './datasets/{}'.format(filename)
    df = pd.read_csv(filepath)
    return common.CsvTable(filename.rstrip(".csv"),df,cols=df.columns)

def LoadDmvMy(filename="dmv_three.csv"):
    filepath = './datasets/{}'.format(filename)
    df = pd.read_csv(filepath)
    return common.CsvTable(filename.rstrip(".csv"),df,cols=df.columns)

def LoadTpcH(filename="order_zip8.csv",sampling=False):
    filepath = './datasets/{}'.format(filename)
    df = pd.read_csv(filepath)
    if(sampling):
        df = df.sample(frac=sampling,random_state=1)
    return common.CsvTable(filename.rstrip(".csv"),df,cols=df.columns)
