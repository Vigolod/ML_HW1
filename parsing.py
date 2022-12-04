import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import category_encoders
import pickle


with open('ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('medians.pkl', 'rb') as f:
    medians = pickle.load(f)

def prepare_df(df):
    df = df.copy()
    df.drop('mileage', axis=1, inplace=True)
    
    for column in ['engine', 'max_power']:
        df[column] = pd.to_numeric(df[column].str.extract(r'(\d+\.?\d*)', expand=False))
    
    transl = str.maketrans('', '', ',')
    
    df['max_torque_rpm'] = df['torque'].str.extract('(?:@|at)\s*([-\d,\.]+)', expand=False) \
                                           .str.translate(transl)
    is_range = df['max_torque_rpm'].str.contains('-', na=False)
    first_term = pd.to_numeric(
        df.loc[is_range, 'max_torque_rpm'] \
          .str.extract(r'([\d,\.]+)', expand = False)
    )
    second_term = pd.to_numeric(
        df.loc[is_range, 'max_torque_rpm'] \
          .str.extract(r'-(\d+)', expand = False)
    )
    df.loc[is_range, 'max_torque_rpm'] = (first_term + second_term) / 2
    df['max_torque_rpm'] = pd.to_numeric(df['max_torque_rpm'])
    df.drop(['torque'], axis=1, inplace=True)

    df['name'] = df['name'].str.extract(r'\s*([a-zA-Z]+)\s', expand=False)

    df['max_torque_rpm'] = (df['max_torque_rpm'] < 2500).astype('int')
    
    df['km_driven'] = np.log(df['km_driven'] + 1e-10)

    df.loc[(df['owner'] == 'Third Owner') | 
            (df['owner'] == 'Fourth & Above Owner'),
            'owner'] = 'Third & Above Owner'
    df.loc[df['owner'] == 'Test Drive Car', 'owner'] = 'First Owner'
    
    df['year_sq'] = df['year'] ** 2
    
    df = df.fillna(medians)
    df[['engine', 'seats', 'max_torque_rpm']] = df[['engine', 'seats', 'max_torque_rpm']].astype('int')
    return df