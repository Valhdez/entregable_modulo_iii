import os
import sys
import pandas as pd
import numpy as np
import functools
from sklearn.pipeline import Pipeline
import pytest
import shutil
from datetime import datetime

sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..'))
from src_utils.utils import Clean, ExtractTitle


@pytest.fixture(scope="function")
def df_natural():
    return pd.read_csv(r'../entregable_modulo_iii/data/test_utils.csv')



@pytest.fixture(scope="function")
def df_test():
    numerical_features=['age','fare']

    df=pd.read_csv(r'../entregable_modulo_iii/data/test_utils.csv')

    df['title'] = df['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    df.replace('?', np.nan, inplace=True)
    # Converting numerical features to float64
    df[numerical_features]=df[numerical_features].astype('float64', errors='ignore')
    #create new columns in clean function
    df['num_dest'] = df['home.dest'].astype(str).apply(lambda x : len( x.split('/')))
    df.loc[df['home.dest'].isnull(),'num_dest'] = 0
    df['if_body'] = df.body.notnull() * 1
    df.fillna(0,inplace=True)

    return df

def obtener_datos_test_integration():
    return [(True)]

@pytest.mark.parametrize('Bool', obtener_datos_test_integration())
def test_add_to_list(Bool, df_test, df_natural):

    pipeline = Pipeline([ ('extract_letter', ExtractTitle()),
                          ('clean_data', Clean())
    ])

    pd.testing.assert_frame_equal(pipeline.transform(df_natural), df_test)