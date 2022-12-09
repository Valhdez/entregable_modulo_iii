import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class Clean(BaseEstimator, TransformerMixin):
    """clean the data.

    Args: df

    Returns: data tranformed


    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        try :
            X.replace('?',np.nan,inplace=True)
            X['age'] = X['age'].astype(float)
            X['fare'] = X['fare'].astype(float)
            X['num_dest'] = X['home.dest'].astype(str).apply(lambda x : len( x.split('/')))
            X.loc[X['home.dest'].isnull(),'num_dest'] = 0
            X['if_body'] = X.body.notnull() * 1
            X.fillna(0,inplace=True)
        except :
          print('There is not variable cabin')
        return X

class ExtractTitle(BaseEstimator, TransformerMixin):
    """Extract the title name.

    Args:

    Returns:
      X: dataframe with new variable title, if exist name.

    Raises:
      'There is not variable name'

    """
    def __init__(self):
        self.variable = 'name'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        try:
          X['title'] = X[self.variable].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        except:
          print('There is not variable name')        
        return X