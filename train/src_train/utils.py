import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropFeatures
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
from src_utils.utils import Clean, ExtractTitle

sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/..'))

def preprocess_train(
         data_path : str='ENTREGABLE_MODULO_III/data/data_titanic.csv',
         drop_features : list=['boat','body','home.dest','cabin', 'name', 'ticket', 'sex','embarked', 'cabin_letter','title','pclass','sibsp','parch',],
         numerical_features : list=['age','fare'],
         target : str='survived',
         random_state : int=666,
         save_model: bool=True
         ):

    """Preprocess and train model.

    Args:
      data_path: direction to titanic data
      drop_features: list vars to be droped
      categorical_features : list vars of categorical
      numerical_features : list vars of categorical
      target : str target var to be train 
      random_state : int 
  

    Returns:
      Saved model

    """
    # Loading data 
    df = pd.read_csv(data_path)
    
    model=LogisticRegression(random_state=random_state)

    pipeline = Pipeline(
                              [ ('clean_data', Clean()),
                                ('extract_letter', ExtractTitle()),
                                ('DropFeatures', DropFeatures(features_to_drop=drop_features)),
                                ('model', model)
                              ])

 
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),df[target],test_size=0.25,random_state=random_state)
    
    pipeline.fit(X_train, y_train)
    
    class_pred = pipeline.predict(X_test)
    proba_pred = pipeline.predict_proba(X_test)[:,1]
    print('test roc-auc : {}'.format(roc_auc_score(y_test, proba_pred)))
    print('test accuracy: {}'.format(accuracy_score(y_test, class_pred)))
    print('---------------------------------------')

    if save_model:
      model_pipeline_path = '../server/models/model_titanic.joblib'
      joblib.dump(pipeline, model_pipeline_path)
      message='Model stored'
    else:
      message='Model not stored'

    print(message)

    return pipeline