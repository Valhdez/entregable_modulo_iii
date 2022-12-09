from pathlib import Path
import sys
import os

import pytest
import pandas as pd

# Add root to sys.path
# https://fortierq.github.io/python-import/
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..'))
from classifier.classifier import TitanicClassifier
from models.models import Titanic

def get_test_response_data() -> list:
    return [(1, 'Sutton, Mr. Frederick', 'male', '61', 0, 0, '36963', 32.3208, 'D50', 'S', '?', '46', 'Haddenfield, NJ', 'survive', 0.57), 
            (2, 'Peruschitz, Rev. Joseph Maria', 'male', '41', 0, 0, '237393', 13.0, '?', 'S', '?', '?', '?', 'survive', 0.57), 
            (1, 'Andrews, Mr. Thomas Jr', 'male', '39', 0, 0, '112050', 0.0, 'A36', 'S', '?', '?', 'Belfast, NI', 'survive', 0.57), 
            (1, 'Silvey, Mrs. William Baird (Alice Munger)', 'female', '39', 1, 0, '13507', 55.9, 'E44', 'S', '11', '?', 'Duluth, MN', 'survive', 0.57), 
            (3, "Johnston, Mrs. Andrew G (Elizabeth 'Lily' Watson)", 'female', '?', 1, 2, 'W./C. 6607', 23.45, '?', 'S', '?', '?', '?', 'survive', 0.57), 
            (3, 'Yousif, Mr. Wazli', 'male', '?', 0, 0, '2647', 7.225, '?', 'C', '?', '?', '?', 'survive', 0.57), 
            (3, 'Danbom, Mr. Ernst Gilbert', 'male', '34', 1, 1, '347080', 14.4, '?', 'S', '?', '197', 'Stanton, IA', 'survive', 0.57), 
            (3, 'Demetri, Mr. Marinko', 'male', '?', 0, 0, '349238', 7.8958, '?', 'S', '?', '?', '?', 'survive', 0.57), 
            (2, 'Byles, Rev. Thomas Roussel Davids', 'male', '42', 0, 0, '244310', 13.0, '?', 'S', '?', '?', 'London', 'survive', 0.57), 
            (1, 'Kimball, Mr. Edwin Nelson Jr', 'male', '42', 1, 0, '11753', 52.5542, 'D19', 'S', '5', '?', 'Boston, MA', 'survive', 0.57)]


@pytest.mark.parametrize(
    "pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked, boat, body, home_dest, predict_survived, predict_proba",
    get_test_response_data(),
)
def test_response_parametrize(
    pclass: any,
    name: any,
    sex: any,
    age: any,
    sibsp: any,
    parch: any, 
    ticket: any, 
    fare: any, 
    cabin: any, 
    embarked: any, 
    boat: any, 
    body: any, 
    home_dest: any, 
    predict_survived: any, 
    predict_proba: any
) -> None:
    titanic = Titanic(
        pclass=pclass,
        name=name,
        sex=sex,
        age=age,
        sibsp=sibsp,
        parch=parch, 
        ticket=ticket, 
        fare=fare, 
        cabin=cabin,
        embarked=embarked,
        boat=boat,
        body=body,
        home_dest=home_dest,
    )

    classifier = TitanicClassifier()

    assert type(classifier.classify_titanic(titanic)["class"]) == type(predict_survived)
