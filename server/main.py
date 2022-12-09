import logging
import os
import sys

from fastapi import FastAPI
from starlette.responses import JSONResponse

sys.path.append( os.path.abspath(os.path.dirname(__file__)+'../..'))

from models.models import Titanic
from classifier.classifier import TitanicClassifier


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s: %(asctime)s|%(name)s|%(message)s")

file_handler = logging.FileHandler("server.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)  # Se agrega handler para stream

app = FastAPI()

@app.get("/")
def read_root():
    return "Titanic classifier is all ready to go!"


@app.get("/healthcheck", status_code=200)
async def healthcheck():
    logger.info("Servers is all ready to go!")
    return "Titanic classifier is all ready to go!"


@app.post("/classify_iris")
async def classify(titanic_features: Titanic):
    logger.debug(f"Incoming titanic features to the server: {titanic_features}")
    titanic_classifier = TitanicClassifier()
    response = JSONResponse(titanic_classifier.classify_titanic(titanic_features))
    logger.debug(f"Outgoing classification from the server: {response}")
    return response
