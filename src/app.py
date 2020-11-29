import os
import warnings
from typing import List

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from src.predictor import detect_elements
from src.utils import store_sketch

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

app = FastAPI(
    title="MetaMorph API",
    description=(
        "Web API for MetaMorph, a UI element detector that detects constituent UI elements of"
        " freehand lo-fi sketches using DNN based object detection models."
    ),
    version="1.0.0",
)

os.makedirs("./sketches", exist_ok=True)


@app.get("/", include_in_schema=False)
async def redirect_to_home():
    return RedirectResponse("https://metamorph.designwitheve.com")


class PositionSchema(BaseModel):
    x: int
    y: int


class DimensionSchema(BaseModel):
    width: int
    height: int


class PredictionResponse(BaseModel):
    name: str
    position: PositionSchema
    dimension: DimensionSchema
    probability: float


@app.post(
    "/predict/",
    response_model=List[PredictionResponse],
    status_code=status.HTTP_200_OK,
    response_description=(
        "Responds with a list of UI element categories present in the input low fidelity sketch"
        " with their location, and their prediction certainty in JSON format. This JSON file"
        " contains a list of predicted UI element categories as JSON objects. Each JSON object"
        " contains predicted bounding box position (top left x,y coordinates) and its dimensions"
        " (width, height)"
    ),
    tags=["Prediction"],
    description="Detect UI elements from low fidelity sketch",
)
async def predict(
    image: UploadFile = File(
        ...,
        description=(
            "An image file (jpg or png) of low fidelity prototype sketch and a minimum detection"
            " threshold value"
        ),
    ),
    minimum_probability: float = Query(
        0.8,
        gt=0,
        lt=1,
        description="Minimum detection probability. Filters elements below this probability",
    ),
):

    mime_type = image.content_type

    if mime_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Image is not a JPG or PNG"
        )

    image_path = store_sketch(image)

    img = cv2.imread(image_path)

    response: List[PredictionResponse] = detect_elements(image=img, min_prob=minimum_probability)

    return response
