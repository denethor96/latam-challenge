import logging
from enum import Enum, IntEnum
from functools import lru_cache
from pathlib import Path
from typing import List

import fastapi
import pandas as pd
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from challenge.model import AIRLINE_COL, DelayModel


logger = logging.getLogger(__name__)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"


class FlightType(str, Enum):
    INTERNATIONAL = "I"
    NATIONAL = "N"


class Month(IntEnum):
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


@lru_cache(maxsize=1)
def get_valid_airlines() -> set[str]:
    """
    Keep airline validation aligned with the training data instead of hardcoding
    duplicated categories in the API layer.
    """
    data = pd.read_csv(
        DATA_PATH,
        usecols=[AIRLINE_COL],
        low_memory=False,
    )
    return set(data[AIRLINE_COL].dropna().unique())


class FlightSchema(BaseModel):
    OPERA: str
    TIPOVUELO: FlightType
    MES: Month

    @field_validator("OPERA")
    @classmethod
    def validate_airline(cls, value: str) -> str:
        if value not in get_valid_airlines():
            raise ValueError("Invalid OPERA.")
        return value


class PredictionRequestSchema(BaseModel):
    flights: List[FlightSchema]


class PredictionResponseSchema(BaseModel):
    predict: List[int]


app = fastapi.FastAPI()
model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: fastapi.Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    The challenge tests expect invalid request payloads to return HTTP 400.
    FastAPI returns HTTP 422 by default for schema validation errors.
    """
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )

@app.on_event("startup")
def load_model() -> None:
    """
    Initialize the model once during application startup.

    The model is loaded from a persisted artifact when available. If the artifact
    does not exist, it is trained once from the bundled dataset as a local/test
    fallback. Prediction requests never trigger repeated training.
    """
    model.ensure_model_is_ready()
    logger.info("Delay model is ready for prediction")

@app.post("/predict", status_code=200, response_model=PredictionResponseSchema)
async def post_predict(
    request: PredictionRequestSchema,
) -> PredictionResponseSchema:
    logger.info("Received prediction batch with %d flights", len(request.flights))

    # JSON mode converts Enum values to the raw values expected by the model.
    data = pd.DataFrame(
        [flight.model_dump(mode="json") for flight in request.flights],
    )

    features = model.preprocess(data=data)
    predictions = model.predict(features=features)

    logger.info("Generated %d predictions", len(predictions))

    return PredictionResponseSchema(predict=predictions)