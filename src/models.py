from pydantic import BaseModel


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
