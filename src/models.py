from typing import List

from pydantic import BaseModel


class PositionSchema(BaseModel):
    x: int
    y: int


class DimensionSchema(BaseModel):
    width: int
    height: int


class ObjectsSchema(BaseModel):
    name: str
    position: PositionSchema
    dimension: DimensionSchema
    probability: float


class ResponseSchema(BaseModel):
    id: str
    width: int
    height: int
    objects: List[ObjectsSchema]
