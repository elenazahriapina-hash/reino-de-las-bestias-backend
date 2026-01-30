from typing import Literal

from pydantic import BaseModel


class TestAnswer(BaseModel):
    questionId: int
    answer: str


class AnalyzeRequest(BaseModel):
    name: str
    lang: Literal["ru", "en", "es", "pt"]
    gender: Literal["male", "female", "unspecified"] = "unspecified"
    answers: list[TestAnswer]
    lockedAnimal: str | None = None
    lockedElement: str | None = None
    lockedGenderForm: str | None = None
    runId: str | None = None


class ShortResult(BaseModel):
    runId: str | None = None
    animal: str
    element: str
    genderForm: str
    text: str


class ShortResponse(BaseModel):
    type: Literal["short"]
    result_id: str
    result: ShortResult


class FullRequest(BaseModel):
    result_id: str


class FullResult(BaseModel):
    animal: str
    element: str
    genderForm: str
    text: str


class FullResponse(BaseModel):
    type: Literal["full"]
    result_id: str
    result: FullResult
