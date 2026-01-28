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
