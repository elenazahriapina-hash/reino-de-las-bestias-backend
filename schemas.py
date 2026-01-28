from typing import List, Literal, Optional

from pydantic import BaseModel


class TestAnswer(BaseModel):
    questionId: int
    answer: str


class AnalyzeRequest(BaseModel):
    name: str
    lang: Literal["ru", "en", "es", "pt"]
    gender: Literal["male", "female", "unspecified"] = "unspecified"
    answers: List[TestAnswer]
    lockedAnimal: Optional[str] = None
    lockedElement: Optional[str] = None
    lockedGenderForm: Optional[str] = None
