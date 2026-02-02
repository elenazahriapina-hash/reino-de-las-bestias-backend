from datetime import datetime
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


class RegisterRequest(BaseModel):
    email: str | None = None
    telegram: str | None = None
    name: str
    lang: Literal["ru", "en", "es", "pt"]


class UserResponse(BaseModel):
    id: int
    email: str | None
    telegram: str | None
    name: str
    lang: str
    has_full: bool
    packs_bought: int
    compat_credits: int
    created_at: datetime


class RegisterResponse(UserResponse):
    auth_token: str


class LookupUserResponse(BaseModel):
    id: int
    name: str
    lang: str


class CompatibilityCheckRequest(BaseModel):
    target_user_id: int
    requestId: str | None = None


class CompatibilityInviteRequest(BaseModel):
    email: str | None = None
    telegram: str | None = None
    requestId: str | None = None


class CompatibilityAcceptInviteRequest(BaseModel):
    token: str


class CompatibilityInviteResponse(BaseModel):
    token: str
    status: str
    prompt_version: str
    created_at: datetime


class CompatibilityReportResponse(BaseModel):
    id: int
    other_user_id: int
    prompt_version: str
    status: str
    text: str
    created_at: datetime


class CompatibilityListResponse(BaseModel):
    items: list[CompatibilityReportResponse]
