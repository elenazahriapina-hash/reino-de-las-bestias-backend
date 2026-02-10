from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
    name: str | None = None
    lang: Literal["ru", "en", "es", "pt"] | None = None
    shortResult: ShortResult | None = None


class GoogleAuthRequest(BaseModel):
    idToken: str
    lang: Literal["ru", "en", "es", "pt"] | None = None
    name: str | None = None


class TelegramAuthRequest(BaseModel):
    id: str | int
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    photo_url: str | None = None
    auth_date: int
    hash: str


class UserResponse(BaseModel):
    id: int
    email: str | None
    telegram: str | None
    name: str
    lang: str
    has_full: bool = Field(..., alias="hasFull")
    packs_bought: int = Field(..., alias="packsBought")
    compat_credits: int = Field(..., alias="compatCredits")
    created_at: datetime

    class Config:
        allow_population_by_field_name = True


class RegisterResponse(BaseModel):
    userId: int
    token: str
    credits: int
    hasFull: bool


class DevSeedUserRequest(BaseModel):
    email: str | None = None
    telegram: str | None = None
    name: str
    lang: Literal["ru", "en", "es", "pt"]
    animal: str
    element: str
    genderForm: str
    short_text: str


class DevSeedUserResponse(BaseModel):
    userId: int
    token: str


class UserMeResponse(BaseModel):
    credits: int
    has_full: bool = Field(..., alias="hasFull")
    user_id: int = Field(..., alias="userId")
    lang: str

    class Config:
        allow_population_by_field_name = True


class LookupUserResponse(BaseModel):
    id: int
    name: str
    lang: str


class CompatibilityLookupRequest(BaseModel):
    q: str | None = None
    email: str | None = None
    telegram: str | None = None


class CompatibilityCheckRequest(BaseModel):
    target_user_id: int = Field(..., alias="targetUserId")
    requestId: str | None = None
    lang: Literal["ru", "en", "es", "pt"]

    model_config = {"populate_by_name": True}


class CompatibilityPackPurchaseRequest(BaseModel):
    packSize: Literal[3, 10]
    requestId: str | None = None


class CompatibilityInviteRequest(BaseModel):
    email: str | None = None
    telegram: str | None = None
    requestId: str | None = None

    model_config = {"populate_by_name": True}


class CompatibilityAcceptInviteRequest(BaseModel):
    token: str


class CompatibilityInviteResponse(BaseModel):
    token: str
    status: str
    prompt_version: str
    created_at: datetime


class CompatibilityCounterpart(BaseModel):
    id: int
    name: str
    email: str | None
    telegram: str | None
    lang: str


class CompatibilityReportResponse(BaseModel):
    id: int
    reportId: int
    other_user_id: int
    lang: str
    prompt_version: str
    status: str
    text: str
    created_at: datetime
    createdAt: datetime
    counterpart: CompatibilityCounterpart | None = None

    class Config:
        allow_population_by_field_name = True


class CompatibilityListResponse(BaseModel):
    items: list[CompatibilityReportResponse]
    history: list[CompatibilityReportResponse]
