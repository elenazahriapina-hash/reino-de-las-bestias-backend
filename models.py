import datetime
import uuid

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import TEXT, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
    name: Mapped[str] = mapped_column(String(80))
    lang: Mapped[str] = mapped_column(String(5))
    gender: Mapped[str] = mapped_column(String(20), default="unspecified")


class RunAnswer(Base):
    __tablename__ = "run_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
    )
    question_id: Mapped[int] = mapped_column(Integer)
    answer: Mapped[str] = mapped_column(String(50))


class ShortResultORM(Base):
    __tablename__ = "short_results"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    animal: Mapped[str] = mapped_column(String(30))
    element: Mapped[str] = mapped_column(String(20))
    gender_form: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(TEXT)


class FullResultORM(Base):
    __tablename__ = "full_results"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    text: Mapped[str] = mapped_column(TEXT)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    telegram: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(120))
    lang: Mapped[str] = mapped_column(String(5))
    auth_token: Mapped[str] = mapped_column(String(64), unique=True)
    has_full: Mapped[bool] = mapped_column(Boolean, default=False)
    packs_bought: Mapped[int] = mapped_column(Integer, default=0)
    compat_credits: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


class UserResult(Base):
    __tablename__ = "user_results"

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    animal_code: Mapped[str] = mapped_column(String(30))
    element_ru: Mapped[str] = mapped_column(String(20))
    genderForm: Mapped[str] = mapped_column(String(20))
    short_text: Mapped[str] = mapped_column(TEXT)
    full_text: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


class CompatReport(Base):
    __tablename__ = "compat_reports"
    __table_args__ = (
        UniqueConstraint("user_low_id", "user_high_id", "prompt_version"),
        UniqueConstraint("request_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_low_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE")
    )
    user_high_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE")
    )
    language: Mapped[str] = mapped_column(String(5), default="ru")
    prompt_version: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(TEXT)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


class Invite(Base):
    __tablename__ = "invites"
    __table_args__ = (UniqueConstraint("request_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token: Mapped[str] = mapped_column(String(64), unique=True)
    inviter_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE")
    )
    invitee_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    prompt_version: Mapped[str] = mapped_column(String(120))
    credit_spent: Mapped[bool] = mapped_column(Boolean, default=False)
    credit_refunded: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20))
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
