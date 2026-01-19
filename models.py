import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, Integer, String
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


class ShortResult(Base):
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
