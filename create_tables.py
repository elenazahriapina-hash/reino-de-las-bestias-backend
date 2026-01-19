from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv


async def main() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    try:
        from db import engine
        from models import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("OK: tables created")
    except Exception as exc:  # noqa: BLE001
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
