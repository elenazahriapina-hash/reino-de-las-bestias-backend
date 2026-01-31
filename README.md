# Reino de las Bestias — Backend 🐍⚙️

Backend service for the **Reino de las Bestias** project.

Provides API endpoints for:
- processing user answers
- archetype (animal + element) analysis
- generating short textual interpretations

## 🧠 Stack
- Python
- FastAPI
- OpenAI API

## ▶️ Run locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Postgres (default credentials):
```bash
docker compose up -d
```

3. Create tables:
```bash
python create_tables.py
```

4. Run the API:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
```

## 🗄️ Database

Default connection string (from `.env`):
```
postgresql+asyncpg://reino:reino_pass@localhost:5432/reino
```

**psql quick connect:**
```bash
psql postgresql://reino:reino_pass@localhost:5432/reino
```

## 🛠️ Troubleshooting

* **`/analyze/short` returns 200 but `short_results` is empty:** ensure Postgres is running and `create_tables.py` has been run. Check that `.env` points to the same database as your `psql` session.
* **`/analyze/full` fails with "short_result missing":** the short result was not persisted for the run. Verify `short_results` contains a row for the given `result_id` and re-run `/analyze/short` to regenerate it.