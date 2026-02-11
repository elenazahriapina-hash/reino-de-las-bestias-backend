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

If you override credentials, update both `.env` and your `psql` connection string so they point at the same database.

## 🔐 Auth environment variables

Add the following to your `.env` when enabling OAuth logins:

```
GOOGLE_WEB_CLIENT_ID=143867517052-471n3m08rqq1h38pgsr2d9692m40npdb.apps.googleusercontent.com
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_AUTH_MAX_AGE_SECONDS=86400
```

## 🛠️ Troubleshooting

* **`/analyze/short` returns 200 but `short_results` is empty:** confirm Postgres is running and `create_tables.py` has been run, then check that `.env` points to the same database as your `psql` session. The response `result_id` should always exist as a row in `short_results`.
```sql
  select * from short_results where run_id = '<result_id>';
  ```
* **`/analyze/full` fails with "short_result missing":** this indicates a data integrity issue (the run exists without a short result). Verify `short_results` contains a row for the given `result_id` and re-run `/analyze/short` to regenerate it.
```sql
  select * from runs where id = '<result_id>';
  select * from short_results where run_id = '<result_id>';
  select * from run_answers where run_id = '<result_id>';
  ```
* **`/analyze/full` with invalid `result_id`:** ensure you pass a UUID string in the payload `{ "result_id": "<uuid>" }`.
main.py
main.py
