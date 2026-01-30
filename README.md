# Reino de las Bestias — Backend 🐍⚙️

Backend service for the **Reino de las Bestias** project.

Provides API endpoints for:
- processing user answers
- archetype (animal + element) analysis
- generating short textual interpretations
- generating full textual interpretations based on stored short results

## 🧠 Stack
- Python
- FastAPI
- OpenAI API

## ▶️ Run locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔌 API (short → full flow)

### 1) Generate a short result (returns `result_id`)
```bash
curl -X POST http://localhost:8000/analyze/short \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ada",
    "lang": "en",
    "gender": "female",
    "answers": [
      { "questionId": 1, "answer": "I like to plan ahead." }
    ]
  }'
```

Response (example):
```json
{
  "type": "short",
  "result_id": "3c0a2e8b-0aa4-4a1b-9ce6-2a5a8cf3c6b7",
  "result": {
    "runId": "3c0a2e8b-0aa4-4a1b-9ce6-2a5a8cf3c6b7",
    "animal": "turtle",
    "element": "water",
    "genderForm": "female",
    "text": "..."
  }
}
```

### 2) Generate a full result (by `result_id`)
```bash
curl -X POST http://localhost:8000/analyze/full \
  -H "Content-Type: application/json" \
  -d '{ "result_id": "3c0a2e8b-0aa4-4a1b-9ce6-2a5a8cf3c6b7" }'
```

Response (example):
```json
{
  "type": "full",
  "result_id": "3c0a2e8b-0aa4-4a1b-9ce6-2a5a8cf3c6b7",
  "result": {
    "animal": "turtle",
    "element": "water",
    "genderForm": "female",
    "text": "..."
  }
}
```

### Legacy full endpoint
If you still need to send `name/lang/answers` directly, use:
```bash
curl -X POST http://localhost:8000/analyze/full/legacy \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ada",
    "lang": "en",
    "gender": "female",
    "answers": [
      { "questionId": 1, "answer": "I like to plan ahead." }
    ]
  }'
```