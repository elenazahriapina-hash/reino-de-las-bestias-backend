from openai import OpenAI
import os
import json
import re
from typing import Dict

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60.0,
)

# допустимые значения

LANG_INSTRUCTIONS = {
    "ru": "Пиши весь текст СТРОГО на русском языке.",
    "en": "Write the entire response STRICTLY in English.",
    "es": "Escribe todo el texto ESTRICTAMENTE en español.",
    "pt": "Escreva todo o texto ESTRITAMENTE em português.",
}

ALLOWED_ANIMALS = {
    "Wolf",
    "Lion",
    "Tiger",
    "Lynx",
    "Panther",
    "Bear",
    "Fox",
    "Wolverine",
    "Deer",
    "Monkey",
    "Rabbit",
    "Buffalo",
    "Ram",
    "Capybara",
    "Elephant",
    "Horse",
    "Eagle",
    "Owl",
    "Raven",
    "Parrot",
    "Snake",
    "Crocodile",
    "Turtle",
    "Lizard",
}

ALLOWED_ELEMENTS = {"Воздух", "Вода", "Огонь", "Земля"}
ALLOWED_GENDERS = {"male", "female", "unspecified"}


def _extract_json(text: str) -> Dict:

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("JSON not found in model output")
        return json.loads(match.group())


def run_short_analysis(prompt: str, lang: str) -> dict:
    language_rule = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["ru"])

    system_instruction = """
    {language_rule}
    
Верни СТРОГО JSON.
Запрещено добавлять любые поля, кроме перечисленных.

Ты аналитическая модель системы «24 зверя × 4 стихии».

❗ Используй ТОЛЬКО утверждённые архетипы.
❗ НЕ используй метафорические или альтернативные названия.
❗ НЕ смешивай языки.
❗ НЕ добавляй текст вне JSON.

animal — один из:
Wolf, Lion, Tiger, Lynx, Panther, Bear, Fox, Wolverine, Deer,
Monkey, Rabbit, Buffalo, Ram, Capybara, Elephant, Horse,
Eagle, Owl, Raven, Parrot, Snake, Crocodile, Turtle, Lizard

element — строго одно из: Воздух | Вода | Огонь | Земля
genderForm — male | female | unspecified

Формат (СТРОГО):
{{
  "animal": "Wolf",
  "element": "Огонь",
  "genderForm": "male"
}}
""".strip()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=120,
    )

    raw_text = (response.output_text or "").strip()
    data = _extract_json(raw_text)

    animal = data.get("animal")
    element = data.get("element")
    gender_form = data.get("genderForm", "unspecified")

    # 🛡️ строгая валидация
    if animal not in ALLOWED_ANIMALS:
        raise ValueError(f"Invalid animal: {animal}")

    if element not in ALLOWED_ELEMENTS:
        raise ValueError(f"Invalid element: {element}")

    if gender_form not in ALLOWED_GENDERS:
        gender_form = "unspecified"

    return {
        "animal": animal,
        "element": element,
        "genderForm": gender_form,
    }


def generate_short_text(prompt: str, lang: str) -> str:

    language_rule = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["ru"])
    system_instruction = f"""
{language_rule}

Ты генерируешь КОРОТКИЙ результат по системе «24 зверя × 4 стихии».
Строго соблюдай структуру из промпта пользователя.
Не добавляй лишних блоков.
""".strip()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=520,
    )

    return (response.output_text or "").strip()


def run_full_analysis(prompt: str, lang: str) -> str:
    language_rule = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["ru"])

    system_instruction = f"""
{language_rule}

Ты формируешь ПОЛНЫЙ психологический профиль
в системе «24 зверя × 4 стихии».

❗ Архетип и стихия УЖЕ ЗАДАНЫ.
❗ НЕ изменяй архетип.
❗ НЕ добавляй новых животных.
❗ НЕ используй метафоры вместо названий.

СТРОГО соблюдай структуру полного профиля.
""".strip()

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": system_instruction,
            },
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=1200,  # достаточно для full-профиля
    )

    return (response.output_text or "").strip()
