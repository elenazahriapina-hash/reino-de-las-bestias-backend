from openai import OpenAI
import os
import json
import re

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

COMPAT_PROMPT_VERSION = "v3"

COMPATIBILITY_PROMPT_V3 = """
You are generating a compatibility report for the “24 animals × 4 elements” system.

STRICT RULES:
1) Output ONLY the final report text. No JSON, no preface, no analysis, no prompt echoing.
2) Use the language specified by the `LANGUAGE:` tag in the user payload (ru/en/es/pt).
3) Use the names, animals, and elements exactly as provided in the payload.
4) The first two lines must be exactly:
   🟢 {nameA} — {animalA} {elementA}
   🔴 {nameB} — {animalB} {elementB}
5) Then output the following numbered section headings in the selected language and provide the content for each section.

SECTION HEADINGS BY LANGUAGE:
ru:
1) Основное сходство
2) Ключевые различия
3) Сильные стороны
4) Возможные сложности
5) Рекомендации
6) Итог

en:
1) Key similarities
2) Key differences
3) Strengths
4) Potential challenges
5) Recommendations
6) Summary

es:
1) Similitudes
2) Diferencias clave
3) Fortalezas
4) Dificultades
5) Recomendaciones
6) Resumen

pt:
1) Semelhanças
2) Diferenças-chave
3) Pontos fortes
4) Desafios
5) Recomendações
6) Resumo

Keep each section concise and focused on the provided data.
""".strip()


def _extract_json(text: str) -> dict:

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


def generate_compatibility_text(system_prompt: str, user_payload: str) -> str:
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        max_output_tokens=1200,
    )

    return (response.output_text or "").strip()
