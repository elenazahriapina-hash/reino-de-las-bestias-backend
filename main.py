import os
import uuid

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from sqlalchemy import text as sql_text
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ai import (
    ALLOWED_ANIMALS,
    ALLOWED_ELEMENTS,
    ALLOWED_GENDERS,
    run_short_analysis,
    generate_short_text,
    run_full_analysis,
)
from db import SessionLocal, engine
from models import Base, Run, RunAnswer, ShortResultORM, FullResultORM
from utils_animals import (
    build_image_key,
    get_animal_display_name,
    get_element_display_name,
    ELEMENT_LABELS,
)
from schemas import AnalyzeRequest, TestAnswer

app = FastAPI()


# -------------------- MODELS --------------------
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ShortResult(BaseModel):
    runId: str
    animal: str  # EN code
    element: str  # RU: Воздух/Вода/Огонь/Земля
    genderForm: str  # male/female/unspecified
    text: str


class ShortResponse(BaseModel):
    type: str
    result: ShortResult


class FullResult(BaseModel):
    animal: Optional[str] = None
    element: Optional[str] = None
    genderForm: Optional[str] = None
    text: str
    runId: Optional[str] = None


class FullResponse(BaseModel):
    type: str
    result: FullResult


class AnalyzeResult(BaseModel):
    animal: str
    element: str
    genderForm: str
    imageKey: str
    text: str


class AnalyzeResponse(BaseModel):
    type: str
    result: AnalyzeResult


# -------------------- PROMPT BUILDERS --------------------

SHORT_PROMPT_LABELS = {
    "ru": {
        "values_title": "Ценности",
        "conclusion_title": "Заключение",
        "point_1": "Пункт 1",
        "point_2": "Пункт 2",
    },
    "en": {
        "values_title": "Values",
        "conclusion_title": "Conclusion",
        "point_1": "Point 1",
        "point_2": "Point 2",
    },
    "es": {
        "values_title": "Valores",
        "conclusion_title": "Conclusión",
        "point_1": "Punto 1",
        "point_2": "Punto 2",
    },
    "pt": {
        "values_title": "Valores",
        "conclusion_title": "Conclusão",
        "point_1": "Ponto 1",
        "point_2": "Ponto 2",
    },
}

FULL_PROMPT_LABELS = {
    "ru": {
        "section_1": "Общий психопрофиль",
        "section_2": "Энергетический профиль",
        "section_3": "Стиль мышления",
        "section_4": "Социальное взаимодействие",
        "section_5": "Конфликтность и поведение в напряжённых ситуациях",
        "values_title": "Ценности",
        "section_7": "Профессиональный стиль",
        "section_8": "Сильные стороны",
        "section_9": "Потенциальные слабые стороны",
        "section_10": "Жизненный путь",
        "conclusion_title": "Итог",
    },
    "en": {
        "section_1": "General psychological profile",
        "section_2": "Energetic profile",
        "section_3": "Thinking style",
        "section_4": "Social interaction",
        "section_5": "Conflict and behavior under tension",
        "values_title": "Values",
        "section_7": "Professional style",
        "section_8": "Strengths",
        "section_9": "Potential weaknesses",
        "section_10": "Life path",
        "conclusion_title": "Conclusion",
    },
    "es": {
        "section_1": "Perfil psicológico general",
        "section_2": "Perfil energético",
        "section_3": "Estilo de pensamiento",
        "section_4": "Interacción social",
        "section_5": "Conflicto y comportamiento bajo tensión",
        "values_title": "Valores",
        "section_7": "Estilo profesional",
        "section_8": "Fortalezas",
        "section_9": "Debilidades potenciales",
        "section_10": "Camino de vida",
        "conclusion_title": "Conclusión",
    },
    "pt": {
        "section_1": "Perfil psicológico geral",
        "section_2": "Perfil energético",
        "section_3": "Estilo de pensamento",
        "section_4": "Interação social",
        "section_5": "Conflito e comportamento sob tensão",
        "values_title": "Valores",
        "section_7": "Estilo profissional",
        "section_8": "Pontos fortes",
        "section_9": "Fraquezas potenciais",
        "section_10": "Caminho de vida",
        "conclusion_title": "Conclusão",
    },
}


def build_answers_text(answers: List[TestAnswer]) -> str:
    return "\n".join(f"Q{a.questionId}: {a.answer}" for a in answers if a.answer)


def build_short_prompt(
    name: str,
    lang: str,
    gender: str,
    animal_display: str,
    element_display: str,
    answers_text: str,
) -> str:
    labels = SHORT_PROMPT_LABELS.get(lang, SHORT_PROMPT_LABELS["ru"])
    return f"""
❗ ВАЖНО:
Используй ТОЛЬКО ЭТО животное:
{animal_display}

❌ Запрещено:
– заменять животное
– использовать других птиц или зверей
– вводить новые образы

❗ ЯЗЫК (ОБЯЗАТЕЛЬНО)
Пиши ВЕСЬ текст СТРОГО на языке: {lang}

Если язык:
ru — русский  
en — английский  
es — испанский  
pt — португальский  

Запрещено:
– смешивать языки
– использовать русский, если lang ≠ ru
– добавлять перевод в скобках
Даже если они кажутся более подходящими.
Ты - аналитическая ИИ-модель, определяющая архетип зверя (строго из списка 24) и стихию (Огонь, Вода, Воздух, Земля) на основе ответов пользователя.
Твоя задача — выдать короткий психологический профиль,
сохраняя все правила системы, выводя только ключевые блоки, включая итоговое заключение,
в форме, удобной и естественной именно для данного пользователя.

1️⃣ ЛОГИКА УЧЁТА ПОЛА
Пол НЕ влияет на анализ.
Пол влияет ТОЛЬКО на форму названия архетипа.
Если пол не указан — используй мужскую (нейтральную) форму.
Пол: {gender}

2️⃣ АЛГОРИТМ АНАЛИЗА (ВНУТРЕННИЙ)
Проанализируй ответы пользователя по 10 осям.
Сравни модель пользователя с критериями всех 24 зверей.
❗ Не описывай алгоритм и не упоминай оси.

3️⃣ ОБЯЗАТЕЛЬНЫЕ БЛОКИ
В финальном выводе должны быть:
– Архетип (животное + стихия)
– Краткое общее описание
– {labels["values_title"]}
– Два наиболее ярких пункта личности
– {labels["conclusion_title"]}


4️⃣ ОТЗЕРКАЛИВАНИЕ СТИЛЯ
Текст должен читаться как «про меня».

5️⃣ СТРОГАЯ СТРУКТУРА (НЕ МЕНЯТЬ)

{name} — {animal_display} {element_display} {{ЗНАЧОК}}
{{Короткая строка-образ. 3–7 слов.}}

{{Краткое общее описание — 1–2 абзаца}}

🧭 {labels["values_title"]} — «{{3–4 ключевых слова}}»
• …
• …
• …
• …

{{{{{labels["point_1"]} — самый яркий}}}}
{{ЗНАЧОК}} {{Название пункта}} — «{{Метафорическое название}}»
{{Короткое описание}}

{{{{{labels["point_2"]} — второй по яркости}}}}
{{ЗНАЧОК}} {{Название пункта}} — «{{Метафорическое название}}»
{{Короткое описание}}

🧩 {labels["conclusion_title"]}
{{Интегральный вывод}}

6️⃣ СТИЛЬ
Тон: взрослый, спокойный, уверенный.
Запрещено: «возможно», «кажется», эзотерика, объяснение анализа.

Имя пользователя: {name}
Язык: {lang}

Ответы пользователя:
{answers_text}
""".strip()


def normalize_locked_element(locked_element: str, lang: str) -> Optional[str]:
    if locked_element in ALLOWED_ELEMENTS:
        return locked_element
    for label_map in (ELEMENT_LABELS.get(lang, {}), ELEMENT_LABELS["ru"]):
        for element_code, label in label_map.items():
            if locked_element == label:
                return element_code
    for label_map in ELEMENT_LABELS.values():
        for element_code, label in label_map.items():
            if locked_element == label:
                return element_code
    return None


def resolve_locked_codes(
    locked_animal: Optional[str],
    locked_element: Optional[str],
    locked_gender_form: Optional[str],
    lang: str,
) -> Optional[dict[str, str]]:
    if not (locked_animal and locked_element and locked_gender_form):
        return None
    if locked_animal not in ALLOWED_ANIMALS:
        raise HTTPException(status_code=400, detail="Invalid lockedAnimal")
    normalized_element = normalize_locked_element(locked_element, lang)
    if normalized_element not in ALLOWED_ELEMENTS:
        raise HTTPException(status_code=400, detail="Invalid lockedElement")
    if locked_gender_form not in ALLOWED_GENDERS:
        raise HTTPException(status_code=400, detail="Invalid lockedGenderForm")
    return {
        "animal": locked_animal,
        "element": normalized_element,
        "genderForm": locked_gender_form,
    }


def build_full_prompt(
    name: str,
    lang: str,
    gender: Optional[str],
    animal_display: str,
    element_label: str,
    element_display: str,
    answers_text: str,
) -> str:
    labels = FULL_PROMPT_LABELS.get(lang, FULL_PROMPT_LABELS["ru"])

    return f"""
Ты — аналитическая ИИ-модель, формирующая полный психологический профиль личности
на основе заданного архетипа зверя, заданной стихии и ответов пользователя
в системе «24 зверя × 4 стихии».

Архетип зверя и стихия ЗАДАНЫ и НЕ ПЕРЕСМАТРИВАЮТСЯ.

Архетип: {animal_display}
Стихия: {element_label}
Пол: {gender}

1️⃣ СИСТЕМА И ГРАНИЦЫ
Система включает:
матрицу 24 архетипов зверей;
4 стихии: Огонь, Вода, Воздух, Земля;
10 внутренних аналитических осей.
Пол:
НЕ влияет на анализ;
влияет ТОЛЬКО на форму названия архетипа.
Используй СТРОГО утверждённые формы архетипов
(список форм — без изменений).
Если пол не указан — используй мужскую (нейтральную) форму.
2️⃣ АЛГОРИТМ АНАЛИЗА (ВНУТРЕННИЙ)
Архетип зверя и стихия заданы.
Проанализируй ответы пользователя по 10 внутренним осям:
темп
энергия
конфликтность
социальность
стиль мышления
стиль действий
стресс-реакция
вектор энергии
ориентация
функция архетипа
На основе анализа:
раскрой проявление стихии внутри данного архетипа;
сформируй целостный психологический портрет;
не изменяй входные параметры.
❗️
Не упоминай оси.
Не описывай механику.
3️⃣ ПОДАЧА С ОТЗЕРКАЛИВАНИЕМ (КРИТИЧЕСКИ ВАЖНО)
Стиль подачи обязан учитывать:
архетип зверя;
стихию;
темп и характер ответов пользователя.
Правило отзеркаливания
Текст должен быть написан в ритме, интонации и плотности,
которые комфортны именно этому архетипу и этому человеку.
Примеры (внутренние, не упоминать в ответе):
для Земли → спокойный, устойчивый, размеренный, без резких формулировок;
для Воздуха → ясный, структурный, лёгкий, логичный;
для Воды → тёплый, эмпатичный, поддерживающий;
для Огня → прямой, собранный, энергичный, уверенный.
Если ответы пользователя:
осторожные → подача мягче;
прямые → подача прямее;
рефлексивные → глубже;
лаконичные → без избыточных украшений.
❗️
Отзеркаливание НЕ должно:
искажать смысл;
упрощать глубину;
менять структуру.
Цель — чтобы текст читался как «про меня и моим языком».
4️⃣ СТИЛЬ И ЗАПРЕТЫ
Общий стиль:
взрослый
спокойный
уверенный
человеческий
Запрещено:
«возможно», «кажется», «вероятно»;
эзотерика и мистика;
диагнозы;
объяснение механики работы модели.
5️⃣ ЭМОДЗИ
Используй эмодзи:
одного визуального стиля;
одного масштаба;
строго по разделам (как в эталоне).
6️⃣ СТРОГАЯ СТРУКТУРА ВЫВОДА (НЕ МЕНЯТЬ):
{name} — {{Архетип (с учётом пола)}} {element_display}
(краткое описание архетипа в скобках)
1. {labels["section_1"]}
2. {labels["section_2"]}
3. {labels["section_3"]}
4. {labels["section_4"]}
5. {labels["section_5"]}
6. {labels["values_title"]}
7. {labels["section_7"]}
8. {labels["section_8"]}
9. {labels["section_9"]}
10. {labels["section_10"]}
{labels["conclusion_title"]}
7️⃣ КЛЮЧЕВОЕ ПРАВИЛО
Ты не просто описываешь архетип.
Ты говоришь с человеком на его языке.
Язык: {lang}
Ответы пользователя:
{answers_text}
""".strip()


# -------------------- ENDPOINT --------------------


def normalize_answers(answers: List[TestAnswer]) -> List[TestAnswer]:
    return answers


@app.post("/analyze/short", response_model=ShortResponse)
async def analyze_short(payload: AnalyzeRequest):
    try:
        print("📥 SHORT payload:", payload)
        print(
            "✅ SHORT parsed:",
            f"lang={payload.lang}, gender={payload.gender}, answers={len(payload.answers)}",
        )

        normalized_answers = normalize_answers(payload.answers)
        answers_text = build_answers_text(normalized_answers)

        codes = resolve_locked_codes(
            payload.lockedAnimal,
            payload.lockedElement,
            payload.lockedGenderForm,
            payload.lang,
        )
        if codes is None:
            codes = run_short_analysis(
                prompt=f"""
Имя: {payload.name}
Язык: {payload.lang}
Пол: {payload.gender or "unspecified"}

Ответы пользователя:
{answers_text}
""".strip(),
                lang=payload.lang,
            )

        animal_display = get_animal_display_name(
            animal_code=codes["animal"],
            lang=payload.lang,
            gender=codes["genderForm"],
        )
        element_display = get_element_display_name(
            element_code=codes["element"],
            lang=payload.lang,
            ru_case="genitive_for_archetype_line",
        )

        # 3) short text
        text_prompt = build_short_prompt(
            name=payload.name,
            lang=payload.lang,
            gender=payload.gender or "unspecified",
            animal_display=animal_display,
            element_display=element_display,
            answers_text=answers_text,
        )
        text = generate_short_text(text_prompt, payload.lang)
        run_id = uuid.uuid4()
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    name=payload.name,
                    lang=payload.lang,
                    gender=payload.gender or "unspecified",
                )
            )
            session.add_all(
                [
                    RunAnswer(
                        run_id=run_id,
                        question_id=answer.questionId,
                        answer=answer.answer,
                    )
                    for answer in normalized_answers
                ]
            )
            session.add(
                ShortResultORM(
                    run_id=run_id,
                    animal=codes["animal"],
                    element=codes["element"],
                    gender_form=codes["genderForm"],
                    text=text,
                )
            )
            await session.commit()

        return {
            "type": "short",
            "result": {
                "runId": str(run_id),
                "animal": codes["animal"],
                "element": codes["element"],  # ✅ RU: Огонь/Вода/Воздух/Земля
                "genderForm": codes["genderForm"],
                "text": text,
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        print("❌ SHORT ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result/short/{runId}", response_model=ShortResponse)
async def get_short_result(runId: str):
    try:
        run_uuid = uuid.UUID(runId)
    except ValueError:
        raise HTTPException(status_code=404, detail="Short result not found")

    async with SessionLocal() as session:
        result = await session.get(ShortResultORM, run_uuid)

    if result is None:
        raise HTTPException(status_code=404, detail="Short result not found")

    return {
        "type": "short",
        "result": {
            "runId": str(result.run_id),
            "animal": result.animal,
            "element": result.element,
            "genderForm": result.gender_form,
            "text": result.text,
        },
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    try:
        print("📥 ANALYZE payload:", payload)

        answers_text = build_answers_text(payload.answers)

        codes = run_short_analysis(
            prompt=f"""
Имя: {payload.name}
Язык: {payload.lang}
Пол: {payload.gender or "unspecified"}

Ответы пользователя:
{answers_text}
""".strip(),
            lang=payload.lang,
        )

        animal_display = get_animal_display_name(
            animal_code=codes["animal"],
            lang=payload.lang,
            gender=codes["genderForm"],
        )
        element_display = get_element_display_name(
            element_code=codes["element"],
            lang=payload.lang,
            ru_case="genitive_for_archetype_line",
        )

        text_prompt = build_short_prompt(
            name=payload.name,
            lang=payload.lang,
            gender=payload.gender or "unspecified",
            animal_display=animal_display,
            element_display=element_display,
            answers_text=answers_text,
        )
        text = generate_short_text(text_prompt, payload.lang)
        image_key = build_image_key(
            animal_code=codes["animal"],
            element=codes["element"],
            gender=codes["genderForm"],
        )

        return {
            "type": "short",
            "result": {
                "animal": codes["animal"],
                "element": codes["element"],
                "genderForm": codes["genderForm"],
                "imageKey": image_key,
                "text": text,
            },
        }

    except Exception as e:
        print("❌ ANALYZE ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/full", response_model=FullResponse)
async def analyze_full(payload: AnalyzeRequest):
    try:
        print("📥 FULL payload:", payload)
        print(
            "✅ FULL parsed:",
            f"lang={payload.lang}, gender={payload.gender}, answers={len(payload.answers)}",
        )

        normalized_answers = normalize_answers(payload.answers)
        answers_text = build_answers_text(normalized_answers)

        locked_codes = resolve_locked_codes(
            payload.lockedAnimal,
            payload.lockedElement,
            payload.lockedGenderForm,
            payload.lang,
        )
        if locked_codes is None:
            locked_codes = run_short_analysis(
                prompt=f"""
Имя: {payload.name}
Язык: {payload.lang}
Пол: {payload.gender or "unspecified"}

Ответы пользователя:
{answers_text}
""".strip(),
                lang=payload.lang,
            )
        animal_code = locked_codes["animal"]
        element_code = locked_codes["element"]
        gender_form = locked_codes["genderForm"]

        animal_display = get_animal_display_name(
            animal_code=animal_code,
            lang=payload.lang,
            gender=gender_form,
        )
        element_label = get_element_display_name(
            element_code=element_code,
            lang=payload.lang,
        )
        element_display = get_element_display_name(
            element_code=element_code,
            lang=payload.lang,
            ru_case="genitive_for_archetype_line",
        )

        prompt = build_full_prompt(
            name=payload.name,
            lang=payload.lang,
            gender=gender_form,
            animal_display=animal_display,
            element_label=element_label,
            element_display=element_display,
            answers_text=answers_text,
        )

        text = run_full_analysis(prompt, payload.lang)

        run_id = uuid.uuid4()
        async with SessionLocal() as session:
            session.add(
                Run(
                    id=run_id,
                    name=payload.name,
                    lang=payload.lang,
                    gender=payload.gender or "unspecified",
                )
            )
            session.add_all(
                [
                    RunAnswer(
                        run_id=run_id,
                        question_id=answer.questionId,
                        answer=answer.answer,
                    )
                    for answer in normalized_answers
                ]
            )
            session.add(
                FullResultORM(
                    run_id=run_id,
                    text=text,
                )
            )
            await session.commit()

        return {
            "type": "full",
            "result": {
                "runId": str(run_id),
                "animal": animal_code,
                "element": element_code,
                "genderForm": gender_form,
                "text": text,
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        print("❌ FULL ANALYSIS ERROR:", e)
        raise HTTPException(status_code=500, detail="Ошибка анализа")


@app.get("/result/full/{runId}", response_model=FullResponse)
async def get_full_result(runId: str):
    try:
        run_uuid = uuid.UUID(runId)
    except ValueError:
        raise HTTPException(status_code=404, detail="Full result not found")

    async with SessionLocal() as session:
        result = await session.get(FullResultORM, run_uuid)

    if result is None:
        raise HTTPException(status_code=404, detail="Full result not found")

    return (
        {
            "type": "full",
            "result": {
                "runId": str(result.run_id),
                "animal": None,
                "element": None,
                "genderForm": None,
                "text": result.text,
            },
        },
    )


@app.get("/health/db")
async def health_db():
    try:
        async with SessionLocal() as session:
            await session.execute(sql_text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
