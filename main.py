import os
import uuid

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from sqlalchemy import delete, select, text as sql_text
from sqlalchemy import or_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai import (
    ALLOWED_ANIMALS,
    ALLOWED_ELEMENTS,
    ALLOWED_GENDERS,
    COMPAT_PROMPT_VERSION,
    COMPAT_SYSTEM_PROMPT,
    run_short_analysis,
    generate_short_text,
    run_full_analysis,
    generate_compatibility_text,
)
from db import SessionLocal, engine
from models import (
    Base,
    Run,
    RunAnswer,
    ShortResultORM,
    FullResultORM,
    User,
    UserResult,
    CompatReport,
    Invite,
)
from utils_animals import (
    build_image_key,
    get_animal_display_name,
    get_element_display_name,
    ELEMENT_LABELS,
    get_animal_ru_name,
)
from schemas import (
    AnalyzeRequest,
    CompatibilityAcceptInviteRequest,
    CompatibilityCheckRequest,
    CompatibilityInviteRequest,
    CompatibilityInviteResponse,
    CompatibilityListResponse,
    CompatibilityReportResponse,
    DevSeedUserRequest,
    DevSeedUserResponse,
    FullRequest,
    FullResponse,
    LookupUserResponse,
    RegisterRequest,
    RegisterResponse,
    ShortResponse,
    TestAnswer,
    UserMeResponse,
    UserResponse,
)

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


def build_answers_text(answers: list[TestAnswer]) -> str:
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


def normalize_locked_element(locked_element: str, lang: str) -> str | None:
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
    locked_animal: str | None,
    locked_element: str | None,
    locked_gender_form: str | None,
    lang: str,
) -> dict[str, str] | None:
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
    gender: str | None,
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
Use ONLY this animal: {animal_display}
Use ONLY this element: {element_label}

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
def normalize_answers(answers: list[TestAnswer]) -> list[TestAnswer]:
    return answers


LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
}


def extract_auth_token(
    authorization: str | None, x_auth_token: str | None
) -> str | None:
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    if x_auth_token:
        return x_auth_token
    return None


async def get_current_user(
    session,
    *,
    authorization: str | None,
    x_auth_token: str | None,
) -> User:
    token = extract_auth_token(authorization, x_auth_token)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    user = await session.scalar(select(User).where(User.auth_token == token))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    return user


def build_compatibility_payload(
    *,
    lang: str,
    a_result: UserResult,
    b_result: UserResult,
) -> str:
    language_name = LANGUAGE_NAMES.get(lang, "Russian")
    a_animal_ru = get_animal_ru_name(a_result.animal_code, a_result.genderForm)
    b_animal_ru = get_animal_ru_name(b_result.animal_code, b_result.genderForm)
    a_full = a_result.full_text or "NOT_PROVIDED"
    b_full = b_result.full_text or "NOT_PROVIDED"

    return f"""
Output language: {language_name} (langCode={lang})

Person A
Archetype: {a_animal_ru} {a_result.element_ru}
SHORT:
<<<SHORT_A
{a_result.short_text}
SHORT_A>>>
FULL:
<<<FULL_A
{a_full}
FULL_A>>>

Person B
Archetype: {b_animal_ru} {b_result.element_ru}
SHORT:
<<<SHORT_B
{b_result.short_text}
SHORT_B>>>
FULL:
<<<FULL_B
{b_full}
FULL_B>>>

Rules:
- If NOT_PROVIDED → do not invent facts
- Analysis must be universal
- Do not mention emails, telegrams, or identities
""".strip()


def is_dev_seed_enabled() -> bool:
    return os.getenv("DEV_SEED_ENABLED", "").lower() == "true"


def serialize_report(
    report: CompatReport, current_user_id: int
) -> CompatibilityReportResponse:
    other_user_id = (
        report.user_high_id
        if report.user_low_id == current_user_id
        else report.user_low_id
    )
    return CompatibilityReportResponse(
        id=report.id,
        other_user_id=other_user_id,
        prompt_version=report.prompt_version,
        status=report.status,
        text=report.text,
        created_at=report.created_at,
    )


async def ensure_run_and_answers(
    session,
    *,
    run_id: uuid.UUID,
    payload: AnalyzeRequest,
    normalized_answers: list[TestAnswer],
) -> str:
    existing_run = await session.get(Run, run_id)
    if existing_run is None:
        session.add(
            Run(
                id=run_id,
                name=payload.name,
                lang=payload.lang,
                gender=payload.gender or "unspecified",
            )
        )
        await session.flush()
        print(f"🆕 DB: created run {run_id}")
    else:
        print(f"🔁 DB: reused run {run_id}")

    await session.execute(delete(RunAnswer).where(RunAnswer.run_id == run_id))
    answers_to_save = [
        RunAnswer(
            run_id=run_id,
            question_id=answer.questionId,
            answer=answer.answer,
        )
        for answer in normalized_answers
    ]
    session.add_all(answers_to_save)
    print(f"💾 DB: saved answers count={len(answers_to_save)}")
    return str(run_id)


async def upsert_short_result(
    session,
    *,
    run_id: uuid.UUID,
    animal: str,
    element: str,
    gender_form: str,
    text: str,
) -> None:
    stmt = (
        insert(ShortResultORM)
        .values(
            run_id=run_id,
            animal=animal,
            element=element,
            gender_form=gender_form,
            text=text,
        )
        .on_conflict_do_update(
            index_elements=[ShortResultORM.run_id],
            set_={
                "animal": animal,
                "element": element,
                "gender_form": gender_form,
                "text": text,
            },
        )
    )
    await session.execute(stmt)


@app.post("/analyze/short", response_model=ShortResponse)
async def analyze_short(
    payload: AnalyzeRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    try:
        requested_run_id = uuid.UUID(payload.runId) if payload.runId else uuid.uuid4()
        print("📥 SHORT payload:", payload)
        print(
            "✅ SHORT parsed:",
            f"lang={payload.lang}, gender={payload.gender}, answers={len(payload.answers)}, run_id={requested_run_id}",
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
        async with SessionLocal() as session:
            async with session.begin():
                run_id = uuid.UUID(
                    await ensure_run_and_answers(
                        session,
                        run_id=requested_run_id,
                        payload=payload,
                        normalized_answers=normalized_answers,
                    )
                )
                await upsert_short_result(
                    session,
                    run_id=run_id,
                    animal=codes["animal"],
                    element=codes["element"],
                    gender_form=codes["genderForm"],
                    text=text,
                )
                token = extract_auth_token(authorization, x_auth_token)
                if token:
                    user = await session.scalar(
                        select(User).where(User.auth_token == token)
                    )
                    if user:
                        existing_result = await session.get(UserResult, user.id)
                        if existing_result:
                            existing_result.animal_code = codes["animal"]
                            existing_result.element_ru = codes["element"]
                            existing_result.genderForm = codes["genderForm"]
                            existing_result.short_text = text
                        else:
                            session.add(
                                UserResult(
                                    user_id=user.id,
                                    animal_code=codes["animal"],
                                    element_ru=codes["element"],
                                    genderForm=codes["genderForm"],
                                    short_text=text,
                                )
                            )
            saved = await session.get(ShortResultORM, run_id)
            if saved is None:
                raise HTTPException(
                    status_code=500, detail="short_result not persisted"
                )
            print(
                "✅ DB verify short_result ok",
                f"run_id={run_id}",
                f"animal={saved.animal}",
                f"element={saved.element}",
                f"gender_form={saved.gender_form}",
            )
        response = {
            "type": "short",
            "result_id": str(run_id),
            "result": {
                "animal": codes["animal"],
                "element": codes["element"],  # ✅ RU: Огонь/Вода/Воздух/Земля
                "genderForm": codes["genderForm"],
                "text": text,
            },
        }
        print(
            "📤 SHORT response keys:",
            list(response.keys()),
            "result keys:",
            list(response["result"].keys()),
        )
        return response

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
        "result_id": str(result.run_id),
        "result": {
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


async def analyze_full_legacy(payload: AnalyzeRequest) -> dict:
    try:
        requested_run_id = uuid.UUID(payload.runId) if payload.runId else uuid.uuid4()
        print("📥 FULL payload:", payload)
        print(
            "✅ FULL parsed:",
            f"lang={payload.lang}, gender={payload.gender}, answers={len(payload.answers)}, run_id={requested_run_id}",
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

        async with SessionLocal() as session:
            async with session.begin():
                run_id = uuid.UUID(
                    await ensure_run_and_answers(
                        session,
                        run_id=requested_run_id,
                        payload=payload,
                        normalized_answers=normalized_answers,
                    )
                )
                session.add(
                    FullResultORM(
                        run_id=run_id,
                        text=text,
                    )
                )
            print(f"💾 DB: saved full_result run_id={run_id}")

        return {
            "type": "full",
            "result_id": str(run_id),
            "result": {
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


@app.post("/analyze/full", response_model=FullResponse)
async def analyze_full(
    payload: FullRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    try:
        print("📥 FULL payload result_id:", payload.result_id)
        try:
            run_uuid = uuid.UUID(payload.result_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid result_id format"
            ) from exc

        async with SessionLocal() as session:
            async with session.begin():
                run = await session.get(Run, run_uuid)
                short_result = await session.get(ShortResultORM, run_uuid)
                answers_query = await session.execute(
                    select(RunAnswer)
                    .where(RunAnswer.run_id == run_uuid)
                    .order_by(RunAnswer.question_id)
                )
                answers_rows = answers_query.scalars().all()
                print(
                    "🔎 FULL lookup:",
                    f"run_found={run is not None}",
                    f"short_result_found={short_result is not None}",
                    f"answers_count={len(answers_rows)}",
                )

                if run is None:
                    raise HTTPException(status_code=404, detail="Run not found")
                if short_result is None:
                    raise HTTPException(
                        status_code=500,
                        detail="short_result missing for existing run",
                    )

                answers = [
                    TestAnswer(questionId=row.question_id, answer=row.answer)
                    for row in answers_rows
                ]
                answers_text = build_answers_text(answers)

                animal_code = short_result.animal
                element_code = short_result.element
                gender_form = short_result.gender_form

                animal_display = get_animal_display_name(
                    animal_code=animal_code,
                    lang=run.lang,
                    gender=gender_form,
                )
                element_label = get_element_display_name(
                    element_code=element_code,
                    lang=run.lang,
                )
                element_display = get_element_display_name(
                    element_code=element_code,
                    lang=run.lang,
                    ru_case="genitive_for_archetype_line",
                )

                prompt = build_full_prompt(
                    name=run.name,
                    lang=run.lang,
                    gender=gender_form,
                    animal_display=animal_display,
                    element_label=element_label,
                    element_display=element_display,
                    answers_text=answers_text,
                )
                text = run_full_analysis(prompt, run.lang)

                print("FULL tx active before save:", session.in_transaction())

                full_stmt = (
                    insert(FullResultORM)
                    .values(run_id=run_uuid, text=text)
                    .on_conflict_do_update(
                        index_elements=[FullResultORM.run_id],
                        set_={"text": text},
                    )
                )
                await session.execute(full_stmt)
                saved_full = await session.get(FullResultORM, run_uuid)
                print("FULL verify saved:", saved_full is not None)
                print(f"💾 DB: saved full_result run_id={run_uuid}")
                token = extract_auth_token(authorization, x_auth_token)
                if token:
                    user = await session.scalar(
                        select(User).where(User.auth_token == token)
                    )
                    if user:
                        existing_result = await session.get(UserResult, user.id)
                        if existing_result:
                            existing_result.animal_code = animal_code
                            existing_result.element_ru = element_code
                            existing_result.genderForm = gender_form
                            existing_result.short_text = short_result.text
                            existing_result.full_text = text
                        else:
                            session.add(
                                UserResult(
                                    user_id=user.id,
                                    animal_code=animal_code,
                                    element_ru=element_code,
                                    genderForm=gender_form,
                                    short_text=short_result.text,
                                    full_text=text,
                                )
                            )

        return {
            "type": "full",
            "result_id": str(run_uuid),
            "result": {
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


@app.post("/analyze/full/legacy", response_model=FullResponse)
async def analyze_full_legacy_endpoint(payload: AnalyzeRequest):
    return await analyze_full_legacy(payload)


@app.post("/auth/register", response_model=RegisterResponse)
async def register(payload: RegisterRequest):
    if not payload.email and not payload.telegram:
        raise HTTPException(status_code=400, detail="Email or telegram required")
    conditions = []
    if payload.email:
        conditions.append(User.email == payload.email)
    if payload.telegram:
        conditions.append(User.telegram == payload.telegram)
    if not conditions:
        raise HTTPException(status_code=400, detail="Email or telegram required")
    async with SessionLocal() as session:
        existing_users = (
            (await session.execute(select(User).where(or_(*conditions))))
            .scalars()
            .all()
        )
        if len(existing_users) > 1:
            raise HTTPException(
                status_code=409, detail="Email and telegram belong to different users"
            )
        if existing_users:
            user = existing_users[0]
            if payload.name:
                user.name = payload.name
            if payload.lang:
                user.lang = payload.lang
            if payload.shortResult:
                existing_result = await session.get(UserResult, user.id)
                if existing_result:
                    existing_result.animal_code = payload.shortResult.animal
                    existing_result.element_ru = payload.shortResult.element
                    existing_result.genderForm = payload.shortResult.genderForm
                    existing_result.short_text = payload.shortResult.text
                else:
                    session.add(
                        UserResult(
                            user_id=user.id,
                            animal_code=payload.shortResult.animal,
                            element_ru=payload.shortResult.element,
                            genderForm=payload.shortResult.genderForm,
                            short_text=payload.shortResult.text,
                        )
                    )
            await session.commit()
            await session.refresh(user)
        else:
            if not payload.name or not payload.lang:
                raise HTTPException(status_code=400, detail="Name and lang required")
            auth_token = uuid.uuid4().hex
            user = User(
                email=payload.email,
                telegram=payload.telegram,
                name=payload.name,
                lang=payload.lang,
                auth_token=auth_token,
                has_full=False,
                packs_bought=0,
                compat_credits=1,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            if payload.shortResult:
                session.add(
                    UserResult(
                        user_id=user.id,
                        animal_code=payload.shortResult.animal,
                        element_ru=payload.shortResult.element,
                        genderForm=payload.shortResult.genderForm,
                        short_text=payload.shortResult.text,
                    )
                )
                await session.commit()

    return RegisterResponse(
        userId=user.id,
        token=user.auth_token,
        credits=user.compat_credits,
        hasFull=user.has_full,
    )


@app.post("/dev/seed_user", response_model=DevSeedUserResponse)
async def dev_seed_user(payload: DevSeedUserRequest):
    if not is_dev_seed_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    if not payload.email and not payload.telegram:
        raise HTTPException(status_code=400, detail="Email or telegram required")
    conditions = []
    if payload.email:
        conditions.append(User.email == payload.email)
    if payload.telegram:
        conditions.append(User.telegram == payload.telegram)
    async with SessionLocal() as session:
        existing_users = (
            (await session.execute(select(User).where(or_(*conditions))))
            .scalars()
            .all()
        )
        if len(existing_users) > 1:
            raise HTTPException(
                status_code=409, detail="Email and telegram belong to different users"
            )
        if existing_users:
            user = existing_users[0]
            return DevSeedUserResponse(userId=user.id, token=user.auth_token)

        auth_token = uuid.uuid4().hex
        user = User(
            email=payload.email,
            telegram=payload.telegram,
            name=payload.name,
            lang=payload.lang,
            auth_token=auth_token,
            has_full=False,
            packs_bought=0,
            compat_credits=1,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        session.add(
            UserResult(
                user_id=user.id,
                animal_code=payload.animal,
                element_ru=payload.element,
                genderForm=payload.genderForm,
                short_text=payload.short_text,
            )
        )
        await session.commit()

    return DevSeedUserResponse(userId=user.id, token=user.auth_token)


@app.post("/compatibility/register", response_model=RegisterResponse)
async def compatibility_register(payload: RegisterRequest):
    return await register(payload)


@app.get("/users/me", response_model=UserMeResponse)
async def get_me(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    async with SessionLocal() as session:
        user = await get_current_user(
            session, authorization=authorization, x_auth_token=x_auth_token
        )
        return UserMeResponse(
            credits=user.compat_credits,
            hasFull=user.has_full,
            userId=user.id,
            lang=user.lang,
        )


@app.get("/compatibility/me", response_model=UserMeResponse)
async def compatibility_me(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    async with SessionLocal() as session:
        user = await get_current_user(
            session, authorization=authorization, x_auth_token=x_auth_token
        )
        return UserMeResponse(
            credits=user.compat_credits,
            hasFull=user.has_full,
            userId=user.id,
            lang=user.lang,
        )


@app.get("/users/lookup", response_model=LookupUserResponse)
async def lookup_user(
    q: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    async with SessionLocal() as session:
        await get_current_user(
            session, authorization=authorization, x_auth_token=x_auth_token
        )
        user = await session.scalar(
            select(User).where((User.email == q) | (User.telegram == q))
        )
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return LookupUserResponse(id=user.id, name=user.name, lang=user.lang)


@app.post("/purchase/full", response_model=UserResponse)
async def purchase_full(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    async with SessionLocal() as session:
        async with session.begin():
            user = await get_current_user(
                session, authorization=authorization, x_auth_token=x_auth_token
            )
            if not user.has_full:
                user.has_full = True
                user.compat_credits += 3
        await session.refresh(user)
        return UserResponse(
            id=user.id,
            email=user.email,
            telegram=user.telegram,
            name=user.name,
            lang=user.lang,
            has_full=user.has_full,
            packs_bought=user.packs_bought,
            compat_credits=user.compat_credits,
            created_at=user.created_at,
        )


@app.post("/purchase/compat_pack", response_model=UserResponse)
async def purchase_compat_pack(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    async with SessionLocal() as session:
        async with session.begin():
            user = await get_current_user(
                session, authorization=authorization, x_auth_token=x_auth_token
            )
            user.packs_bought += 1
            user.compat_credits += 3
        await session.refresh(user)
        return UserResponse(
            id=user.id,
            email=user.email,
            telegram=user.telegram,
            name=user.name,
            lang=user.lang,
            has_full=user.has_full,
            packs_bought=user.packs_bought,
            compat_credits=user.compat_credits,
            created_at=user.created_at,
        )


@app.post("/compatibility/check", response_model=CompatibilityReportResponse)
async def compatibility_check(
    payload: CompatibilityCheckRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    async with SessionLocal() as session:
        user = await get_current_user(
            session, authorization=authorization, x_auth_token=x_auth_token
        )
        if payload.requestId:
            existing = await session.scalar(
                select(CompatReport).where(
                    CompatReport.request_id == payload.requestId,
                    (CompatReport.user_low_id == user.id)
                    | (CompatReport.user_high_id == user.id),
                )
            )
            if existing:
                return serialize_report(existing, user.id)
        target = await session.get(User, payload.target_user_id)
        if not target:
            raise HTTPException(status_code=404, detail="Target user not found")
        if target.id == user.id:
            raise HTTPException(status_code=400, detail="Cannot compare same user")

        a_result = await session.get(UserResult, user.id)
        b_result = await session.get(UserResult, target.id)
        if not a_result:
            raise HTTPException(status_code=400, detail="Complete test first")
        if not b_result:
            raise HTTPException(
                status_code=400, detail="Target user must complete test first"
            )

        user_low_id, user_high_id = sorted([user.id, target.id])
        existing = await session.scalar(
            select(CompatReport).where(
                CompatReport.user_low_id == user_low_id,
                CompatReport.user_high_id == user_high_id,
                CompatReport.prompt_version == COMPAT_PROMPT_VERSION,
            )
        )
        if existing:
            return serialize_report(existing, user.id)

        if user.compat_credits < 1:
            raise HTTPException(status_code=402, detail="Not enough credits")

        report = CompatReport(
            user_low_id=user_low_id,
            user_high_id=user_high_id,
            prompt_version=COMPAT_PROMPT_VERSION,
            status="pending",
            text="",
            request_id=payload.requestId,
        )

        try:
            async with session.begin():
                user.compat_credits -= 1
                session.add(report)
        except IntegrityError:
            await session.rollback()
            existing = await session.scalar(
                select(CompatReport).where(
                    CompatReport.user_low_id == user_low_id,
                    CompatReport.user_high_id == user_high_id,
                    CompatReport.prompt_version == COMPAT_PROMPT_VERSION,
                )
            )
            if existing:
                return serialize_report(existing, user.id)
            if payload.requestId:
                existing = await session.scalar(
                    select(CompatReport).where(
                        CompatReport.request_id == payload.requestId,
                        (CompatReport.user_low_id == user.id)
                        | (CompatReport.user_high_id == user.id),
                    )
                )
                if existing:
                    return serialize_report(existing, user.id)
            raise HTTPException(status_code=409, detail="Compatibility already exists")

        payload_text = build_compatibility_payload(
            lang=user.lang,
            a_result=a_result,
            b_result=b_result,
        )
        try:
            text = generate_compatibility_text(COMPAT_SYSTEM_PROMPT, payload_text)
            async with session.begin():
                saved_report = await session.get(CompatReport, report.id)
                if saved_report:
                    saved_report.status = "ready"
                    saved_report.text = text
            return serialize_report(saved_report or report, user.id)
        except Exception as exc:
            async with session.begin():
                saved_report = await session.get(CompatReport, report.id)
                if saved_report:
                    saved_report.status = "failed"
                    saved_report.text = ""
            raise HTTPException(status_code=500, detail=str(exc))


@app.post("/compatibility/invite", response_model=CompatibilityInviteResponse)
async def compatibility_invite(
    payload: CompatibilityInviteRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    if not payload.email and not payload.telegram:
        raise HTTPException(status_code=400, detail="Email or telegram required")
    async with SessionLocal() as session:
        user = await get_current_user(
            session, authorization=authorization, x_auth_token=x_auth_token
        )
        if payload.requestId:
            existing = await session.scalar(
                select(Invite).where(
                    Invite.request_id == payload.requestId,
                    Invite.inviter_id == user.id,
                )
            )
            if existing:
                return CompatibilityInviteResponse(
                    token=existing.token,
                    status=existing.status,
                    prompt_version=existing.prompt_version,
                    created_at=existing.created_at,
                )
        existing_user = await session.scalar(
            select(User).where(
                (User.email == payload.email) | (User.telegram == payload.telegram)
            )
        )
        if existing_user:
            raise HTTPException(status_code=409, detail="Target user already exists")
        if user.compat_credits < 1:
            raise HTTPException(status_code=402, detail="Not enough credits")

        invite = Invite(
            token=uuid.uuid4().hex,
            inviter_id=user.id,
            invitee_id=None,
            prompt_version=COMPAT_PROMPT_VERSION,
            credit_spent=True,
            credit_refunded=False,
            status="sent",
            request_id=payload.requestId,
        )
        try:
            async with session.begin():
                user.compat_credits -= 1
                session.add(invite)
        except IntegrityError:
            await session.rollback()
            if payload.requestId:
                existing = await session.scalar(
                    select(Invite).where(
                        Invite.request_id == payload.requestId,
                        Invite.inviter_id == user.id,
                    )
                )
                if existing:
                    return CompatibilityInviteResponse(
                        token=existing.token,
                        status=existing.status,
                        prompt_version=existing.prompt_version,
                        created_at=existing.created_at,
                    )
            raise HTTPException(status_code=409, detail="Invite already exists")

        return CompatibilityInviteResponse(
            token=invite.token,
            status=invite.status,
            prompt_version=invite.prompt_version,
            created_at=invite.created_at,
        )


@app.post("/compatibility/accept_invite", response_model=CompatibilityReportResponse)
async def compatibility_accept_invite(
    payload: CompatibilityAcceptInviteRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    async with SessionLocal() as session:
        invitee = await get_current_user(
            session, authorization=authorization, x_auth_token=x_auth_token
        )
        invite = await session.scalar(
            select(Invite).where(Invite.token == payload.token)
        )
        if not invite:
            raise HTTPException(status_code=404, detail="Invite not found")
        if invite.inviter_id == invitee.id:
            raise HTTPException(status_code=400, detail="Cannot accept own invite")
        if invite.status == "completed":
            if invite.invitee_id != invitee.id:
                raise HTTPException(status_code=409, detail="Invite already used")
            existing = await session.scalar(
                select(CompatReport).where(
                    CompatReport.user_low_id == min(invite.inviter_id, invitee.id),
                    CompatReport.user_high_id == max(invite.inviter_id, invitee.id),
                    CompatReport.prompt_version == invite.prompt_version,
                )
            )
            if not existing:
                raise HTTPException(status_code=404, detail="Report missing")
            return serialize_report(existing, invitee.id)
        if invite.invitee_id and invite.invitee_id != invitee.id:
            raise HTTPException(status_code=409, detail="Invite already used")

        inviter = await session.get(User, invite.inviter_id)
        if not inviter:
            raise HTTPException(status_code=404, detail="Inviter not found")

        inviter_result = await session.get(UserResult, inviter.id)
        invitee_result = await session.get(UserResult, invitee.id)
        if not invitee_result:
            raise HTTPException(status_code=400, detail="Complete test first")
        if not inviter_result:
            raise HTTPException(
                status_code=400, detail="Inviter must complete test first"
            )

        user_low_id, user_high_id = sorted([inviter.id, invitee.id])
        report = await session.scalar(
            select(CompatReport).where(
                CompatReport.user_low_id == user_low_id,
                CompatReport.user_high_id == user_high_id,
                CompatReport.prompt_version == invite.prompt_version,
            )
        )
        if not report:
            report = CompatReport(
                user_low_id=user_low_id,
                user_high_id=user_high_id,
                prompt_version=invite.prompt_version,
                status="pending",
                text="",
            )

        async with session.begin():
            invite.invitee_id = invitee.id
            invite.status = "completed"
            if report.id is None:
                session.add(report)
            if (
                invite.credit_spent
                and not invite.credit_refunded
                and (inviter.has_full or inviter.packs_bought > 0)
            ):
                inviter.compat_credits += 1
                invite.credit_refunded = True

        if report.status == "ready":
            return serialize_report(report, invitee.id)

        payload_text = build_compatibility_payload(
            lang=invitee.lang,
            a_result=inviter_result,
            b_result=invitee_result,
        )
        try:
            text = generate_compatibility_text(COMPAT_SYSTEM_PROMPT, payload_text)
            async with session.begin():
                saved_report = await session.get(CompatReport, report.id)
                if saved_report:
                    saved_report.status = "ready"
                    saved_report.text = text
            return serialize_report(saved_report or report, invitee.id)
        except Exception as exc:
            async with session.begin():
                saved_report = await session.get(CompatReport, report.id)
                if saved_report:
                    saved_report.status = "failed"
                    saved_report.text = ""
            raise HTTPException(status_code=500, detail=str(exc))


@app.get("/compatibility/list", response_model=CompatibilityListResponse)
async def compatibility_list(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
):
    async with SessionLocal() as session:
        user = await get_current_user(
            session, authorization=authorization, x_auth_token=x_auth_token
        )
        query = await session.execute(
            select(CompatReport)
            .where(
                (CompatReport.user_low_id == user.id)
                | (CompatReport.user_high_id == user.id)
            )
            .order_by(CompatReport.created_at.desc())
        )
        reports = query.scalars().all()
        return CompatibilityListResponse(
            items=[serialize_report(report, user.id) for report in reports]
        )


@app.get("/result/full/{runId}", response_model=FullResponse)
async def get_full_result(runId: str):
    try:
        run_uuid = uuid.UUID(runId)
    except ValueError:
        raise HTTPException(status_code=404, detail="Full result not found")

    async with SessionLocal() as session:
        result = await session.get(FullResultORM, run_uuid)
        short_result = await session.get(ShortResultORM, run_uuid)

    if result is None or short_result is None:
        raise HTTPException(status_code=404, detail="Full result not found")

    return (
        {
            "type": "full",
            "result_id": str(result.run_id),
            "result": {
                "animal": short_result.animal,
                "element": short_result.element,
                "genderForm": short_result.gender_form,
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
