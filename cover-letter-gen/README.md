# cover-letter-gen

3-pass пайплайн для генерации сопроводительных писем под Flutter-разработчика.

Это самостоятельный Python-инструмент, лежащий в подпапке этого Flutter-репо — он не зависит от Dart-кода и не используется приложением `create_order_app`.

## Зачем

Старый подход (один монолитный промпт ~700 строк + `T=0.85`) давал нестабильный результат:
- письма часто совпадали слово в слово между разными вакансиями;
- модель регулярно выдумывала факты вроде «production-проекты с сотнями пользователей», которых нет в резюме;
- внутренние «чек-листы» в том же вызове LLM игнорировались.

Новый пайплайн разнесёт это на три специализированных вызова.

## Архитектура

```
[Vacancy + Resume]
       │
       ▼
┌──────────────────────┐
│ Pass 1: Analyzer     │  T=0.0, JSON-only
│  → vacancy_type      │
│  → top_requirements  │
│  → best_project      │
│  → evidence[]        │
│  → hook_phrase       │
│  → allowed_numbers   │  ← белый список чисел (anti-hallucination)
└──────────┬───────────┘
           │ JSON
           ▼
┌──────────────────────┐
│ Pass 2: Writer       │  T=0.4, plain text
│  Короткий промпт     │
│  + JSON-факты        │
│  → cover letter      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│ Pass 3: Validator        │
│  • Deterministic (regex) │  длина, абзацы, числа из allowed_numbers,
│  • Semantic (LLM, T=0)   │  запреты, англицизмы, hook-phrase, advice-to-company
└──────────┬───────────────┘
           │ если passed=false
           ▼
       [Writer rewrites with feedback]   max 2 ретрая
```

Ключевые свойства:
- **allowed_numbers** собирается Analyzer'ом из резюме + автоматически дополняется числами из достижений. Writer'у запрещено писать любые числа вне этого списка; детерминированный валидатор подтверждает это regex'ом — никаких «сотен пользователей».
- **used_starts** копится по батчу — Writer получает список уже использованных первых фраз и должен сформулировать новое начало.
- **Двухслойный валидатор**: дешёвый детерминированный (длина, запретные фразы, числа, англицизмы) + дорогой семантический (LLM смотрит «обращаемся ли к hook_phrase», «не консультирует ли кандидат компанию», «не слабый ли финал»).
- **Ретраи с фидбэком**: при провале валидации Writer получает в user-промпте список конкретных нарушений и переписывает только их.
- **Промпты сокращены**: Writer-системник ~30 строк против ~700 в старой версии. Запреты, повторяющиеся в трёх местах, удалены — деривация ушла в код.

## Структура

```
cover-letter-gen/
├── config/
│   ├── settings.example.yaml    # шаблон настроек (api_key через ${LLM_API_KEY})
│   └── resume.example.yaml      # анонимизированный шаблон резюме
├── data/
│   └── (пусто — vacancies.json кладёшь сам)
├── scripts/
│   └── generate.py              # CLI
├── src/
│   ├── prompts/                 # три промпта (analyzer, writer, validator)
│   ├── models.py                # Profile, Vacancy, Project и пр.
│   ├── profile_loader.py
│   ├── vacancy_loader.py
│   ├── llm_client.py            # async httpx + retries + JSON-mode fallback
│   ├── analyzer.py              # Pass 1
│   ├── writer.py                # Pass 2
│   ├── validator.py             # Pass 3 (детерминированный + семантический)
│   └── pipeline.py              # оркестратор
├── tests/
│   ├── test_validator.py        # юнит-тесты детерминированных правил
│   └── test_pipeline.py         # e2e-тесты с замоканным LLM
├── .env.example
├── .gitignore                   # настоящие .env / resume.yaml / vacancies.json не коммитятся
└── requirements.txt
```

## Установка

```bash
cd cover-letter-gen
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Настройка

```bash
cp .env.example .env                       # затем подставь свой LLM_API_KEY
cp config/settings.example.yaml config/settings.yaml
cp config/resume.example.yaml config/resume.yaml   # заполни своими данными
```

`.env`, `config/settings.yaml`, `config/resume.yaml` — все игнорируются git'ом.

`vacancies.json` (выход вашего `vacancy-agent-skill`) положите в `data/vacancies.json`. Опциональный фильтр — список ID одной строкой в `data/selected_vacancy_ids.txt`.

## Запуск

```bash
python scripts/generate.py \
  --resume config/resume.yaml \
  --settings config/settings.yaml \
  --vacancies data/vacancies.json \
  --selected data/selected_vacancy_ids.txt \
  --out letters/
```

Письма пишутся в `letters/<company>_<id8>.txt`. Сводный JSON по каждому письму (analyzer_json, нарушения, число попыток) — в `letters/_summary.json`.

Флаги:
- `--limit N` — обработать только первые N вакансий.
- `--no-semantic` — пропустить семантический валидатор (дешевле, чуть менее строго).
- `--log-level DEBUG` — больше подробностей.

## Тесты

```bash
cd cover-letter-gen
PYTHONPATH=. pytest -q
```

Тесты не делают реальных HTTP-вызовов — `FakeLLMClient` подменяет LLM по системному промпту и проигрывает заранее заданные ответы.

## Anti-hallucination на практике

Все три уровня защиты:

| Уровень | Что делает | Где живёт |
|---|---|---|
| 1. allowed_numbers | Жёсткий белый список чисел | `analyzer.py:_normalize` |
| 2. promt-конструкт | «Числа — ТОЛЬКО из allowed_numbers» в системнике писателя | `prompts/writer.py` |
| 3. постпроверка regex | Любое число в письме сверяется со списком | `validator.py:validate_deterministic` |

Если модель всё-таки выдумала «сотни пользователей» — детерминированный валидатор отловит «сотни» как невалидное (после нормализации пробелов) и Writer перепишет письмо с явным указанием на нарушение.

## Что НЕ делает

- Не парсит вакансии — это работа отдельного `vacancy-agent-skill`.
- Не отправляет письма автоматически — только генерирует текст в файлы.
- Не использует `getattr`/`Any`-лазейки — все данные типизированы dataclass'ами.
