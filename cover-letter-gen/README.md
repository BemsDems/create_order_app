# cover-letter-gen

3-pass пайплайн для генерации сопроводительных писем под Flutter-разработчика.

Это самостоятельный Python-инструмент, лежащий в подпапке этого Flutter-репо — он не зависит от Dart-кода и не используется приложением `create_order_app`.

## Архитектура (v4)

### v4 hardening (точечные баг-фиксы поверх v3)

После аудита v3 нашлось несколько багов, ломавших качество писем в крайних случаях. Все исправления — точечные, обратная совместимость по схеме `_summary.json` и публичному API сохранена.

| Симптом | Корневая причина | Исправление |
|---|---|---|
| Релевантные вакансии (`Clean Architecture`, `Secure Storage`) пропускались с `skipped_no_tech_overlap` | Pre-Analyzer fit-gate сравнивал `tech.lower() in tokens_lower` — `tokens_lower` это множество **отдельных слов**, мульти-словные tech (`"clean architecture"`) никогда не матчили | `vacancy_fit` теперь подстрочно ищет мульти-словные tech в полном lowercased-тексте; одно-словные по-прежнему через токен-сет (`"dart"` не матчит `"darts"`). Тот же фикс применён к `primary_match`. |
| `no_years_in_opener` пропускал «В 2024 году я начал...» | Регэксп `_OPENER_YEARS_RE` принимал любую цифру в первом предложении | Регэксп теперь требует `\d{1,2}` рядом со словом «год/года/лет», и цифра не должна быть частью соседнего числа («2024 году» не матчит) |
| LLM иногда вставлял выдуманный `hook_phrase` («сотни тысяч пользователей»), и Writer строил вокруг него абзац | Analyzer'овый `_ground` фильтровал проекты, числа, достижения — но не `hook_phrase` | `_ground` принимает `vacancy` и проверяет, что ≥2 значащих слова (без стоп-слов, 3+ символов, prefix-3 на русские словоформы) из hook'а присутствуют в тексте вакансии; иначе hook стирается |
| `forbidden_claim "финтех"` срабатывал даже когда индустрия позиции в резюме = «Финтех» | `extract_canonical_facts` не клал `position.industry` в `profile_text_lower` | Индустрия теперь добавляется в `text_chunks` рядом с описаниями проектов |
| `invented_number 10` ловило `"Python 3.10"`, `iOS 17` и т.п. | Извлечение чисел не отличало версии технологий от метрик | `_extract_numbers` принимает `tech_whitelist` и пропускает цифры, идущие сразу после whitelist'нутой tech-имени (`"Python 3.10"`, `"Dart 3"`, `"iOS 17"`) |
| 429-ответы провайдера падали с raise без ретрая; пустой `content`+`reasoning_content` молча возвращался как `""` | LLM-клиент не различал 429 (rate-limit) и не-эмпти content от провайдера | Добавлено: `429` → backoff с уважением `Retry-After`-заголовка (cap 60s); пустой content → ретрай как transient error; пустой после всех попыток → понятный `RuntimeError("empty content")` |
| `_strip_signature_lines` срезал только «С уважением», но не «Best regards», «Спасибо», «Sincerely» и т.п. | Хардкод одной фразы | Расширенный whitelist префиксов подписей |
| Multi-word tech (`Secure Storage`, `Clean Architecture`) флагалось `unknown_tech_term` в валидаторе — токенизатор видел только отдельные слова `Secure`/`Storage` | Тот же класс багов, что 1.1, но в `validate_deterministic` (вторая точка); fix 1.1 покрывал только `vacancy_fit` | `tech_allowed` дополняется составляющими словами из мульти-словных терминов. «Secure Storage» → также разрешает «Secure» и «Storage» |
| `invented_number 3` для письма с опенером «3+ года» — даже когда у кандидата `experience.total_years: 3` | Валидатор сверялся с `analyzer.selected_numbers` (стилистическая подвыборка Analyzer'а), а не с `facts.allowed_numbers` (полный whitelist). Опенеры всегда подставляют `experience_years` через `{years}+ года`, но Analyzer мог не включить эту цифру в `selected_numbers` | `pipeline` передаёт в валидатор `facts.allowed_numbers` (объединённое с `selected_numbers`), а не только Analyzer-подвыборку |
| `_opener_has_years` не принимал «За три года...» (только «3 года») | Регэксп ловил только цифровую форму | Поддержка русских словесных числительных «один».."двадцать" + опционально «с лишним» |
| Mistral систематически писал короткие письма (60-94 слова) | После v3 убраны литералы длины из system prompt, словесного guidance не было — модель видела «два абзаца» и интерпретировала как два предложения | В system prompt добавлено «Каждый абзац — РАЗВЁРНУТЫЙ: минимум четыре-пять полноценных предложений с конкретикой. Не короткое summary». Числительные словами, чтобы не попасть в numeric whitelist |

Добавлены тесты:
- `tests/test_facts.py`: мульти-словные tech в `vacancy_fit`, `position.industry` в `profile_text_lower`.
- `tests/test_validator.py`: `_opener_has_years` ложноположительные/положительные, версии технологий в `_extract_numbers`, расширенные префиксы подписей.
- `tests/test_pipeline.py`: грааундинг hook'а (выдуманный → стирается, реальный → сохраняется).
- `tests/test_llm_client.py` (новый файл): 429 + `Retry-After`, пустой content, fallback `reasoning_content`, JSON-mode 400-fallback, parsing `Retry-After`, 5xx ретрай.

Всего 64 теста (39 v3 + 25 v4) проходят локально.

Smoke-проверено на NVIDIA Mistral (`mistralai/mistral-large-3-675b-instruct-2512`, `mistralai/mistral-small-4-119b-2603`). До v4: 4-6 нарушений на попытку, большинство — ложные срабатывания (`unknown_tech_term: Secure`, `invented_number: 3`). После v4 — 1-3 реальных нарушения качества модели (`too_short`, `forbidden_phrase: благодаря`); ложноположительных не наблюдалось.

## Зачем

Старый подход (один монолитный промпт ~700 строк + `T=0.85`) давал нестабильный результат:
- письма часто совпадали слово в слово между разными вакансиями;
- модель регулярно выдумывала факты вроде «production-проекты с сотнями пользователей», которых нет в резюме;
- внутренние «чек-листы» в том же вызове LLM игнорировались.

Новый пайплайн разнесёт это на три специализированных вызова + детерминированный слой фактов.

## Архитектура (v3)

### v3 hardening (на основе результата реального запуска)

В v2 пайплайн прошёл все мокнутые тесты, но на реальном LLM (`owl-alpha`) часто валился по `_summary.json`:

| Симптом в письме | Причина | Как исправлено в v3 |
|---|---|---|
| `anglicism openers / confidence / achievements` | Writer получал JSON-блоки с английскими ключами в user-prompt'е и переписывал их в письмо | `build_writer_user` теперь строит пользовательский промпт как чистый русский текст — никаких JSON-дампов, никаких английских ключей |
| `invented_number 90 / 115` | Системный промпт содержал «90–115 слов» / «100–130 слов» — модель копировала цифры в текст | Из системных промптов убраны все литералы длины. Длину контролирует валидатор + ретраи с подсказками «удлини на N слов» |
| `unknown_tech_term Go / Kafka / Redis` для backend-вакансий | Backend-вакансии без Flutter всё равно шли в Analyzer | Pre-Analyzer **fit gate**: если пересечения tech-stack нет и primary skill вакансии не упомянут — вакансия отбрасывается без LLM-вызова (`error="skipped_no_tech_overlap"`) |
| Один и тот же `invented_number` или `forbidden_claim` после retry | Обычный feedback в user-prompt'е модель игнорировала на повторе | Repeat-violation **escalation**: если та же пара (rule, evidence) повторилась — следующий промпт получает блок «СТРОГО (нарушение → автоматическое отклонение)» с прямым запретом |
| Иногда в письмо попадал `selected_project`, `fix_hint` и т.п. как слова | Эти ключи реально были в JSON, который Writer видел в v2 | Новое правило валидатора `meta_leak`: чёрный список явных служебных слов + регексп `[a-z]+_[a-z_]+` на любые snake_case-токены |

В v3 ничего не сломано из v2 — обратная совместимость по схеме `_summary.json` и публичному API пайплайна сохранена. Pre-Analyzer fit-gate можно отключить флагом `enforce_fit_gate: false` в `settings.yaml`.

## Архитектура (v2)

```
                    [Profile (resume.yaml)]
                            │
                            ▼  deterministic, no LLM
                  ┌─────────────────────┐
                  │  CanonicalFacts     │
                  │  (src/facts.py)     │
                  │  • allowed_numbers  │  ← числа из резюме
                  │  • allowed_tech     │  ← стек из проектов
                  │  • allowed_projects │
                  │  • projects[name]:  │
                  │    achievements[]   │  ← дословно из резюме
                  │  • forbidden_claims │  ← «финтех», «сотни пользователей» и т.п.
                  └────────┬────────────┘
                           │
              [Vacancy]    │   единый источник истины
                  │        │
                  ▼        ▼
            ┌──────────────────────────────┐
            │ Pass 1: Analyzer             │  T=0.0, JSON-only
            │  Вход: vacancy + CanonicalFacts
            │  Выход:                      │
            │   • selected_project (∈ allowed_project_names)
            │   • confidence (0.0-1.0)     │
            │   • selected_numbers (⊂ allowed_numbers)
            │   • selected_achievements (⊂ project.achievements, дословно)
            │   • hook_phrase              │
            │  + Python grounding отбрасывает
            │    всё, чего нет в CanonicalFacts
            └──────────┬───────────────────┘
                       │
                       ▼   confidence < 0.2?  → skip
                       │   confidence < 0.5?  → universal mode (1 абзац, без hook)
                       │
            ┌──────────────────────────────┐
            │ Pass 2: Writer               │  T=0.4
            │  Вход: analyzer JSON + opener pool из CanonicalFacts
            │  Системник: ~40 строк        │
            │  Выход: текст письма         │
            └──────────┬───────────────────┘
                       │
            ┌──────────────────────────────┐
            │ Pass 3: Validator            │
            │  Детерминистика (regex):     │
            │   • длина, абзацы            │
            │   • forbidden_phrase         │
            │   • invented_number          │
            │   • forbidden_claim ← grounded
            │   • unknown_tech_term ← grounded
            │   • library_name             │
            │   • anglicism (с whitelist)  │
            │  Семантика (LLM, T=0):       │
            │   • hook_not_addressed       │
            │   • advice_to_company        │
            │   • weak_ending              │
            │   • invented_facts           │
            └──────────┬───────────────────┘
                       │ если passed=false → Writer.rewrite(feedback)   max 2 ретрая
```

### Anti-hallucination: 3 уровня + grounding

| Уровень | Что делает | Где живёт |
|---|---|---|
| **0. CanonicalFacts** | Whitelist строится из `resume.yaml` детерминированно. Источник истины. | `src/facts.py:extract_canonical_facts` |
| 1. Analyzer grounding | LLM-выход фильтруется: проект → должен быть в `allowed_project_names`; числа → в `allowed_numbers`; достижения → дословно из `projects[*].achievements`. Всё лишнее ВЫРЕЗАЕТСЯ ещё до Writer'а. | `src/analyzer.py:_ground` |
| 2. Prompt constraints | Системник Writer'а: «числа — только из selected_numbers; факты — только из selected_achievements». | `src/prompts/writer.py` |
| 3. Deterministic validator | regex-проверка: числа в письме сверяются со списком; smell-фразы из `forbidden_claims` сравниваются с резюме (если фразы нет в резюме — нельзя её употреблять). | `src/validator.py:validate_deterministic` |
| 4. Semantic validator (опц.) | LLM проверяет 5 семантических нарушений, которые regex не ловит. | `src/validator.py:validate_semantic` |

Аналитик **больше не формирует** whitelist — он только выбирает из готового. Если первая LLM-стадия ошибётся, эта ошибка не доживёт до Writer'а.

### Confidence routing

Analyzer ставит `confidence ∈ [0,1]` — насколько выбранный проект релевантен этой конкретной вакансии.

| `confidence` | Действие | Системник Writer'а |
|---|---|---|
| ≥ 0.5 | standard | два абзаца, второй отвечает на `hook_phrase` |
| 0.2 – 0.5 | universal | один абзац, без агрессивной привязки к вакансии |
| < 0.2 | skip | письмо не генерируется, попадает в `_summary.json` с `error="skipped_low_confidence"` |

Пороги конфигурируются в `settings.yaml` (`low_confidence_threshold`, `skip_below_confidence`).

### Curated opener pool

Вместо «не используй эти заходы» (негативный constraint) Writer получает 2 канонических шаблона первой фразы из `src/prompts/opener_pool.py` и должен выбрать/адаптировать один. Использованные openers копятся в `pipeline.used_starts` по батчу.

### LLM retry vs Writer retry — разные уровни

Не путать:
- **LLMClient retry** (`src/llm_client.py`): сетевые ошибки, 5xx, таймауты, пустой `choices: []` → экспоненциальный backoff (0.5s → 1s → 2s → 4s → 8s), `max_retries=3` по умолчанию.
- **Writer retry** (`src/pipeline.py`): письмо не прошло валидацию → Writer переписывает с конкретным списком нарушений, `max_writer_retries=2` по умолчанию.

Первое — про инфраструктуру, второе — про качество. Их счётчики независимы и хорошо видны в логах.

## Структура

```
cover-letter-gen/
├── config/
│   ├── settings.example.yaml    # шаблон настроек
│   └── resume.example.yaml      # анонимизированный шаблон резюме
├── data/                        # vacancies.json кладёшь сам (git-ignored)
├── scripts/
│   └── generate.py              # CLI
├── src/
│   ├── prompts/                 # три промпта + opener_pool
│   ├── models.py                # Profile, Vacancy, Project и пр.
│   ├── facts.py                 # CanonicalFacts — единый whitelist
│   ├── profile_loader.py
│   ├── vacancy_loader.py
│   ├── llm_client.py            # async httpx + retries + JSON-mode fallback
│   ├── analyzer.py              # Pass 1 (с grounding)
│   ├── writer.py                # Pass 2
│   ├── validator.py             # Pass 3 (детерминированный + семантический)
│   └── pipeline.py              # оркестратор
├── tests/
│   ├── test_facts.py            # извлечение CanonicalFacts
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

Письма пишутся в `letters/<company>_<id8>.txt`. Детальный JSON по каждому письму — в `letters/_summary.json`, включая:

```json
{
  "vacancy_id": "...",
  "company": "...",
  "selected_project": "OtherMark",
  "confidence": 0.85,
  "confidence_reason": "...",
  "used_numbers": ["3", "5", "11000"],
  "used_tech": ["Flutter", "BLoC", "Clean", "Architecture"],
  "universal_mode": false,
  "semantic_validator_used": true,
  "word_count": 124,
  "passed": true,
  "attempts": 1,
  "violations": [],
  "error": null
}
```

Флаги:
- `--limit N` — обработать только первые N вакансий.
- `--no-semantic` — пропустить семантический валидатор (быстрее, дешевле).
- `--log-level DEBUG` — больше подробностей.

## Тесты

```bash
cd cover-letter-gen
PYTHONPATH=. pytest -q
```

Тесты не делают реальных HTTP-вызовов — `FakeLLMClient` подменяет LLM по системному промпту и проигрывает заранее заданные ответы. Покрывают: извлечение `CanonicalFacts`, все правила детерминированного валидатора, retry-поведение пайплайна, anti-hallucination grounding, low-confidence routing.

## Что НЕ делает

- Не парсит вакансии — это работа отдельного `vacancy-agent-skill`.
- Не отправляет письма автоматически — только генерирует текст в файлы.
- Не использует `getattr`/`Any`-лазейки — все данные типизированы dataclass'ами.
- Не выдумывает факты: всё, что попадает в письмо, проходит через `CanonicalFacts`.
