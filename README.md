# Candidate Reader

The project provides two small readers:

- `candidate-reader` converts a candidate CV and candidate constraints into validated, cacheable
  Pydantic JSON.
- `company-reader` converts a company employment offer into validated, cacheable Pydantic JSON.

Company NDAs/constraint documents, comparisons, legal analysis, and report generation are outside
this milestone.
## Setup

```powershell
Copy-Item .env.example .env
```

Add an OpenAI API key to `.env`. Tracing must remain disabled:

```text
OPENAI_MODEL=gpt-5.6-terra
OPENAI_AGENTS_DISABLE_TRACING=1
MAX_OPENAI_COST_USD=0.50
```

`OPENAI_MODEL` accepts any model name supported by the configured OpenAI endpoint. The CLI does
not override it.

The application has built-in prices for common GPT-4.1, GPT-4o, GPT-5, GPT-5.3, GPT-5.4,
GPT-5.5, GPT-5.6, and GPT-6 models. For any other model, set all three rates in `.env` so the cost
ceiling can be calculated:

```text
OPENAI_INPUT_PRICE_PER_MILLION=...
OPENAI_CACHE_WRITE_PRICE_PER_MILLION=...
OPENAI_OUTPUT_PRICE_PER_MILLION=...
```

Install dependencies into the managed Python environment:

```powershell
uv sync --python 3.12
```

## Run

Interactive mode discovers the current candidate files:

```powershell
uv run candidate-reader read
```

Explicit non-interactive mode:

```powershell
uv run candidate-reader read `
  --cv "candidate\cv\candidate.pdf" `
  --constraints "candidate\constraints\constraints.txt" `
  --non-interactive `
  --yes
```

Output:

```text
structured_data\candidate_cv.json
structured_data\candidate_constraints.json
```

Valid unchanged output is reused without an OpenAI call. Use `--force` to regenerate it.

Read the company employment offer:

```powershell
uv run company-reader read
```

Explicit non-interactive mode:

```powershell
uv run company-reader read `
  --offer "company\contract\offer.pdf" `
  --non-interactive
```

Company output:

```text
structured_data\company_offer.json
```

The company reader parses numbered employment-contract sections locally and does not make an
OpenAI call.

## Compare candidate requirements with the offer

```powershell
uv run comparison run --yes --non-interactive
```

The comparison command validates the three files in `structured_data`, evaluates measurable
requirements in Python, sends only unresolved structured evidence for semantic analysis and
independent verification, and writes `report.md` plus
`structured_data\comparison_result.json`. Unchanged inputs reuse the cached result.

## Cost protection

Before every model call, the application reserves a conservative worst-case amount using:

- source UTF-8 byte length;
- schema and SDK overhead;
- configured maximum output tokens;
- standard input, cache-write, and output prices;
- all earlier calls in the same process.

The call is blocked if the projected total exceeds `MAX_OPENAI_COST_USD`. Automatic SDK and HTTP
retries are disabled. One application-level structured-output repair attempt is allowed only when
it also fits within the remaining budget.

The guard is a conservative software estimate based on known model prices, not an OpenAI billing
limit. Keep an API-platform project budget as a second line of protection.

## Validate

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```
