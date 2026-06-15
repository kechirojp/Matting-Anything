70https://zenn.dev/green_tea/articles/d310e5cf809190


48

[テーマ「この春、始めたこと」](https://zenn.dev/contests/zennfes-spring-2026-new-start)

要約
Copilot指示体系の設計



AGENTS.md はルーター、skills は作業別手順、docs/agent はプロジェクト固有知識


このリポジトリでは、エージェント向けの指示を3層に分割しています：

1. .github/copilot-instructions.md— 薄い共通指示。全タスクで必要な最小限のルールと、詳細な指示への案内。
2. AGENTS.md— エージェント用ルーター。タスクの種類に応じて適切なskillファイルへ誘導する。
3. .github/skills/*/SKILL.md— 作業別の詳細手順。Python、SQL、データ処理、可視化など。
4. docs/agent/*— プロジェクト固有の知識。データカタログ、指標定義、分析ワークフローなど。
5. .github/instructions/*.instructions.md— パス別の補助指示。ファイルの種類に応じた自動適用ルール。
6. .github/prompts/*.prompt.md— 再利用可能なプロンプト。分析計画、SQLレビュー、レポート作成など。
この設計により、全部入りの巨大な指示ファイルを避け、トークン効率よく必要な情報だけを参照できます。





## はじめに

2026年現在、多くのエンジニアは GitHub Copilot, Claude Code, Cline, Cursor をはじめとするAIコーディングツールを使っているでしょう。データサイエンティストも例外に漏れず、AIコーディングツールを使っています。

AIコーディングツールの能力を最大限引き出すためには、AIに適切な前提知識を教えてあげることが重要です。本記事では、私が普段の分析業務で `AGENTS.md` に書いている内容に加え、本記事執筆を良い機会と思って Skills に整理した内容を紹介します。

大きな方針は次の通りです。

- 全タスクで守ってほしいことは `AGENTS.md` に薄く書く
- 作業別の詳しいルールは skills に分ける
- プロジェクト固有の情報は docs に分ける
- よく使う依頼は prompts にする
- 本当に守らせたいことは scripts や CI でも検査する

## 結論

こちらです。以下ぐだぐだ書かれたものを読むよりも、一旦`.github` をそのままご自身のプロジェクトフォルダに置いてその効果をご確認いただく方が早いかもしれません。

<iframe src="https://embed.zenn.studio/card#zenn-embedded__b223d8739b90b" frameborder="0" height="122"></iframe>

```
.
├── AGENTS.md
├── .github/
│   ├── copilot-instructions.md
│   ├── instructions/
│   │   ├── python.instructions.md
│   │   ├── sql.instructions.md
│   │   ├── notebooks.instructions.md
│   │   ├── docs.instructions.md
│   │   └── data.instructions.md
│   ├── prompts/
│   │   ├── plan-analysis.prompt.md
│   │   ├── review-sql.prompt.md
│   │   ├── run-eda.prompt.md
│   │   ├── run-modeling.prompt.md
│   │   ├── summarize-analysis.prompt.md
│   │   ├── prepare-pr.prompt.md
│   │   └── update-agent-docs.prompt.md
│   └── skills/
│       ├── python-project-ops/
│       │   └── SKILL.md
│       ├── safe-data-handling/
│       │   └── SKILL.md
│       ├── sql-analysis/
│       │   └── SKILL.md
│       ├── python-style/
│       │   └── SKILL.md
│       ├── dataframe-polars/
│       │   └── SKILL.md
│       ├── visualization/
│       │   └── SKILL.md
│       ├── path-and-io/
│       │   └── SKILL.md
│       ├── notebook-workflow/
│       │   └── SKILL.md
│       ├── statistical-ml-review/
│       │   └── SKILL.md
│       └── analysis-reporting/
│           └── SKILL.md
├── docs/
│   └── agent/
│       ├── project-overview.md
│       ├── repository-structure.md
│       ├── data-catalog.md
│       ├── metrics-and-definitions.md
│       ├── analysis-workflow.md
│       ├── statistical-and-ml-guidelines.md
│       ├── validation-and-testing.md
│       ├── reporting-guidelines.md
│       ├── security-and-privacy.md
│       └── agent-behavior.md
├── scripts/
│   ├── check_no_raw_data_commit.py
│   ├── check_no_sensitive_patterns.py
│   ├── run_quality_checks.sh
│   └── validate_agent_docs.py
└── src/
```

`AGENTS.md` は秘伝のタレ [^1] のようにするのではなく、各種ルールへのルーターにします。

| ファイル・ディレクトリ | 役割 |
| --- | --- |
| `.github/copilot-instructions.md` | Copilot に常に読ませたい最小限の共通方針 |
| `AGENTS.md` | AI エージェント向けのルーター |
| `.github/instructions/*.instructions.md` | Python, SQL, notebook など、パス別に効かせる指示 |
| `.github/skills/*/SKILL.md` | SQL、Polars、可視化、データ保護などの作業別手順 |
| `.github/prompts/*.prompt.md` | 分析計画、SQL レビュー、結果要約などの再利用プロンプト |
| `docs/agent/*` | プロジェクト固有の知識、データ定義、指標定義など |
| `scripts/*` | AI にお願いするのではなく機械的に検査したいもの [^2] |

GitHub Copilot の場合、repository-wide instructions [^3] は `.github/copilot-instructions.md` 、path-specific instructions [^4] は `.github/instructions/*.instructions.md` 、agent instructions [^5] は `AGENTS.md` に置けます [^6] 。また、agent skills は `.github/skills/<skill-name>/SKILL.md` のように置けます [^7] 。Prompt files は `.github/prompts/*.prompt.md` に置くと、VS Code などから再利用しやすくなります [^8] 。

## なぜ全部 AGENTS.md に書かないのか

当初私はそんなにAIに守らせたいルールがたくさんなかったので、最初は `AGENTS.md` に全部書けばよいと思っていました。

実際、Python の実行コマンド、SQL の書き方、docstring の書き方、データの扱い、Polars のルール、Matplotlib のルール、ファイルパスのルールなどを全部 `AGENTS.md` に書いておけば、AI はそれなりに従ってくれます。

ただ、やっているうちに以下のような問題が出てきます。

- 関係ない知識を読み込んでしまう（SQL を書かない作業でも SQL の長いルールを読ませることになる）
- `AGENTS.md` が長くなり、人間が読まなくなる/メンテしなくなる
- プロジェクト固有の知識と全プロジェクト共通のルールが混ざる
- 似たような指示が増えて矛盾しやすくなる

特にデータサイエンスのプロジェクトは、コードだけでなくデータ、指標定義、SQL、notebook、可視化、分析レポートまで扱います。全部を 1 ファイルに入れると、すぐに巨大化します。

なので、私は [結論](#%E7%B5%90%E8%AB%96) のように分けるのが良いと思っています [^9] 。

特に2026年6月に GitHub Copilotは Premium Requests 制（呼び出す回数に上限がある）から、GitHub AI Credits制（使えるトークン数に上限がある）へ移行しましたので、尚更コンテキストの管理はシビアになってきます。Claude Code など他のツールも大概がトークンベースです。

<iframe src="https://embed.zenn.studio/card#zenn-embedded__e153a3e75708b" frameborder="0" height="122"></iframe>

## まずは AGENTS.md を薄く作る

`AGENTS.md` はこのくらいにします。ここに詳細ルールを全部書かないのがポイントです。

AGENTS.md の例

```
# AGENTS.md

This file is an **agent router**. It provides high-level rules and directs agents to the appropriate skill files for detailed instructions.

Detailed task-specific procedures are in \`.github/skills/*/SKILL.md\`.
Project-specific context is in \`docs/agent/*\`.

## Hard Rules (Always Apply)

- Never commit raw data, credentials, API keys, tokens, or customer-level records.
- Never modify, overwrite, delete, or regenerate raw data directly.
- Prefer small, reviewable changes.
- Explain assumptions before non-trivial analytical decisions.
- Ask for clarification when data semantics are unclear.
- Use \`uv\` exclusively for Python dependency management. Never use pip, conda, poetry, or pipenv.

## Routing Table

| Task | Skill |
|------|-------|
| Dependencies, tests, lint, type check, notebook execution | [python-project-ops](.github/skills/python-project-ops/SKILL.md) |
| Reading / writing / moving data files | [safe-data-handling](.github/skills/safe-data-handling/SKILL.md) + [path-and-io](.github/skills/path-and-io/SKILL.md) |
| Writing or reviewing SQL | [sql-analysis](.github/skills/sql-analysis/SKILL.md) |
| Writing or reviewing Python code | [python-style](.github/skills/python-style/SKILL.md) |
| DataFrame operations | [dataframe-polars](.github/skills/dataframe-polars/SKILL.md) |
| Charts and visualization | [visualization](.github/skills/visualization/SKILL.md) |
| Notebook creation and editing | [notebook-workflow](.github/skills/notebook-workflow/SKILL.md) |
| Statistics or ML | [statistical-ml-review](.github/skills/statistical-ml-review/SKILL.md) |
| Analysis summaries and reports | [analysis-reporting](.github/skills/analysis-reporting/SKILL.md) |
| File paths and I/O | [path-and-io](.github/skills/path-and-io/SKILL.md) |

## Project Context (docs/agent)

| Document | Purpose |
|----------|---------|
| [project-overview.md](docs/agent/project-overview.md) | プロジェクトの目的とスコープ |
| [repository-structure.md](docs/agent/repository-structure.md) | ディレクトリ構成 |
| [data-catalog.md](docs/agent/data-catalog.md) | データセット一覧と定義 |
| [metrics-and-definitions.md](docs/agent/metrics-and-definitions.md) | 指標定義 |
| [analysis-workflow.md](docs/agent/analysis-workflow.md) | 分析ワークフロー |
| [statistical-and-ml-guidelines.md](docs/agent/statistical-and-ml-guidelines.md) | 統計・MLガイドライン |
| [validation-and-testing.md](docs/agent/validation-and-testing.md) | テスト・検証方針 |
| [reporting-guidelines.md](docs/agent/reporting-guidelines.md) | 報告テンプレート |
| [security-and-privacy.md](docs/agent/security-and-privacy.md) | セキュリティ・プライバシー |
| [agent-behavior.md](docs/agent/agent-behavior.md) | エージェント行動指針 |
```

最低限にしたつもりですが、気になる点があれば教えてください。

## .github/copilot-instructions.md に書くこと

GitHub Copilot を使うなら、 `AGENTS.md` だけでなく `.github/copilot-instructions.md` も置いておくとよいです。

ここにはほぼ全タスクで効かせたいことだけを書きます。これも長くしません。

.github/copilot-instructions.md の例

```
# Repository-Wide Custom Instructions

This is a **Python 3.11 data science / analysis project**.

## Package Management

- Use **uv** exclusively for all dependency management.
- Never use pip, pip3, \`python -m pip\`, poetry, conda, pipenv, or easy_install.

## Data Safety

- Never commit raw data, credentials, API keys, tokens, or customer-level records.
- Never modify or delete raw data directly.
- Treat \`data/raw\` and \`data/external\` as immutable.

## Where to Find Detailed Rules

- **Task-specific skills**: \`.github/skills/*/SKILL.md\` — see \`AGENTS.md\` for routing.
- **Project context**: \`docs/agent/*\` — data catalog, metrics, workflow, etc.
- **Path-specific hints**: \`.github/instructions/*.instructions.md\`

## Common Commands

\`\`\`bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run python scripts/check_no_raw_data_commit.py
uv run python scripts/check_no_sensitive_patterns.py
```

## Key Conventions

- DataFrame operations: prefer **polars** over pandas.
- Visualization: use `fig, ax = plt.subplots(...)`, not `plt.figure(...)`.
- File paths: use `pathlib.Path`, no absolute local paths.
- Docstrings: Google-style.
- Inline comments: Japanese.
- Reports and documentation: Japanese.

`copilot-instructions.md` は、毎回 Copilot に渡されても困らないくらいの量にしておくのが良さそうです。

## 全プロジェクト共通

プロジェクトごとに設定する項目と、分析プロジェクトなら必ず設定しているものとがあります。まずは Python を用いた分析なら絶対に指定しているものから紹介します。

ただし、ここから先は基本的に `AGENTS.md` へ直接書くのではなく、skills に分けます。

### 環境構築・実行コマンド

Python のバージョン指定や、 [ruff](https://docs.astral.sh/ruff/) [^10] や [uv](https://docs.astral.sh/uv/) [^11] の使い方を教えています。私や私の所属する組織では、Python のパッケージ管理では uv を使うことにしているので、uv 以外絶対に [^12] 使ってほしくないです。

これは `AGENTS.md` に長々と書くのではなく、必要な時に読んでくれればよいので、`.github/skills/python-project-ops/SKILL.md` に書きます。

.github/skills/python-project-ops/SKILL.md

```
---
name: python-project-ops
description: Use this when managing Python dependencies with uv, running tests with pytest, linting with ruff, formatting code, type checking with mypy, or executing notebooks.
---

# Skill: Python Project Operations

Use this skill when changing dependencies, running tests, linting, formatting, type checking, or executing notebooks.

## Package Manager: uv Only

- Use \`uv\` for all dependency installation, synchronization, addition, removal, and updates.
- **Never** use pip, pip3, \`python -m pip\`, poetry, conda, pipenv, or easy_install.
- **Never** manually create or edit \`requirements.txt\`.
- Use \`uv add <package>\` when adding dependencies.
- Use \`uv add --group dev <package>\` for dev-only dependencies.
- Review diffs in \`pyproject.toml\` and \`uv.lock\` after dependency changes.

## Python Version

- Python 3.11.

## Common Commands

\`\`\`bash
uv sync                    # Install/synchronize dependencies
uv run pytest              # Run tests
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run mypy src            # Type check
uv run papermill notebooks/input.ipynb notebooks/output.ipynb  # Execute notebook
bash scripts/run_quality_checks.sh  # Run all quality checks

## Workflow

1. After modifying \`pyproject.toml\`, run \`uv sync\`.
2. After adding code, run \`uv run ruff check .\` and \`uv run ruff format .\`.
3. Before committing, run \`uv run pytest\` and \`uv run mypy src\`.
4. For notebook execution in CI or automation, prefer \`papermill\`.
```

<iframe src="https://embed.zenn.studio/card#zenn-embedded__f0d22f4744faa" frameborder="0" height="122"></iframe>

### データ取り扱いルール

大切なデータを勝手にあれこれされてはたまったものではありません。個人的にはまだ一度も AI にデータに関して「悪さ」をされた経験はなかったのですが、お守りと思って書いています。もちろんこの skill を過信せず、修正履歴は目視確認しましょう。

これは `AGENTS.md` にも最低限残しつつ、詳しくは `.github/skills/safe-data-handling/SKILL.md` に置きます。

.github/skills/safe-data-handling/SKILL.md

```
---
name: safe-data-handling
description: Use this when reading, writing, moving, copying, modifying, deleting, or generating data files — including any operation that touches data/raw, data/external, data/interim, data/processed, or outputs directories.
---

# Skill: Safe Data Handling

Use this skill before reading, writing, moving, modifying, deleting, or generating data files.

## Hard Rules

- **Never** commit raw data, credentials, API keys, tokens, or customer-level records.
- **Never** directly modify, overwrite, delete, or regenerate raw data.
- Treat \`data/raw/\` and \`data/external/\` as **immutable**.
- Write derived data to \`data/interim/\`, \`data/processed/\`, or \`outputs/\`.
- Before writing output, confirm the target path is **not** under \`data/raw/\` or \`data/external/\`.

## Recommended Workflow

1. **Identify** whether input data is raw, external, interim, processed, or output.
2. **Read** raw/external data as immutable input — never modify the source.
3. **Write** generated artifacts to a separate output path (\`data/interim/\`, \`data/processed/\`, or \`outputs/\`).
4. **Summarize** files read and written at the end of the operation.

## Directory Roles

| Directory | Role | Mutable? |
|-----------|------|----------|
| \`data/raw/\` | Original source data | No |
| \`data/external/\` | Third-party reference data | No |
| \`data/interim/\` | Intermediate transforms | Yes |
| \`data/processed/\` | Final cleaned/derived data | Yes |
| \`outputs/\` | Figures, tables, reports | Yes |

## PII and Customer-Level Records

- Do not include personally identifiable information (PII) or customer-level records in committed files.
- If analysis requires customer-level data, keep it in \`data/raw/\` (gitignored) and never commit.
- Aggregated or anonymized outputs are acceptable for \`data/processed/\` or \`outputs/\`.
- When in doubt, ask before writing customer-level data to any path.
```

`AGENTS.md` だけだと「お願い」ですが、 `scripts/check_no_raw_data_commit.py` や secret scanning （後段で出てきます）と組み合わせるとだいぶ安心できます。

### SQL のルール

SQL に関しては過去、とんでもないものを書かれた経験がありまして、いろいろ書いています [^14] 。

これも `AGENTS.md` に全部入れると重いですしSQLを書くときだけで良いので、`.github/skills/sql-analysis/SKILL.md` に書きます。

.github/skills/sql-analysis/SKILL.md

```
---
name: sql-analysis
description: Use this when writing, reviewing, or modifying SQL queries — including SELECT, CTEs, joins, aggregations, window functions, and validating query correctness or performance.
---

# Skill: SQL Analysis

Use this skill when writing, reviewing, or modifying SQL queries.

## Rules

- Use **explicit column names** — avoid \`SELECT *\` except for quick exploration.
- Use **CTEs** (Common Table Expressions) for readability and modularity.
- Add **date filters** for large fact tables to limit scan scope.
- Check **join keys** and **join cardinality** before writing joins.
- **Validate row counts** before and after joins to detect fanout or data loss.
- Avoid **implicit cross joins**.
- **Never** run \`DROP\`, \`TRUNCATE\`, \`DELETE\`, or \`UPDATE\` unless explicitly requested by the user.
- If destructive SQL is requested, propose a dry-run, backup, or transaction strategy first.

## Query Structure

\`\`\`sql
WITH base AS (
    SELECT
        column_a,
        column_b,
        event_date
    FROM schema.table_name
    WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
),
aggregated AS (
    SELECT
        column_a,
        COUNT(*) AS row_count,
        SUM(column_b) AS total_b
    FROM base
    GROUP BY column_a
)
SELECT
    column_a,
    row_count,
    total_b
FROM aggregated
ORDER BY row_count DESC;
```

## Review Checklist

Before finalizing a query, verify:

- Are **grains** (unit of analysis per row) clear?
- Are **date ranges** explicit and appropriate?
- Are **NULLs** handled (filtered, coalesced, or documented)?
- Are **duplicates** considered (distinct, dedup logic)?
- Is **join cardinality** validated (1:1, 1:N, M:N)?
- Are **business definitions** documented in comments or CTEs?
- Are **row counts** checked before and after key transformations?
- Is there any risk of **implicit cross join**?
- Are **destructive operations** absent or explicitly approved?

SQL は「それっぽいけど間違っている」ものが一番怖いです。集計結果が正しいことを別途確認したり、実行に時間がかかりすぎていないかに気をかけるのも良いですが、SQL 文そのものを見るのも重要です。

### docstring のルール

AI が書いたコードはAIも人間も読むので、私はコメントは充実している方が良いという立場です。Python の関数には必ず docstring をつけるようにしてもらっています。

これも `.github/skills/python-style/SKILL.md` に書くのが良いでしょう。

.github/skills/python-style/SKILL.md

```
---
name: python-style
description: Use this when creating, editing, or reviewing Python code — including type hints, docstrings, naming conventions, imports, error handling, and code structure.
---

# Skill: Python Style

Use this skill when creating, editing, or reviewing Python code.

## Type Hints

- Add type hints to all public function signatures.
- Use \`from __future__ import annotations\` when convenient.

## Docstrings

- Use **Google-style** docstrings for all public modules, classes, functions, and methods.
- Docstrings should describe:
  - Purpose
  - Args
  - Returns
  - Raises (if applicable)
  - Examples (when helpful)
  - Important assumptions

## Comments

- **Inline comments and explanatory comments must be written in Japanese.**
- Comments should explain non-obvious intent, assumptions, or business logic.
- Do not comment obvious syntax.

## Code Style

- Prefer small, pure functions where practical.
- Prefer explicit error handling over bare \`except\`.
- Use \`pathlib.Path\` for file paths — see [path-and-io skill](../path-and-io/SKILL.md).

## Example

\`\`\`python
from pathlib import Path

def load_config(config_path: Path) -> dict:
    """設定ファイルを読み込んで辞書として返す。

    Args:
        config_path: 設定ファイルのパス。

    Returns:
        設定内容を格納した辞書。

    Raises:
        FileNotFoundError: 指定されたパスにファイルが存在しない場合。
    """
    # JSONファイルを読み込む
    import json

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)
\`\`\`
```

### コメントのルール

日本語で書いてほしいということだけを伝えています。

```
- Inline comments and explanatory comments must be written in Japanese.
```

これは `python-style` skill に入れておけば十分だと思います。

### Pandas の廃止

AI は過去の学習データを見過ぎたせいか、pandas が大好きで、データを与えるとなんでも pandas を使っちゃいます。依存関係以外の理由で未だに `pandas` を使う理由が思いつきません。極力 [Polars](https://pola.rs/) を使いましょう。そして、遅延評価は巨大なデータ扱う我々の心強い味方です。

.github/skills/dataframe-polars/SKILL.md

```
---
name: dataframe-polars
description: Use this when performing DataFrame operations — including loading, filtering, joining, aggregating, transforming, or reshaping tabular data with polars or pandas.
---

# Skill: DataFrame Operations with Polars

Use this skill for DataFrame operations.

## Default: Polars

- Prefer **polars** for all DataFrame work.
- Prefer **LazyFrame** for loading, filtering, joins, aggregations, and transformations.
- Use eager execution when simpler and data is small.

## Pandas: Only When Required

- Use pandas **only** when required by an existing dependency, external library, or legacy code.
- If pandas is needed, keep its usage minimal and convert back to polars as soon as practical.

## Transformations

- Transformations should be reproducible and scriptable.
- Avoid manual, spreadsheet-like edits.
- Document data transformations with comments (in Japanese).

## Examples

### Lazy Scan and Filter

\`\`\`python
import polars as pl

# Parquetファイルを遅延読み込み
lf = pl.scan_parquet("data/raw/events.parquet")

# 日付フィルタと列選択
result = (
    lf.filter(pl.col("event_date") >= "2024-01-01")
    .select(["user_id", "event_type", "event_date"])
    .collect()
)
\`\`\`

### Group By and Aggregation

\`\`\`python
# ユーザーごとのイベント数を集計
summary = (
    lf.group_by("user_id")
    .agg(
        pl.col("event_type").count().alias("event_count"),
        pl.col("event_date").max().alias("last_event"),
    )
    .collect()
)
\`\`\`

### Safe Join with Row Count Check

\`\`\`python
left = pl.scan_parquet("data/processed/users.parquet")
right = pl.scan_parquet("data/processed/orders.parquet")

# 結合前の行数を確認
left_count = left.select(pl.len()).collect().item()
right_count = right.select(pl.len()).collect().item()

joined = left.join(right, on="user_id", how="left").collect()

# 結合後の行数を確認（ファンアウトの検出）
assert joined.height >= left_count, "結合で行が減少した"
print(f"left={left_count}, right={right_count}, joined={joined.height}")
\`\`\`
```

### 可視化のルール

Matplotlib で可視化しようとすると、AI は何故か `plt.figure(figsize=(14, 7))` の形式で書きます。可視化は最終的に手で [^15] 微修正することが多いので [^16] 、 `fig, ax = plt.subplots(...)` で書いてほしいのです。また、見にくいカラーマップや小さすぎるフォント、日本語フォントの文字化けなんかも人間の手での修正を最小限にしたいので、ルール化しておきます。

可視化の時だけ見れくれればよいので `.github/skills/visualization/SKILL.md` に書くことになります。

.github/skills/visualization/SKILL.md

```
---
name: visualization
description: Use this when creating, modifying, reviewing, or saving charts, figures, plots, or visual summaries — including matplotlib/seaborn code, EDA figures, report figures, dashboards, or any task involving Japanese chart labels, color palettes, or figure styling.
---

# Skill: Visualization

Use this skill whenever the user asks to:

- Plot, visualize, chart, or graph data.
- Create figures for EDA, reports, dashboards, or presentations.
- Modify or improve existing matplotlib / seaborn code.
- Save figures to disk for reports or notebooks.

## Library

- Use **matplotlib** for charts.
- Use **seaborn** alongside matplotlib for theming, palettes, and statistical plots.

## Global Setup (do this first)

At the start of any notebook or script that produces figures, set the theme once. \`font_scale\` enlarges all text elements proportionally, so individual \`fontsize=\` arguments are usually unnecessary.

\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    palette="muted",
    font_scale=1.2,
)
\`\`\`

### Japanese text

If any text (titles, labels, legends, annotations) contains Japanese, configure a CJK-capable font, **otherwise characters render as tofu (□□□)**.

Preferred approach (cross-platform):

\`\`\`python
import japanize_matplotlib  # pip install japanize-matplotlib
\`\`\`

Alternative (set an installed CJK font explicitly):

\`\`\`python
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # or "IPAexGothic", "Hiragino Sans", "Yu Gothic"
\`\`\`

## Figure Creation

- **Do not** use the stateful \`plt.figure(...)\` / \`plt.plot(...)\` style.
- **Always** create figures and axes explicitly, and prefer \`constrained_layout=True\` over \`tight_layout()\`:

\`\`\`python
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
\`\`\`

- Use the **object-oriented** API: \`ax.set_title()\`, \`ax.set_xlabel()\`, \`ax.plot()\`, etc.
- \`figsize=(10, 6)\` is a reasonable default. Adjust to content: wide time series → \`(12, 4)\`, square scatter → \`(6, 6)\`, multi-panel → scale up accordingly.

## Color Map / Color Palette

Choose by data type:

- **Categorical**: \`"muted"\`, \`"Set2"\`, \`"colorblind"\` (seaborn palettes).
- **Sequential (continuous)**: \`"viridis"\`, \`"cividis"\`, \`"mako"\` — perceptually uniform.
- **Diverging**: \`"coolwarm"\`, \`"RdBu"\`, \`"vlag"\`.
- **Forbidden**: \`"jet"\`, \`"rainbow"\` — not perceptually uniform, poor for colorblind viewers.

## Font Size

\`sns.set_theme(font_scale=1.2)\` covers most cases. Override per-element **only when needed** (e.g. a long title needs to be smaller, or one label needs emphasis):

\`\`\`python
ax.set_title("...", fontsize=18)
\`\`\`

Do **not** repeat \`fontsize=\` on every call — it is redundant when \`font_scale\` is set.

## Axis Scale Guidelines

- **Bar charts**: start y-axis at 0 (\`ax.set_ylim(bottom=0)\`). Truncated bars are misleading.
- **Line / scatter**: do **not** force y-axis to 0 — it can hide meaningful variation. Let matplotlib autoscale, or set limits based on the data range.
- **Log scale**: use \`ax.set_yscale("log")\` when data spans multiple orders of magnitude.

## Legend

- Use \`ax.legend()\` only when ≥2 series are plotted.
- If the auto-placement overlaps data, set explicitly: \`ax.legend(loc="upper left")\` or place outside: \`ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))\`.

## Saving

- Save final figures under \`outputs/figures/\`. Create the directory if needed:

\`\`\`python
from pathlib import Path
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
\`\`\`

- Use descriptive snake_case filenames: \`monthly_sales_2024h1.png\`, not \`fig1.png\`.
- DPI guidance:
  - Notebook / README / slides: \`dpi=150\`
  - Publication / print: \`dpi=300\`
- Always call \`plt.close(fig)\` after saving to free memory.

\`\`\`python
fig.savefig("outputs/figures/monthly_sales.png", dpi=150, bbox_inches="tight")
plt.close(fig)
\`\`\`

## Chart Quality Checklist

Before finalizing a chart, verify:

- [ ] **Title** clearly describes what the chart shows.
- [ ] **Axis labels** include units where applicable (e.g. "売上 (万円)", "Latency (ms)").
- [ ] **Font sizes** are readable (rely on \`font_scale=1.2\` as baseline).
- [ ] **Color palette** is perceptually uniform / colorblind-friendly (no jet/rainbow).
- [ ] **Date range** noted in title, subtitle, or annotation when relevant.
- [ ] **Sample / filter note** when data is subsetted (e.g. "n=1,234, 2024年1月〜6月").
- [ ] **Bar charts** start y-axis at 0; other chart types use sensible limits.
- [ ] **Legend** present when multiple series; placement does not overlap data.
- [ ] **Japanese text** renders correctly (japanize-matplotlib or CJK font configured).
- [ ] **Saved** to \`outputs/figures/\` with descriptive filename, followed by \`plt.close(fig)\`.

## Example

\`\`\`python
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # noqa: F401  # 日本語フォント有効化

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.bar(categories, values)
ax.set_title("月別売上推移 (2024年1月〜6月)")
ax.set_xlabel("月")
ax.set_ylabel("売上 (万円)")
ax.set_ylim(bottom=0)  # 棒グラフは0起点

fig.savefig("outputs/figures/monthly_sales_2024h1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
\`\`\`
```

### ファイルパスのルール

ローカルで分析するとき、データファイルまでのパスを埋め込まれると後々面倒になります。メンテナンスのことも考えると、文字列を使うのではなく `pathlib` も強要したいところです。

.github/skills/path-and-io/SKILL.md

```
---
name: path-and-io
description: Use this when reading from or writing to local files — including constructing file paths with pathlib, creating directories, choosing output locations, and using path utilities from src/analysis_project/paths.py.
---

# Skill: Path and I/O

Use this skill when reading from or writing to local files.

## Rules

- Use \`pathlib.Path\` for all file path operations.
- **Do not** hard-code absolute local paths.
- Prefer paths relative to repository root or configured directories.
- Use the path utilities in \`src/analysis_project/paths.py\`.
- **Do not** write outputs into raw data directories (\`data/raw/\`, \`data/external/\`).
- Create parent directories explicitly when writing outputs: \`path.parent.mkdir(parents=True, exist_ok=True)\`.
- Use descriptive file names.
- Include dates or run identifiers when outputs are time-dependent.
- Avoid overwriting existing outputs unless explicitly requested.

## Example

\`\`\`python
from analysis_project.paths import outputs_dir, ensure_parent_dir

# 出力パスを構成
output_path = outputs_dir() / "tables" / "summary_2024q1.csv"

# 親ディレクトリを作成してから書き込み
ensure_parent_dir(output_path)
df.write_csv(output_path)
```

### Notebook のルール

私は個人的にあまり使わないのですが、データ分析では notebook を使うことも多いようです。notebook は便利なのですが、AI に任せると（あるいはAIに任せなくても。。。）hidden state [^17] だらけになったり、再実行できない notebook ができたりします。他にも、再利用するロジックは src/ に切り出すようなことも指示しています。

使わない人もいると思うので、私は notebook 用の skill として分ければよいのかなと思っています。 [marimo](https://marimo.io/) のような新しい便利な notebook も登場しているので、そちらを使うような指示にしても良いかもしれません。

.github/skills/notebook-workflow/SKILL.md

```
---
name: notebook-workflow
description: Use this when creating, editing, executing, or reviewing Jupyter notebooks — including cell structure, kernel management, extracting reusable logic to src/, and ensuring notebooks are restartable.
---

# Skill: Notebook Workflow

Use this skill when creating, editing, executing, or reviewing Jupyter notebooks.

## Purpose of Notebooks

- Notebooks are for **exploration and communication**.
- Reusable logic should be extracted to \`src/analysis_project/\`.

## Rules

- Keep notebooks **restartable from a clean kernel** (Kernel → Restart & Run All must work).
- Avoid hidden state — do not rely on cells being run in a non-linear order.
- Do not include secrets or customer-level records in notebook outputs.
- Prefer saving final charts and tables to \`outputs/\`.

## Naming Convention
```

NNN\_short\_description.ipynb

```
Example: \`001_data_exploration.ipynb\`, \`002_feature_analysis.ipynb\`

## Automation

- Use **papermill** for parameterized notebook execution when automation is needed.

\`\`\`bash
uv run papermill notebooks/input.ipynb notebooks/output.ipynb -p param_name value
```

## Structure

1. **Header cell**: Title, author, date, objective.
2. **Imports**: All imports in the first code cell.
3. **Configuration**: Parameters, paths, constants.
4. **Analysis**: Exploratory or analytical cells.
5. **Summary**: Key findings and next steps.

## Cleanup Before Commit

- Clear large outputs that are not essential for review.
- Ensure no credentials or PII in cell outputs.
- Verify the notebook runs end-to-end with a fresh kernel.

## path-specific instructions も置いておく

Copilot 向けには `.github/instructions/*.instructions.md` も置けます。これは `*.py` や `*.sql` のように、ファイルパスに応じて効かせる指示です。

例えば Python ファイル向けにはこうします。

```
---
applyTo: "**/*.py"
---
Follow these skills for Python files:
- \`.github/skills/python-style/SKILL.md\` — type hints, Google-style docstrings, Japanese comments.
- \`.github/skills/dataframe-polars/SKILL.md\` — prefer polars over pandas.
- \`.github/skills/path-and-io/SKILL.md\` — use pathlib.Path, no absolute paths.
```

SQL ファイル向けにはこうします。

```
---
applyTo: "**/*.sql"
---

- Use explicit column names; avoid SELECT *.
- Use CTEs for readability.
- Add date filters for large tables.
- Validate join cardinality and row counts.
- Never run DROP, TRUNCATE, DELETE, or UPDATE unless explicitly requested.
```

`AGENTS.md` はルーター、skills は詳しい手順、path-specific instructions はファイル単位の補助、というイメージです。

## prompts も置いておく

よく使う依頼は `.github/prompts/*.prompt.md` にしておくと便利です。将来的には [^18] 、非エンジニアがこの prompts を使って分析を終わらせる日が来るかもしれませんね。

```
.github/prompts/
├── plan-analysis.prompt.md
├── review-sql.prompt.md
├── run-eda.prompt.md
├── run-modeling.prompt.md
├── summarize-analysis.prompt.md
├── prepare-pr.prompt.md
└── update-agent-docs.prompt.md
```

例えば分析計画を作る prompt を作りました（ `plan-analysis.prompt.md` ）。これは、コードを書き始める前に「目的・データソース・分析単位・主要指標・リスク・検証方法・成果物」を整理した分析計画を日本語で作らせる プロンプトです。 `docs/agent/` のデータカタログや指標定義を参照させることで、プロジェクト固有の前提を踏まえた計画にしてもらいます。

github/prompts/plan-analysis.prompt.md

```
---
agent "agent"
description: "Create an analysis plan before coding"
---
You are a data science planning assistant. Before writing any code, create a structured analysis plan.

Ask or determine the following:

1. **Objective**: What question are we trying to answer?
2. **Data sources**: What data will be used? (tables, files, APIs)
3. **Unit of analysis**: What does one row represent?
4. **Key metrics**: What metrics will be calculated? How are they defined?
5. **Risks**: What could go wrong? (data quality, leakage, bias, missing data)
6. **Validation**: How will results be validated?
7. **Outputs**: What deliverables are expected? (tables, charts, reports, models)

Format the plan in Japanese. Reference \`docs/agent/metrics-and-definitions.md\` and \`docs/agent/data-catalog.md\` for project-specific context.
```

SQL レビューもよく使います。  
`sql-analysis` skill のチェックリストに沿って SQL をレビューさせる プロンプトです。SELECT \*・日付フィルタの欠落・join のカーディナリティ・破壊的操作などを確認し、該当行と修正案を日本語で返してもらいます。

.github/prompts/review-sql.prompt.md

```
---
agent "agent"
description: "Review SQL for correctness and safety"
---
Review the provided SQL query using the checklist from \`.github/skills/sql-analysis/SKILL.md\`.

Check for:
- \`SELECT *\` usage (should use explicit columns)
- Missing date filters on large tables
- Join cardinality issues (1:1, 1:N, M:N)
- Row count validation before and after joins
- Destructive statements (DROP, TRUNCATE, DELETE, UPDATE)
- Unclear metric definitions
- NULL handling
- Duplicate risk
- Implicit cross joins

Provide feedback in Japanese with specific line references and suggested fixes.
```

## プロジェクトごとに設定するもの

以降はプロジェクトごとに設定するものです。正直反面教師かもしれませんが、短期的な検証プロジェクトなら書かないことも多いです。逆に、複数人でやるテーマや、半年以上やるテーマなら絶対に書きます。

プロジェクト固有の知識は `docs/agent/` に分けるのが良さそうです。

### Project overview

プロジェクト開始時に書くべきことのひな型は以下のような感じですかね？正解はないと思いますので、各組織で育てていってみてください。

```
# プロジェクト概要

## 目的

<!-- TODO: このプロジェクトが解決しようとしている課題を記述してください -->

## 利用者

<!-- TODO: 分析結果を利用するステークホルダーを記述してください -->

## 意思決定

<!-- TODO: この分析がどのような意思決定に使われるか記述してください -->

## スコープ外

<!-- TODO: このプロジェクトで扱わないことを明記してください -->

## 重要な前提

<!-- TODO: 分析の前提条件を記述してください -->
```

### Repository structure

これは完全にAIに出力させました。違和感はないと思っています。

```
# リポジトリ構成

各ディレクトリの役割を説明します。

| ディレクトリ | 役割 |
|-------------|------|
| \`src/analysis_project/\` | 再利用可能なPythonモジュール |
| \`notebooks/\` | 探索・分析用Jupyter Notebook |
| \`scripts/\` | CI・検証用スクリプト |
| \`tests/\` | pytest用テスト |
| \`data/raw/\` | 元データ（不変・gitignore対象） |
| \`data/external/\` | 外部データ（不変・gitignore対象） |
| \`data/interim/\` | 中間加工データ |
| \`data/processed/\` | 最終加工データ |
| \`outputs/figures/\` | グラフ・図 |
| \`outputs/tables/\` | 集計テーブル |
| \`outputs/reports/\` | レポート |
| \`docs/agent/\` | エージェント向けプロジェクト文書 |
| \`.github/skills/\` | 作業別スキルファイル |
| \`.github/instructions/\` | パス別補助指示 |
| \`.github/prompts/\` | 再利用プロンプト |
```

### Data catalog

こういった定義はAIの為だけでなく、人間のためにもかなり有用と思います。

```
# データカタログ

分析で使用するデータセットの一覧です。

## データセット一覧

<!-- TODO: 以下のテンプレートに従ってデータセットを追加してください -->

### データセット名

| 項目 | 内容 |
|------|------|
| パス | \`data/raw/xxx.parquet\` |
| 粒度 | （例: ユーザー×日） |
| 更新頻度 | （例: 日次、月次、不定期） |
| オーナー | （例: データエンジニアリングチーム） |
| 機密度 | （例: 社内限定、個人情報含む） |
| 注意点 | （例: 2023年以前はスキーマが異なる） |

## カラム定義

<!-- TODO: 主要カラムの定義を記述してください -->

| カラム名 | 型 | 説明 | 備考 |
|---------|-----|------|------|
| \`user_id\` | string | ユーザー識別子 | |
| \`event_date\` | date | イベント発生日 | |
```

### Metrics and definitions

これもプロジェクトの最初に決めるべき事柄ですね。ただ、やっていくうちに変わっていくこともあるので、サボらずにメンテしていくことが重要です。

```
# 指標定義

分析で使用する主要指標の定義です。

<!-- TODO: プロジェクト固有の指標を追加してください -->

## 指標テンプレート

### 指標名

| 項目 | 内容 |
|------|------|
| 定義 | <!-- 指標の説明 --> |
| 分子 | <!-- 分子の定義 --> |
| 分母 | <!-- 分母の定義 --> |
| 除外条件 | <!-- 除外するケース --> |
| 日付の扱い | <!-- 発生日 or 集計日 or 報告日 --> |
| 粒度 | <!-- 日次、週次、月次 --> |
| 備考 | <!-- 注意点 --> |
```

### Validation and testing

これは、pytest・ruff・mypy・notebook 検証・データ検証の実行方法と設定の置き場所をまとめたものです。それぞれのコマンドと、pyproject.toml のどのセクションに設定があるかを書いておくことで、AI が設定ファイルを探し回らずに済みます。最後に run\_quality\_checks.sh で一括実行できるようにしてあるので、コミット前にこれを回す運用にしています。運用回りの話になるので「プロジェクトごとに設定するもの」として扱っています。

docs/agent/validation-and-testing.md

```
# テスト・検証方針

## pytest

- テストは \`tests/\` ディレクトリに配置する
- \`uv run pytest\` で実行する
- テストは高速に保つ（外部依存を最小限に）

## ruff

- \`uv run ruff check .\` でリントする
- \`uv run ruff format .\` でフォーマットする
- 設定は \`pyproject.toml\` の \`[tool.ruff]\` セクション

## mypy

- \`uv run mypy src\` で型チェックする
- 設定は \`pyproject.toml\` の \`[tool.mypy]\` セクション

## Notebook検証

- Notebookがクリーンなカーネルから再実行できることを確認する
- 秘密情報がセル出力に含まれていないことを確認する

## データ検証

- \`uv run python scripts/check_no_raw_data_commit.py\` — rawデータのコミット防止
- \`uv run python scripts/check_no_sensitive_patterns.py\` — 秘密情報パターンの検出

## エージェント文書検証

- \`uv run python scripts/validate_agent_docs.py\` — 必須ファイルの存在確認

## 一括実行

\`\`\`bash
bash scripts/run_quality_checks.sh
\`\`\`
```

### Reporting guidelines

ここでは、分析レポートのテンプレートを定義しています。「結論を最初に書く」構成で、背景・目的、データと手法、結果、解釈と提言、制約・注意点、そして再現手順までを型にしています。特に再現手順（入力データ・スクリプト・出力・実行コマンド）を必ず残させるのがポイントで、後から自分や他の人が同じ結果を再現できるようにしています。詳細は analysis-reporting skill 側に置いています。もちろん、組織のルールや好みがかなりあると思うので、どんどん育てていってください。

docs/agent/reporting-guidelines.md

```
# 報告ガイドライン

分析結果の報告テンプレートです。詳細は \`.github/skills/analysis-reporting/SKILL.md\` を参照してください。

## 報告テンプレート

### タイトル

**分析者**: （名前）
**期間**: YYYY-MM-DD 〜 YYYY-MM-DD
**ステータス**: ドラフト / レビュー中 / 完了

---

### 結論

<!-- 最も重要な発見を最初に書く -->

### 分析の背景と目的

<!-- なぜこの分析を行ったか -->

### データと手法

- **データソース**: <!-- 使用したデータのパスと説明 -->
- **対象期間**: <!-- 分析対象期間 -->
- **サンプルサイズ**: <!-- レコード数 -->
- **手法**: <!-- 使用した分析手法 -->

### 結果

<!-- 事実に基づく結果を記述 -->

### 解釈と提言

<!-- 結果の解釈と推奨アクション -->

### 制約・注意点

<!-- 限界、バイアス、注意すべき点 -->

### 再現手順

- **入力データ**: \`data/raw/xxx.parquet\`
- **分析スクリプト**: \`notebooks/NNN_analysis.ipynb\`
- **出力**: \`outputs/figures/xxx.png\`, \`outputs/tables/xxx.csv\`
- **実行コマンド**: \`uv run papermill notebooks/NNN_analysis.ipynb notebooks/NNN_output.ipynb\`
```

### Security and privacy

これは、raw データ・認証情報・PII の扱いに関するプロジェクト固有のルールをまとめたものです。data/raw と data/external は.gitignore 対象かつ不変、.env はコミットしない（キー名だけ.env.example に置く）、API キーやパスワードをハードコードしない、PII や顧客レベルのレコードはコミットしない、といった事項を明記しています。safe-data-handling skill が「作業手順」なのに対して、こちらは「このプロジェクトでの取り決め」という位置づけです。あわせて検証スクリプトと CI での自動チェックにも触れ、自然言語ベースのお願いで終わらせない構成にしています。

docs/agent/security-and-privacy.md

```
# セキュリティ・プライバシー

## Raw Data

- \`data/raw/\` と \`data/external/\` は \`.gitignore\` でコミット対象外にしている。
- これらのディレクトリのデータは不変として扱う。

## Credentials・Secrets

- \`.env\` ファイルは \`.gitignore\` でコミット対象外。
- \`.env.example\` にキー名のみ記載し、実際の値は含めない。
- APIキー、トークン、パスワードをコード中にハードコードしない。
- \`python-dotenv\` を使って環境変数から読み込む。

## PII・顧客データ

- 個人を特定できる情報（PII）をコミットしない。
- 顧客レベルのレコードをコミットしない。
- 集計・匿名化したデータのみ \`data/processed/\` や \`outputs/\` に保存可能。
- 分析結果にも個人が特定されないよう注意する。

## 検証スクリプト

- \`scripts/check_no_raw_data_commit.py\` — rawデータのコミットを検知する。
- \`scripts/check_no_sensitive_patterns.py\` — 秘密情報のパターンを検知する。

## CIでの保護

- GitHub Actions CI で上記スクリプトを自動実行し、違反を検知する。
```

### Agent behavior

最後はやや毛色が違って、AI エージェントそのものに期待するふるまいを定義しています。「小さな差分にする」「仮定を明示する」「危険な操作の前に確認する」「不明点は推測せず質問する」という基本方針に加え、「やるべきこと」「やってはいけないこと」を対比で並べています。skill やルーティングは「どこを見るか」を示すものですが、こちらは「どういう姿勢で動いてほしいか」をまとめた、いわばエージェントの行動規範です。

docs/agent/agent-behavior.md

```
# エージェント行動指針

AIエージェント（GitHub Copilot等）に期待するふるまいを定義します。

## 基本方針

- **小さな差分**: 変更は小さく、レビュー可能な単位で行う。
- **仮定の明示**: 分析上の仮定は必ず明記する。
- **危険操作前の確認**: データの削除、上書き、破壊的SQL実行の前に確認する。
- **不明点の確認**: データの意味が不明な場合は推測せず質問する。

## やるべきこと

- \`AGENTS.md\` のルーティングテーブルに従い、適切なskillを参照する。
- \`docs/agent/*\` のプロジェクト固有文書を参照する。
- コード変更後は \`uv run ruff check .\` と \`uv run pytest\` を実行する。
- 分析結果にはデータ期間、フィルタ条件、サンプルサイズを明記する。

## やってはいけないこと

- \`data/raw/\` や \`data/external/\` のデータを変更・削除する。
- 秘密情報やPIIをコミットする。
- pip、conda、poetryを使ってパッケージをインストールする。
- \`SELECT *\` を本番クエリで使う。
- 根拠なくデータの因果関係を主張する。
- 過度に大きな変更を一度に行う。
```

## 本当に守らせたいものは scripts や CI にする

`AGENTS.md` や skills は便利ですが、あくまで自然言語の「お願い」であり、本当に守らせたいものは、scripts や CI にした方が良いです。

既にもういくつか記事中に登場していますが、例えば以下のようなものです。

```
scripts/
├── check_no_raw_data_commit.py
├── check_no_sensitive_patterns.py
├── run_quality_checks.sh
└── validate_agent_docs.py
```

`check_no_raw_data_commit.py` では、 `data/raw` や `data/external` に `.gitkeep` 以外のファイルを入れていないか確認します。

`check_no_sensitive_patterns.py` では、API key や token っぽい文字列が入っていないか確認します。

`validate_agent_docs.py` では、 `AGENTS.md` 、`.github/copilot-instructions.md` 、`.github/skills/*/SKILL.md` 、 `docs/agent/*` が存在するか確認します。

`run_quality_checks.sh` では、上記のスクリプトに加え ruff, mypy, pytest をまとめて実行します。

これらを CI で流せば、AI がうっかり変なファイルを追加しても気づきやすくなります。

## 小さいプロジェクト/初手ではどこまでやるか

ここまで書くと、ちょっと大げさに見えるかもしれません。

短期の検証プロジェクトなら、最初からすべて全部は必要はないと思っています。例えば、以下のような最低限の構成も考えられます。

```
.
├── AGENTS.md
├── .github/
│   ├── copilot-instructions.md
│   ├── instructions/
│   │   ├── python.instructions.md
│   │   ├── sql.instructions.md
│   │   ├── notebooks.instructions.md
│   │   ├── docs.instructions.md
│   │   └── data.instructions.md
│   └── skills/
│       ├── python-project-ops/
│       │   └── SKILL.md
│       ├── safe-data-handling/
│       │   └── SKILL.md
│       ├── sql-analysis/
│       │   └── SKILL.md
│       ├── python-style/
│       │   └── SKILL.md
│       ├── dataframe-polars/
│       │   └── SKILL.md
│       ├── visualization/
│       │   └── SKILL.md
│       ├── path-and-io/
│       │   └── SKILL.md
│       └─── notebook-workflow/
│           └── SKILL.md
├── scripts/
│   ├── check_no_raw_data_commit.py
│   ├── check_no_sensitive_patterns.py
│   ├── run_quality_checks.sh
│   └── validate_agent_docs.py
└── docs/
    └── agent/
        ├── data-catalog.md
        └── metrics-and-definitions.md
```

Skills や instructions はほとんど全部入れますが、統計・ML、レポーティングは、プロジェクトが長期化したり、複数人で触るようになってから分けても良いです。

`docs/` はプロジェクトの最初に決めるべきことや、引継ぎを想定してデータに関する記述をしっかりしておくと安心です。PR（Pull Request）の際のチェック項目にしておけば属人化して書かれないといった事態も避けられます。

## デモ

kaggleの練習用でも有名なタイタニックデータを利用して、ここで作ったプロジェクトの動作感を確認します。

<iframe src="https://embed.zenn.studio/card#zenn-embedded__d2792f87f157b" frameborder="0" height="122"></iframe>

データはローカルで `data/raw/titanic/train.csv` に置いておきます。  
`data/raw` は raw data なので、Git 管理には含めません。

ここまでで、 `AGENTS.md` 、`.github/copilot-instructions.md` 、`.github/skills/*` 、 `docs/agent/*` 、`.github/prompts/*` を用意しました。

データ分析業務に慣れている人は、ご自身でプロンプトを書いたり、自分が思い描いた方法を指示すれば良いですが、慣れていない人でも、例えば用意したカスタムプロンプトを用いて以下のように指示すればざっくり動いてくれます。

```
/plan-analysis

dataset_path: data/raw/titanic/train.csv
objective: Titanic passengers の生存要因を可視化し、Survived を予測する簡単なモデルを作る
```

```
/run-eda

データは \`data/raw/titanic/train.csv\` にあります。
```

```
/run-modeling

dataset_path: data/raw/titanic/train.csv
target: Survived
task: Predict whether each Titanic passenger survived
```

```
/summarize-analysis

topic: Titanic survival analysis demo
output_dir: outputs/
```

```
/prepare-pr
```

### 実行結果の概要

上記のプロンプトを順番に実行すると、Copilot は以下のファイルを自動生成しました。

```
src/analysis_project/
├── eda.py         # EDA 用の集計・可視化関数
├── features.py    # 特徴量エンジニアリング
├── modeling.py    # 前処理パイプライン・モデル定義・評価関数
└── paths.py       # パスユーティリティ（既存）

scripts/
├── run_titanic_eda.py       # EDA 実行スクリプト
└── run_titanic_modeling.py  # モデリング実行スクリプト

tests/
└── test_features.py   # 特徴量エンジニアリングのテスト（15ケース）
```

人間が書いたのはプロンプトだけです。  
以下、生成されたコードと出力を見ながら **`.github` 以下のルールがちゃんと効いているか** を確認していきます。

### ルールの効き具合を確認

#### 可視化ルール（visualization skill）

まず一番わかりやすいのがグラフです。 `visualization` skill には以下のようなルールを書きました。

- `fig, ax = plt.subplots(...)` を使う（ `plt.figure()` は禁止）
- `font_scale=1.2` でフォントサイズを統一する
- 日本語フォントを設定する（文字化け防止）
- 棒グラフは y 軸を 0 起点にする
- `"muted"` パレットを使う（ `"jet"` / `"rainbow"` は禁止）
- `constrained_layout=True` でレイアウト崩れを防ぐ
- `dpi=150` で保存する

実際に生成された図を見てみましょう。  
![性別ごとの生存率](https://static.zenn.studio/user-upload/4761d585ad6d-20260602.png)

このグラフを見ると、以下が確認できます。

- **日本語のタイトル・軸ラベルが文字化けしていない** （「性別ごとの生存率（Titanic）」「生存率」がちゃんと表示されている）
- **フォントサイズが十分に大きい** （ `font_scale=1.2` が効いている）
- **y 軸が 0 起点で 1.0 まで** （棒グラフのルール通り）
- **バーの上にサンプルサイズ `n=314`, `n=577` が表示されている** （データの信頼度が一目でわかる）
- **`"muted"` パレット** の落ち着いた色合い

`visualization` skill を書いていなかったころは、 `plt.figure(figsize=(14,7))` で作られたり、日本語が □□□ になったり [^19] 、y 軸が切り詰められたりしていました。毎回手で直すのは面倒なので、skill に書いておく価値があります。

もう少し複雑なグラフも見てみます。

![性別×客室クラスごとの生存率](https://static.zenn.studio/user-upload/99b9fb051eb7-20260602.png)  
性別×客室クラスのクロス集計です。hue（色分け）にクラスが使われ、凡例も配置されています。 `visualization` skill には「2 系列以上のときは `legend` を使う」と書きましたが、ちゃんと右上に凡例が出ています。

ヒストグラムも確認します。

![生存別の年齢分布](https://static.zenn.studio/user-upload/38738aa74996-20260602.png)  
生存別の年齢分布では、凡例のラベルが「死亡」「生存」と **日本語** になっています。これは `visualization` skill と `python-style` skill（コメントは日本語）の組み合わせで実現されています。ヒストグラムは 0 起点のルールの対象外なので、 `matplotlib` のオートスケールに任せています。

![相関行列](https://static.zenn.studio/user-upload/2e55fd5b008c-20260602.png)  
相関行列のヒートマップです。カラーマップに `"coolwarm"` （ダイバージングパレット）が使われており、 `visualization` skill の「ダイバージング → `coolwarm`, `RdBu`, `vlag` 」のルール通りです。 `annot=True` で相関係数が表示されているので、数値も読めます。

#### DataFrame ルール（dataframe-polars skill）

生成されたコードでは、すべての集計処理が polars で書かれています。

```
import polars as pl

df = pl.read_csv(raw_path)

# グループごとの生存率を集計
return (
    df.group_by(list(group_cols))
    .agg(
        pl.len().alias("count"),
        pl.col("Survived").sum().alias("survived"),
        pl.col("Survived").mean().alias("survival_rate"),
    )
    .sort(list(group_cols))
)
```

pandas が使われているのは sklearn に渡す直前の `.to_pandas()` 変換のみです。 `dataframe-polars` skill に「pandas は外部ライブラリの要求時のみ使用し、最小限にする」と書きましたが、その通りになっています。

#### パス管理ルール（path-and-io skill）

実装の途中に突然ハードコードされた絶対パスは一切なく、すべて `pathlib.Path` ベースです。

```
from analysis_project.paths import data_dir, outputs_dir, ensure_parent_dir

raw_path = data_dir() / "raw" / "titanic" / "train.csv"
output_path = outputs_dir() / "tables" / "missing_values.csv"
ensure_parent_dir(output_path)
```

`path-and-io` skill に書いた「 `pathlib.Path` を使う」「絶対パスを埋め込まない」「 `src/analysis_project/paths.py` のユーティリティを使う」がすべて守られています。

#### Python スタイル（python-style skill）

生成された関数にはすべて Google スタイルの docstring がついており、コメントは日本語です。

```
def missing_value_summary(df: pl.DataFrame) -> pl.DataFrame:
    """各カラムの欠損値数と欠損率を集計する。

    Args:
        df: 分析対象の DataFrame。

    Returns:
        カラム名・欠損数・欠損率を含む DataFrame。
    """
    total = df.height
    # 各カラムの null 数を集計
    null_counts = [df[col].null_count() for col in df.columns]
    ...
```

型ヒント（ `df: pl.DataFrame -> pl.DataFrame` ）もついています。

#### テストと品質チェック（python-project-ops skill）

```
$ uv run pytest
tests/test_features.py ...............    [78%]
tests/test_paths.py ....              [100%]
19 passed

$ uv run ruff check src/ scripts/
All checks passed!
```

テストでは敬称抽出、家族サイズ計算、HasCabin フラグ、リーケージ検出、build\_features の出力カラム確認など 19 ケースが検証されています。 `python-project-ops` skill に「コード変更後は `uv run ruff check .` と `uv run pytest` を実行する」と書きましたが、スクリプト生成後にちゃんと実行しています。

### 分析結果

ルールの確認ができたので、分析結果自体も簡単にまとめておきます。

#### EDA 結果

| 属性 | カテゴリ | 生存率 | 人数 |
| --- | --- | --- | --- |
| 全体 | — | 38.4% | 891 |
| 性別 | 女性 | 74.2% | 314 |
| 性別 | 男性 | 18.9% | 577 |
| 客室クラス | 1等 | 63.0% | 216 |
| 客室クラス | 2等 | 47.3% | 184 |
| 客室クラス | 3等 | 24.2% | 491 |

#### モデリング結果

使用した特徴量は 11 個で、 `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked` に加え、派生特徴量として `FamilySize`, `IsAlone`, `HasCabin`, `Title` を追加しています。

| モデル | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| ベースライン（性別のみ） | 0.777 | 0.738 | 0.652 | 0.692 | 0.753 |
| **ロジスティック回帰** | **0.832** | **0.800** | **0.754** | **0.776** | **0.872** |
| ロジスティック回帰（5-Fold CV） | 0.828 ± 0.010 | — | — | — | — |

![混同行列](https://static.zenn.studio/user-upload/227732e3f81b-20260602.png)  
*左がベースライン、右がロジスティック回帰。ロジスティック回帰では偽陰性（生存者の見逃し）が 24→17 に減少。ここでも日本語タイトルが正しく表示され、フォントサイズも十分。*

![ROC曲線](https://static.zenn.studio/user-upload/9dc865e0f77e-20260602.png)  
*ROC 曲線の凡例にも日本語が使われている（「ベースライン（性別ベース）」「ロジスティック回帰」）。2 系列以上なので凡例が表示されており、visualization skill 通り。*

![特徴量重要度](https://static.zenn.studio/user-upload/558e615f8a52-20260602.png)  
*ロジスティック回帰の係数。正（緑）が生存に寄与、負（赤）が死亡に寄与。x=0 に補助線が引かれている。*

特徴量重要度の上位は `Title_Master` （+1.28）、 `Title_Mr` （-1.27）、 `Sex_female` （+0.81）、 `Pclass` （-0.59）で、EDA で見た「女性と上位クラスの生存率が高い」という知見と整合しています。

### まとめ

「AIにほぼ丸投げ」状態でそれっぽい図やソースコードが得られました。このように、skills を設定しておくことで「pandas で書かれた」「 `plt.figure()` で書かれた」「日本語が □□□ になっている」「raw data を上書きした」「リーケージチェックがない」といった **毎回同じ修正を繰り返す手間** はなくなります。これが、`.github` 以下を整備する最大のメリットです。

繰り返しになりますが、生成されたコードや分析結果を鵜呑みにせずに、実行者が責任を持って検証しましょう。

## 結び

本記事ではデータ分析する上で、 `AGENTS.md` や Skills として設定しておくと便利な項目と具体例を紹介しました。分析スクリプトを書くためのノウハウをこのように「実行可能な形」でドキュメント化できるようになったのはとても大きく、この手の取り組みは組織の分析力の底上げ/効率化につながると確信しています。

脚注

70

48

[^1]: こういった変な比喩を使うのは個人的に好きなのですが、どうしてもAIっぽくなるので使うか毎回迷った挙句使うことにしています。

[^2]: 機械的にできることはできるだけこっちでやる方がよいです。

[^3]: リポジトリ全体に適用される指示・ルールのことです。

[^4]: リポジトリ内の特定のパスに一致するファイルのコンテキストで適用される指示のことです。

[^5]: AI coding agent に対して与える作業ルール・振る舞いの指示です。

[^6]: [github-custom-instructions](https://docs.github.com/ja/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions)

[^7]: [github-skills](https://docs.github.com/ja/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills)

[^8]: [github-prompt-files](https://docs.github.com/ja/copilot/tutorials/customization-library/prompt-files/your-first-prompt-file)

[^9]: AI コーディングを使い倒している人からすると当たり前かもしれませんが、データサイエンス観点で再整理しました。

[^10]: Rust で実装された高速に動作する Pythonの linter であり code formatter です。

[^11]: Rust で実装された高速に動作する Python のパッケージマネージャーです。

[^12]: 絶対ではないです。

[^13]: Personally Identifiable Information.「個人を特定できる情報」や「個人識別情報」のことです。そもそも組織によっては PII はAIが読みに行ってはいけないルールを課している場合があるので、各自で確認してください。

[^14]: AI に相談して書いてもらいました。

[^15]: もちろん図を直接修正するといういみではなく、Python を修正するという意味です。

[^16]: 図を顧客に見せる機会が多いので、注意深く作っています。

[^17]: 「画面上で見えているコードの状態」と「裏側（メモリ上/カーネル）で保持されている変数の状態」が食い違ってしまっている状態のことです。適当にやりたいように notebook を触っているとこのようなことになってしまいます。

[^18]: いや、もうすでにそういうタイミングなのかもしれません

[^19]: よく豆腐って言いますよね。