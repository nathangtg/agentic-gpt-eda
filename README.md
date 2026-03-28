<div align="center">

# Agentic EDA

### Open-source AI-powered Exploratory Data Analysis with execution-first workflows

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](#)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](#)
[![LangChain](https://img.shields.io/badge/LLM-LangChain-2f855a.svg)](#)
[![OSS Friendly](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](#contributing)

</div>

Agentic EDA helps analysts, data scientists, and builders go from raw dataset to actionable insights using a multi-agent workflow:

- Planner Agent: builds a structured EDA plan
- Executor Agent: executes analysis tools and captures artifacts
- Reason Agent: synthesizes findings into recommendations
- Streamlit BI Dashboard: renders KPIs, charts, and report-style outputs
- Sidebar Chat Assistant: answers dataset-aware follow-up questions

## Why This Project

Most AI data apps stop at generated text. This project focuses on execution-backed analysis:

- Uses real tool outputs, not only model guesses
- Produces chartable artifacts and feature-impact signals
- Adds robust JSON and serialization handling for reliability
- Keeps a clear plan trace so users understand what happened and why

## Features

- Upload CSV, JSON, XLSX datasets directly from the UI
- BI-style dashboard with KPI cards and visual summaries
- Auto-generated charts from analysis artifacts
- Regularized modeling (Ridge/Lasso/Logistic) for feature impact
- Data quality reporting (missingness, duplicates, schema)
- Orchestrated planner -> executor -> reasoner pipeline
- Sidebar chatbot grounded in current dataset context

## Quickstart

### 1. Clone

```bash
git clone https://github.com/<your-org>/agentic-eda.git
cd agentic-eda
```

### 2. Configure environment

Create a `.env` file in project root:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. Install dependencies

Using uv (recommended):

```bash
uv sync
```

### 4. Run app

```bash
uv run streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in terminal.

## Run With Docker

You can run the Streamlit app with optional Postgres + Redis services for persistence and memory-style workflows.

### Build and run

```bash
docker compose up --build
```

Services:

- App: http://localhost:8501
- Postgres: localhost:5432
- Redis: localhost:6379

### Stop stack

```bash
docker compose down
```

### Remove containers and volumes

```bash
docker compose down -v
```

## How It Works

1. Upload dataset in sidebar
2. Inspect BI overview and baseline charts
3. Run orchestrated analysis
4. Review:
	 - AI Plan Trace
	 - Step Execution Results
	 - Auto-Generated Visual Outputs
	 - Final Insights and Recommendations
5. Ask follow-up questions in sidebar chatbot

## Project Structure

```text
agentic-eda/
	agents/
		base.py
		planner.py
		executor.py
		reason.py
		orchestrator.py
	tools/
		common_tools.py
		exploratory_tools.py
	main.py
	streamlit_app.py
	pyproject.toml
```

## Roadmap

- Add task-specific templates by domain (finance, healthcare, growth)
- Add experiment tracking for each pipeline run
- Add export to PDF/Markdown analysis report
- Add pluggable model providers and local inference support
- Add unit/integration test suite and CI

## Contributing

Contributions are welcome and encouraged.

Ways to contribute:

- Report bugs and edge cases
- Propose improvements to tools and prompts
- Add new chart types or BI widgets
- Improve model evaluation and guardrails
- Improve docs and examples

To contribute:

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request with clear description and screenshots

If you are looking for a place to start, open an issue with the title prefix `good first issue`.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md).

## Support

If this project helps you, consider:

- Starring the repository
- Sharing it with your team/community
- Opening issues with real-world datasets and use-cases

That feedback helps prioritize roadmap work for the OSS community.
