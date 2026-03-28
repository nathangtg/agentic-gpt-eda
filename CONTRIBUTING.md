# Contributing to Agentic EDA

Thanks for your interest in contributing.

This project is open to bug fixes, feature ideas, docs updates, and tests.

## Quick Start

1. Fork the repository.
2. Clone your fork and create a branch.
3. Install dependencies:

```bash
uv sync
```

4. Run the app locally:

```bash
uv run streamlit run streamlit_app.py
```

5. Run tests before opening a PR:

```bash
uv run pytest -q
```

## Development Workflow

1. Create a branch from `main`.

```bash
git checkout -b feat/your-feature-name
```

2. Make focused changes.
3. Add or update tests when behavior changes.
4. Keep commits small and descriptive.
5. Push branch and open a pull request.

## What to Contribute

- Tooling improvements in `tools/`
- Agent reliability in `agents/`
- UI/UX improvements in `streamlit_app.py`
- CI/testing enhancements
- Documentation examples and onboarding guidance

## Pull Request Checklist

- [ ] Code runs locally
- [ ] Tests pass (`uv run pytest -q`)
- [ ] New behavior has tests (or reason why not)
- [ ] Docs are updated if needed
- [ ] PR description explains what changed and why

## Issue Guidelines

When reporting bugs, include:

- What you expected
- What happened
- Steps to reproduce
- Dataset shape/sample (if relevant)
- Logs or screenshots

When requesting features, include:

- Problem statement
- Proposed behavior
- Acceptance criteria

## Testing Notes

The test suite uses mocked LLM calls so contributors can run tests without API keys.

## Style Notes

- Prefer clear, explicit names.
- Keep functions small and composable.
- Avoid hidden side effects.
- Preserve backward compatibility when practical.

## Security

Do not commit secrets (API keys, tokens, credentials). Use `.env` locally.

## Community

By participating, you agree to follow the Code of Conduct in `CODE_OF_CONDUCT.md`.

We appreciate your time and contributions.
