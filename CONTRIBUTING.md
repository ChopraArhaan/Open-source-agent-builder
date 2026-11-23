# Contributing to Open Source Agent Builder

Thank you for your interest in contributing to Open Source Agent Builder!

## Code of Conduct

Please be respectful, collaborative, and constructive when contributing to discussions and code.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ChopraArhaan/Open-source-agent-builder.git
   cd Open-source-agent-builder
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

- **Branching**: Branch off `main` using descriptive names (e.g. `feature/my-feature` or `fix/issue-description`).
- **Coding Standards**:
  - Follow type annotations strictly across all functions and classes.
  - Run `ruff check .` to enforce linting and formatting.
  - Run `python -m mypy src` to verify static types.
  - Run `pytest -q` to ensure all tests pass.
- **Testing**: Add automated unit and integration tests under `tests/` for any new functionality.
- **Pull Requests**: Open a pull request against `main` describing the changes and referencing any related issues.
