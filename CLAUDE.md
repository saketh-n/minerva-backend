# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Test environment: `python test_env.py`
- Train model: `python train.py`
- Run simulation test: `python simulation.py`
- Analyze model: `python analyze_model.py [checkpoint_path]`

## Code Style Guidelines

- **Python Version**: Python 3.9 only (must match simulation bindings)
- **Imports**: Group imports by standard library, third-party, and local modules with blank lines between groups
- **Docstrings**: Use triple quotes for docstrings with Args and Returns sections
- **Naming**: 
  - Use snake_case for variables and functions
  - Use CamelCase for classes
  - Use ALL_CAPS for constants
- **Error Handling**: Use try/except blocks with specific exception types
- **Type Hints**: Optional but encouraged for function signatures
- **Formatting**: Maintain consistent indentation (4 spaces)
- **Comments**: Add comments for complex sections, but prefer self-documenting code