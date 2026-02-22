# CLAUDE.md

super-transcribe is an OpenClaw (agentic chatbot) skill that transcribes audio via two backends: Parakeet and faster-whisper. Generally, the former is faster and more accurate, while the latter has more features.

## Standards and Guidelines

### Coding Standards

- **Python**: 3.12+, FastAPI, `async/await` preferred.
- **Formatting**: `ruff` enforces 96-char lines, double quotes, sorted imports. Standard `ruff` linter rules.
- **Typing**: Strict (Pydantic v2 models preferred); `from __future__ import annotations`.
- **Naming**: `snake_case` (functions/variables), `PascalCase` (classes), `SCREAMING_SNAKE` (constants).
- **Error Handling**: Typed exceptions; context managers for resources.
- **Documentation**: Google-style docstrings for public functions/classes.
- **Testing**: Separate test files matching source file patterns.

**Error handling patterns**:

- Use typed, hierarchical exceptions defined in `exceptions.py`
- Catch specific exceptions, not general `Exception`
- Use context managers for resources (database connections, file handles)
- For async code, use `try/finally` to ensure cleanup

Example:

```python
from agents_api.common.exceptions import ValidationError

async def process_data(data: dict) -> Result:
    try:
        # Process data
        return result
    except KeyError as e:
        raise ValidationError(f"Missing required field: {e}") from e
```

### Anchor Comments

Add specially formatted comments throughout the codebase, where appropriate, for yourself as inline knowledge that can be easily `grep`ped for. 

#### Guidelines

- Use `AIDEV-NOTE:`, `AIDEV-TODO:`, or `AIDEV-QUESTION:` (all-caps prefix) for comments aimed at AI and developers.
- Keep them concise (≤ 120 chars).
- **Important:** Before scanning files, always first try to **locate existing anchors** `AIDEV-*` in relevant subdirectories.
- **Update relevant anchors** when modifying associated code.
- **Do not remove `AIDEV-NOTE`s** without explicit human instruction.
- Make sure to add relevant anchor comments, whenever a file or piece of code is:
  - too long, or
  - too complex, or
  - very important, or
  - confusing, or
  - could have a bug unrelated to the task you are currently working on.

Example:

```python
# AIDEV-NOTE: perf-hot-path; avoid extra allocations (see ADR-24)
async def render_feed(...):
    ...
```
