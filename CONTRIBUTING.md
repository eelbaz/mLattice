# Contributing

## Running tests

```bash
# unit tests (no GPU or checkpoints required)
python3 tests/test_kernel.py

# acceptance tests (require CUDA GPU + checkpoints)
python3 tests/test_capacity.py cuda
python3 tests/test_forgetting.py cuda
python3 tests/test_realtext.py cuda
```

All tests must pass before merge.

## PR style

- One logical change per PR.
- Include the failing test case for any bug fix.
- Update `PAPER.md` numbers only from actual test runs, not manual edits.
- Write in neutral, descriptive prose.

## Code style

- Python ≥ 3.10. Type hints on public functions.
- Docstrings: state what the function does, not what it replaced.
- No commented-out code. No TODO markers without an issue link.
