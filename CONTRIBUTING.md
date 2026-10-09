# Contributing

## Workflow

1. Open an issue describing the change (use the *Bug report* or *Feature / research task* template).
2. Create a branch from `main`: `git switch -c feature/short-name`.
3. Make small commits with clear messages, e.g. `feat: add late-submission feature`,
   `fix: handle empty VLE table`, `test: check cut-off leakage`, `docs: update README`.
4. Run the checks locally:
   ```bash
   ruff format . && ruff check . && pytest --cov
   ```
5. Push the branch and open a pull request. CI must be green before merging.

## Rules for research code

- Never commit the OULAD data or any personal student data (`data/` is in `.gitignore`).
- Every feature must use only information available **before** the cut-off day; add a test for it.
- Fix random seeds so results can be reproduced.
- Record any change that affects results in `CHANGELOG.md`.
