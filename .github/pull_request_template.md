## What changed
<!-- Files and functions added or changed, in a few bullets -->

## Why
<!-- Which module or decision this serves (see TEAM_BRIEF.md) -->

## How it was tested
<!-- Paste the summary lines of:
     pytest --cov=src --cov-branch --cov-report=term-missing
     flake8 src tests -->

## Checklist
- [ ] Branch is `feature/<module>`, PR targets `dev`
- [ ] Function names and inputs match the contract in TEAM_BRIEF.md
- [ ] No hard-coded paths, columns or thresholds (values come from config.yaml)
- [ ] Every `raise` has a test; coverage on my module is at least 90%
- [ ] flake8 prints nothing
- [ ] No data, model, log or `.env` files in *Files changed*
- [ ] Skipped integration tests, if any, name the module they wait for
