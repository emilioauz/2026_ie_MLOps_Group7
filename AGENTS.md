# AGENTS.md: Group 7 build plan and rules

For every team member and every AI assistant in this repo. AI tools: read this file, `TEAM_BRIEF.md` and
`config.yaml` before writing code, and ask which module the student owns. Make sure the student at least mostly
understands what they're supposed to do and what they're about to do.

## The assignment

Turn the Kickstarter notebook into a tested, modular pipeline that runs with `python -m src.main`.
Deadline **Wed 11 Nov 2026**. The professor grades `main` as it is then, runs the tests and runs automated checks.

- **Code:** nine modules in `src/` with the professor's fixed function names (section 4). His scripts call them on
  dummy data, so the names and inputs must match exactly.
- **Settings:** everything specific to our dataset (target, columns, cleaning rules, split, model, gate) lives in
  `config.yaml`. `main.py` reads it and passes the values to the modules as parameters. The code itself stays
  dataset-agnostic: swap the dataset, change the config, and the pipeline still runs.
- **Tests:** unit and integration tests for every module, a test that runs `main()`, 90%+ branch coverage
  (aim for 95%). A unit test checks one function on a tiny fake DataFrame. An integration test checks two
  neighbouring modules together. Branch coverage means the tests run both sides of the `if`s: in practice, one
  test for every error the code can raise.

**Three things cap the mark at 4:**
1. Any test fails or is skipped, a module has no integration test, no test runs `main()`, or coverage < 90%.
2. One commit directly on `main`.
3. One member authors more than 70% of all commits.

## Who does what

| Owner | Modules (+ matching test files) | Branches |
|---|---|---|
| **Emilio** (admin) | team setup, `main.py` | `feature/team-setup`, `feature/main` |
| **Cedric** | `utils.py`, `load_data.py`, README sections 2-5 | `feature/utils`, `feature/load-data`, `feature/readme` |
| **Diajeng** | `clean_data.py`, `validate.py` | `feature/clean-data`, `feature/validate` |
| **Arthur** | notebook, `features.py`, `train.py` | `feature/notebook`, `feature/features`, `feature/train` |
| **Minghao** | README section 1 (business case), `evaluate.py`, `infer.py` | `feature/business-case`, `feature/evaluate`, `feature/infer` |

Touch only your own files. If a shared file (`config.yaml`, `TEAM_BRIEF.md`, `environment.yml`, this file) looks
wrong, flag it to the team instead of editing it.

## Workflow

`dev` is the branch everyone works from. `main` is only touched once, at the end.

1. **Branch off `dev`** for your module: `git checkout dev && git pull && git checkout -b feature/<module>`.
2. **Write the code** in `src/<module>.py` and the **tests** in `tests/test_<module>.py`. Commit often, in small
   steps, from your own GitHub account.
3. **Check** before pushing: `pytest` passes and `flake8 src tests` prints nothing.
4. **Push and open a PR into `dev`** (never `main`). A teammate reviews, then the author clicks Merge.

AI assistants never push, merge or change GitHub settings: they propose, the student commits and pushes.

**Build order:** `utils` first (everyone calls it) → all other modules in parallel → `main.py` last.
If your integration test needs a module not yet in `dev`, don't wait: skip it, and import inside the test.
```python
@pytest.mark.skip(reason="Awaiting validate in dev")
def test_integration_clean_output_passes_validate():
    from src.validate import validate_dataframe
    ...
```
When the neighbour merges: `git merge dev` into your branch, remove the skip, push. No skip may reach `main`.

## Rules for every module

**Code**
- No dataset names, column names, numbers or paths in `src/`. They come from `config.yaml` through `main.py`.
- Extra parameters always have a default (`None` = rule off), so the professor's calls with only the contract
  inputs still work.
- Support both `problem_type="classification"` and `"regression"` where the contract takes it; anything else
  raises `ValueError`. Our config uses classification.
- `logger = logging.getLogger(__name__)` in each module, lazy format: `logger.info("Rows: %s", n)`. No `print`.
- Type hints and a docstring on every public function. Max 120 characters per line.
- A clear `ValueError` (or the error listed below) for bad input. Never fix data silently.
- `.fit()` only in `train.py`. Never use `pledged`, `backers`, `usd pledged`, `usd_pledged_real` or `state` as
  inputs (leakage).
- Inside `src/`, import modules, not functions: `from src import utils`, then `utils.load_csv(...)`.

**Tests**
- At least 2 unit tests (`test_unit_...`) and 1 integration test (`test_integration_...`) per module.
- One test for every `raise`.
- Tiny synthetic data only (5-40 rows, ~100 for the `main()` test), or `tmp_path`. Never the real CSV.

**Git**
- Never commit on `dev` or `main`. Never commit data, models, reports, logs or `.env`.

## The modules

The function names and inputs are fixed by the professor.

| Module | Function | What it does | Errors to raise |
|---|---|---|---|
| `utils.py` | `load_csv`, `save_csv`, `save_model`, `load_model`, `load_config` | Reads and writes files; creates folders | Missing file → `FileNotFoundError`; bad YAML → `ValueError` |
| `load_data.py` | `load_raw_data` | Reads the raw CSV, changes nothing | Missing path, folder, empty file |
| `clean_data.py` | `clean_dataframe` | Applies the notebook's cleaning rules. Training mode filters rows and builds the target; inference mode keeps every row | Not a DataFrame, empty, missing column |
| `validate.py` | `validate_dataframe` | Checks the data, returns `True`, never fixes | One `ValueError` per failed rule |
| `features.py` | `get_feature_preprocessor` | Returns the **unfitted** recipe: impute, log, scale, one-hot | Empty or overlapping column lists |
| `train.py` | `train_model` | Builds the model from config and fits the pipeline. The only `.fit()` | Unknown model or problem type, empty or mismatched data |
| `evaluate.py` | `evaluate_model` | Returns metrics as floats: `roc_auc` and the shortlist hit rate | Empty or mismatched data, no `predict_proba` (`TypeError`) |
| `infer.py` | `run_inference` | Returns `prediction` (and `probability`) with the input's index | Empty input, missing columns |
| `main.py` | `split_data`, `main` | Loads config, runs every step, reports the gate, saves outputs. No logic of its own | Bad split sizes; `main()` exits with code 1 on error |

Exact signatures:
```python
# utils.py
def load_csv(filepath: Path) -> pd.DataFrame
def save_csv(df: pd.DataFrame, filepath: Path) -> None              # creates parent folders, index=False
def save_model(model, filepath: Path) -> None                       # joblib, creates parent folders
def load_model(filepath: Path)
def load_config(filepath: Path) -> dict

# load_data.py
def load_raw_data(raw_data_path: Path) -> pd.DataFrame

# clean_data.py   target_column given = training mode; None = inference mode
def clean_dataframe(df_raw: pd.DataFrame, target_column: str | None = None, *,
                    state_column=None, training_states=None, first_valid_year=None, broken_country=None,
                    min_goal_usd=None, min_investable_goal_usd=None, leakage_columns=None) -> pd.DataFrame

# validate.py
def validate_dataframe(df: pd.DataFrame, required_columns: list[str], target_column: str | None = None, *,
                       duration_range=None, launch_year_range=None, forbidden_columns=None) -> bool

# features.py
def get_feature_preprocessor(numeric_columns: list[str], categorical_columns: list[str],
                             log_columns: list[str] | None = None,
                             rare_category_minimum: int = 100) -> ColumnTransformer

# train.py   model_params = {"type": "logistic_regression" | "random_forest", "params": {...}}
def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer,
                problem_type: str, model_params: dict | None = None, random_state: int = 42) -> Pipeline

# evaluate.py   keys: roc_auc, accuracy, precision, recall, f1, precision_top_share
def evaluate_model(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str, *,
                   threshold: float = 0.5, shortlist_share: float = 0.1) -> dict[str, float]

# infer.py   columns: prediction (0/1), plus probability if include_proba
def run_inference(model: Pipeline, X_infer: pd.DataFrame, include_proba: bool = False, *,
                  threshold: float = 0.5) -> pd.DataFrame

# main.py   returns X_train, X_val, X_test, y_train, y_val, y_test
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float,
               random_state: int, stratify: bool) -> tuple
def main() -> None
```

## Final checks (what the professor runs)

Emilio runs these from a fresh clone of `dev` before merging into `main`:
```bash
pytest --cov=src --cov-branch                                            # all pass, none skipped, 90%+ (aim 95%)
flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests  # prints nothing
pylint src                                                               # 9.5 or more
radon cc -s -a src                                                       # average A
detect-secrets scan                                                      # no results
python -m src.main                                                       # writes outputs, reports the gate MET
git ls-files data models logs .env                                       # prints nothing
```
Plus: metric, threshold, direction and target identical in README, `TEAM_BRIEF.md`, `config.yaml` and the code;
every import listed in `environment.yml`; every member owns a module; nobody above 70% of commits.
