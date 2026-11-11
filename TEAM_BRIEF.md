# Team brief

How we build the pipeline. README.md says why; config.yaml holds the values; this page is the agreement every
module (and every AI chat) starts from. Changes go through the admin in a PR.

## Business and data

Business case: README.md, section 1
Use case: an early-stage investor decides which newly launched Kickstarter campaigns to contact first, using the predicted probability of reaching the funding goal
Notebook: notebooks/Kickstarter_success_Group7.ipynb (frozen)
Target: success (1 = successful, 0 = failed; other states dropped in training) | Problem type: classification | ID column: ID
Excluded columns: pledged, backers, usd pledged, usd_pledged_real (leakage: only known after the campaign ends); state (source of the target)

## Features, model, metric and split

Feature groups: log numeric: goal_usd -> median impute + missing flag -> log1p -> scale (heavily right-skewed)
Feature groups: numeric: duration_days, launch_month, launch_year, name_words -> median impute + missing flag -> scale
Feature groups: categorical: main_category, category, country -> impute "Unknown" -> one-hot, rare (< 100) grouped
Model: random_forest with n_estimators=150, max_depth=20, min_samples_leaf=20, random_state=42
Metric: ROC AUC, key in evaluate_model output: "roc_auc", quality gate: >= 0.70 on the validation split, higher is better
Split: test 0.20 and validation 0.20, both of the full dataset (60/20/20), stratified; random_state 42
Validation rules: required columns present; no duplicate ID; duration_days in [1, 92]; launch_year in [2009, 2030]; goal_usd > 0; no leakage column present; target only 0/1 with no missing (each stops the pipeline with a clear ValueError)
README follow-ups: section 1 business case -> Minghao; sections 2-5 (install, run, test, structure, owners) -> Cedric
features.py signature: get_feature_preprocessor(numeric_columns, categorical_columns, log_columns=None, rare_category_minimum=100)
Agreed extra parameters: clean_dataframe(..., *, state_column, training_states, first_valid_year, broken_country, min_goal_usd, min_investable_goal_usd, leakage_columns), all default None; validate_dataframe(..., *, duration_range, launch_year_range, forbidden_columns), all default None; evaluate_model(..., *, threshold=0.5, shortlist_share=0.1); run_inference(..., *, threshold=0.5). A rule whose parameter is None is skipped.

## Owners (team)

Owners (team): main.py, test_main.py, team setup -> Emilio (emilioauz), feature/main
Owners (team): utils.py, load_data.py, README sections 2-5 -> Cedric (Ced9999), feature/utils, feature/load-data, feature/readme
Owners (team): clean_data.py, validate.py -> Diajeng (diajengputri2602), feature/clean-data, feature/validate
Owners (team): notebook, features.py, train.py -> Arthur (CaillouMalade), feature/notebook, feature/features, feature/train
Owners (team): README section 1, evaluate.py, infer.py -> Minghao (20011011liu-create), feature/business-case, feature/evaluate, feature/infer

Shared rules: feature branches from dev, PR + 1 approval, author merges own PR; conda env from environment.yml;
pytest and flake8 must pass before every push; entry point: python -m src.main.
