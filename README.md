# Kickstarter Launch-Day Success Predictor (Group 7)

## 1. Business case

**Decision.** An early-stage investor wants to know, on launch day, which Kickstarter
campaigns will reach their funding goal. The investor contacts the most promising
creators first, so the model's job is to rank new campaigns by their chance of success.

**Data.** Kaggle *Kickstarter Projects* (`ks-projects-201801.csv`).

**Scope.** Finished campaigns only (state `successful` or `failed`), launched from 2009
onwards, with a goal of at least 10,000 USD. Smaller campaigns are not worth an
investor's time.

**Target.** `success` = 1 if the campaign was funded (`successful`), 0 if it `failed`.

**Inputs.** Only what is known at launch: goal, duration, launch month and year, number
of words in the name, category, main category and country. Pledged amounts and backer
counts (`pledged`, `backers`, `usd pledged`, `usd_pledged_real`) are dropped because they
are only known after the campaign ends and would leak the outcome.

**Model.** Random forest classifier, chosen in the notebook.

**Success metric and gate.**

| Item | Value |
|---|---|
| Metric | ROC AUC (`roc_auc`) on the validation split |
| Direction | Higher is better |
| Gate | Met when ROC AUC >= 0.70 |
| Notebook result | 0.737 (gate met) |

ROC AUC measures how well the model ranks successful campaigns above failed ones,
which is exactly what a shortlist needs.

**Business result.** Ranking campaigns by predicted probability and keeping the top 10%
(`shortlist_share: 0.10`) gives a shortlist that succeeds 58.6% of the time, against
28.4% for random picks: about 2x more hits for the same outreach effort.

**How the output is used.** For each new campaign the pipeline returns a probability of
success and a 0/1 prediction (cutoff 0.29, F1-optimal on validation). The investor
works down the list from the highest probability.

## 2. Install

Requirements: [Miniconda](https://docs.anaconda.com/miniconda/) and Git. Everything else, including Python 3.12, is
installed from `environment.yml`.

```bash
git clone https://github.com/emilioauz/2026_ie_MLOps_Group7.git
cd 2026_ie_MLOps_Group7
conda env create -f environment.yml
conda activate group7-mlops
```

After a teammate changes `environment.yml`, run `conda env update -f environment.yml --prune`.

**Data (not in Git).** Download *Kickstarter Projects* by kemical from
[Kaggle](https://www.kaggle.com/datasets/kemical/kickstarter-projects), unzip it and put `ks-projects-201801.csv` in
`data/raw/`. The older file in the same download, `ks-projects-201612.csv`, is not used.

No secrets are needed at this stage. `.env.example` lists the names of any future ones; copy it to `.env`
(git-ignored) to fill in values.

## 3. Run

From the repository root, with the environment active:

```bash
python -m src.main
```

One command runs every step in order: **load → clean → validate → split → features → train → evaluate → save →
infer**.

| Step | Module | What happens | Output |
|---|---|---|---|
| Load | `load_data.py` | Reads `data/raw/ks-projects-201801.csv` unchanged | in memory |
| Clean | `clean_data.py` | Keeps successful and failed campaigns with a goal of at least 10,000 USD, builds the `success` target, drops columns only known after the campaign ends | `data/processed/clean.csv` |
| Validate | `validate.py` | Checks columns, ranges and target; stops with a clear error on bad data | log |
| Split | `main.py` | 60 / 20 / 20 train, validation and test, stratified, `random_state` 42 | in memory |
| Features, train | `features.py`, `train.py` | Builds the preprocessing recipe and fits the random forest on the training split only | `models/model.joblib` |
| Evaluate | `evaluate.py` | Scores the validation split and reports the quality gate, ROC AUC ≥ 0.70, as **MET** or **NOT MET** | `reports/metrics.json` |
| Infer | `infer.py` | Scores new campaigns from `data/inference/new_campaigns.csv` (the held-out test split if that file is absent) | `reports/predictions.csv` |

Every setting (paths, target, cleaning rules, split, model, gate) lives in `config.yaml`; `src/` holds no dataset
names, columns or numbers. Each run logs to the terminal and to `logs/pipeline.log`. If a step fails, the run stops
with the reason and exits with code 1.

## 4. Test

Run these from the repository root before every pull request. They are the checks used for grading.

```bash
pytest --cov=src --cov-branch --cov-report=term-missing                  # all pass, none skipped, >= 90 % branch coverage
flake8 --isolated --max-line-length=120 --extend-ignore=E203 src tests  # prints nothing
pylint src                                                               # 9.5 or more
radon cc -s -a src                                                       # average A
detect-secrets scan                                                      # "results": {}
git ls-files data models logs .env                                       # prints only the folder README.md files
```

Each module in `src/` has a matching `tests/test_<module>.py` with at least two unit tests (one function on tiny
synthetic data), one integration test (two neighbouring modules together) and a test for every error the module can
raise. `tests/test_main.py` runs the whole pipeline on about 100 synthetic rows. Tests never read the real CSV.

## 5. Structure

```text
.
├── README.md                 # business case, install, run, test, structure, owners
├── TEAM_BRIEF.md             # one page of technical decisions
├── AGENTS.md                 # build plan and rules for the team and AI assistants
├── config.yaml               # every non-secret setting
├── environment.yml           # Python 3.12 and every import
├── pytest.ini  .flake8  .gitignore  .env.example
├── .github/pull_request_template.md
├── notebooks/
│   └── Kickstarter_success_Group7.ipynb   # frozen: the analysis the pipeline reproduces
├── src/
│   ├── utils.py              # read/write CSV, save/load model, load config
│   ├── load_data.py          # read the raw CSV unchanged
│   ├── clean_data.py         # cleaning rules and the success target
│   ├── validate.py           # data quality gate: checks, never fixes
│   ├── features.py           # unfitted preprocessing recipe
│   ├── train.py              # the only place that fits
│   ├── evaluate.py           # metrics on the validation split
│   ├── infer.py              # predictions on new campaigns
│   └── main.py               # orchestrator: python -m src.main
├── tests/                    # test_<module>.py for every module, plus test_main.py
├── data/                     # raw/ processed/ inference/ (only README files in Git)
└── models/  reports/  logs/  # created by the pipeline, not in Git
```

## 6. Owners

| Owner | GitHub | Modules and files |
|---|---|---|
| Emilio (admin) | [emilioauz](https://github.com/emilioauz) | Team setup (`config.yaml`, `environment.yml`, `TEAM_BRIEF.md`, `AGENTS.md`, `pytest.ini`, `.flake8`), `main.py` |
| Cedric | [Ced9999](https://github.com/Ced9999) | `utils.py`, `load_data.py`, README sections 2–5 |
| Diajeng | [diajengputri2602](https://github.com/diajengputri2602) | `clean_data.py`, `validate.py` |
| Arthur | [CaillouMalade](https://github.com/CaillouMalade) | Notebook, `features.py`, `train.py` |
| Minghao | [20011011liu-create](https://github.com/20011011liu-create) | README section 1 (business case), `evaluate.py`, `infer.py` |

Each owner also owns the matching test files. All work goes through `feature/<module>` branches and reviewed pull
requests into `dev`; `dev` merges into `main` once, for the release.
