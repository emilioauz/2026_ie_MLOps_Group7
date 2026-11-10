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

_To be written by B._

## 3. Run

_To be written by B._

## 4. Test

_To be written by B._

## 5. Structure

_To be written by B._

## 6. Owners

_To be filled in._
