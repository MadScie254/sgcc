# GridSentinel — SGCC Theft Detector

Electricity-theft detection on the SGCC smart-meter dataset (42,372 customers,
daily kWh from 2014-01-01 to 2016-10-31, ~8.5% labelled theft): the SMOTE+ENN +
XGBoost framework of the research proposal evaluated against its baselines, a hybrid
ensemble (a Wide & Deep CNN on the daily readings blended with XGBoost on engineered
features) served by a FastAPI backend, and a React operations console.

## The console

Operations pages work on the **population in service** (unlabelled customers) and
never show a label. The research page is the study record on the **test customers**.

| Page | What it does |
|---|---|
| Command center | Customers flagged, expected thefts among them (sum of calibrated probabilities), the scoring pipeline's last run, the investigation queue, the risk distribution, and what the threshold means on validation customers |
| Case files | Every flagged customer as a case, with status, who changed it last and the strongest signal; search waits for typing to pause |
| Case file | Consumption history (gaps shaded), the hybrid's probability and its two parts, a SHAP waterfall of the XGBoost part, the weeks of readings that raised the CNN part's score, the explanation-consistency check (LIME), the next workflow steps allowed for your role, a resolution form (finding and evidence reference), the case history with actors, and a case-file PDF |
| Pipeline | The scoring workflow (ingest → features → score → explain → route) with timings, run history and who started each run; the stages of the training run behind the served model |
| Threshold studio | Precision and recall on the validation customers, workload and expected thefts in the population, and an inspection budget (capacity, cost per visit, value per theft) with the net-value curve; supervisors publish the threshold |
| Reports & scoring | Operations PDFs (portfolio, dataset, case) and the research PDF; upload meter data or feature rows and see the scores; supervisors promote a meter-data upload to the population in service and delete files |
| Research evaluation | Test-set metrics of the served pipeline, the five study pipelines with bootstrap intervals and significance tests, the CNN and the hybrid, calibration (reliability diagram, Brier, ECE), what SMOTE+ENN does to the training data, global SHAP drivers, and the training provenance (code commit, data hash, library versions) |
| Settings | Your API key (kept for the browser tab only), who you are signed in as, service health, and, for supervisors, the audit log |

## Model

The research proposal's framework, **SMOTE+ENN + tuned XGBoost**, is compared with
standard XGBoost, random forest + SMOTE, logistic regression + SMOTE and an untreated
default XGBoost. All five use the same customers and the same 70/15/15 split. Each is
calibrated on the validation customers (Platt scaling) and scored at the threshold that
maximised F1 on its calibrated validation probabilities.

Results on the 6,356 test customers (542 thieves), scored once after every choice
(hyperparameters, early stopping, calibration, thresholds, which pipeline to serve)
was made on training and validation customers:

<!-- metrics:start -->
| Pipeline | ROC-AUC | PR-AUC (95% CI) | Recall | Precision | F1 | MCC | G-Mean |
|---|---|---|---|---|---|---|---|
| SMOTE+ENN + XGBoost (proposed) | 0.828 | 0.462 (0.422–0.502) | **0.494** | 0.389 | 0.435 | 0.379 | **0.677** |
| **XGBoost, no resampling** | **0.850** | **0.513** (0.474–0.553) | 0.467 | **0.487** | **0.476** | **0.429** | 0.667 |
| XGBoost, default settings | 0.839 | 0.488 (0.448–0.528) | 0.483 | 0.439 | 0.460 | 0.408 | 0.675 |
| Random forest + SMOTE | 0.812 | 0.411 (0.373–0.451) | 0.432 | 0.366 | 0.396 | 0.336 | 0.634 |
| Logistic regression + SMOTE | 0.754 | 0.314 (0.278–0.354) | 0.371 | 0.318 | 0.342 | 0.277 | 0.586 |
| Wide & Deep CNN (daily readings) | 0.880 | 0.538 (0.498–0.579) | 0.520 | 0.516 | 0.518 | 0.473 | 0.705 |
| **Hybrid: CNN + XGBoost (served)** | **0.908** | **0.617** (0.579–0.654) | **0.611** | **0.525** | **0.564** | **0.522** | **0.761** |

Served threshold: 0.216 (calibrated probability). The first five rows are the study's
pipelines; the CNN and the hybrid were added after the study (see "Served model" below).
<!-- metrics:end -->

**Significance** (`scripts/significance.py`, `artifacts/significance.json`): a paired
stratified bootstrap of the test customers (10,000 resamples) gives 95% intervals for
every metric and for each difference from the proposed pipeline, with Holm-corrected
p-values; McNemar's exact test compares the flag decisions.

- **Against standard XGBoost**, SMOTE+ENN is significantly worse on PR-AUC (−0.051,
  95% CI −0.076 to −0.027, p < 0.001), ROC-AUC, precision, F1 and MCC. Its recall gain
  (+0.028, −0.009 to +0.065) is not significant (p = 0.31).
- **Against default XGBoost**, no ranking difference is significant (PR-AUC −0.026,
  p = 0.07); its precision is lower (p = 0.002).
- **Against the SMOTE baselines**, it is significantly better on PR-AUC, ROC-AUC, F1,
  recall and MCC (random forest: PR-AUC +0.050, p < 0.001; logistic regression: +0.148).

Among the study's pipelines, standard XGBoost had the higher validation PR-AUC; the
console now serves the hybrid built on it (below).
`docs/proposal-alignment.md` discusses the outcome against the proposal's objectives.

**Calibration** (`artifacts/calibration.json`): resampling and class weights inflate
raw scores. On the test customers, Platt scaling fitted on validation brings the
proposed pipeline's expected calibration error from 0.093 to 0.007 and its Brier score
from 0.080 to 0.060; standard XGBoost goes from 0.010 to 0.007 and the served hybrid from
0.025 to 0.004 (Brier 0.049). Isotonic regression
is reported alongside. Calibration never changes the ranking, so ROC-AUC, PR-AUC and the
case queue are unaffected.

**What each part is worth** (`python scripts/ablation.py`, `artifacts/ablation.json`):
5-fold cross-validation on the 36,016 development customers (the test customers stay
untouched), one change at a time, each fold scored separately:

<!-- ablation:start -->
| Variant | ROC-AUC (mean ± SD) | PR-AUC (mean ± SD) |
|---|---|---|
| A. Raw readings, the proposal's 25 core features | 0.829 ± 0.009 | 0.438 ± 0.007 |
| **B. Raw readings, all 87 features (standard XGBoost)** | 0.854 ± 0.005 | 0.507 ± 0.011 |
| C. Cleaned readings (proposal section 3.7), all features | 0.847 ± 0.005 | 0.492 ± 0.011 |
| D. Cleaned + SMOTE | 0.830 ± 0.009 | 0.453 ± 0.018 |
| E. Cleaned + SMOTE+ENN (proposed) | 0.827 ± 0.006 | 0.420 ± 0.013 |
| F. B with scale_pos_weight = 1 | 0.855 ± 0.007 | 0.510 ± 0.012 |
| G. B with XGBoost's default hyperparameters | 0.832 ± 0.010 | 0.460 ± 0.011 |
<!-- ablation:end -->

- **Features are the main gain:** +0.068 PR-AUC from the 25 core features to all 87
  (gaps and zero runs, drops, trends, reversals, year-over-year ratios, change points
  and a 34-month usage profile).
- **Tuning helps:** +0.046 PR-AUC over XGBoost's default hyperparameters.
- **Class weighting adds nothing once the model is tuned:** F (no weight) and B are
  within one fold SD of each other.
- **Imputation and resampling cost ranking quality.** Filling gaps smooths away theft
  signal, and resampling gives no better ranking (see the stress tests below).
- **SMOTE+ENN does clean the training data** (Objective 1, `artifacts/resampling.json`).
  ENN removes 4,539 honest boundary rows, the Fisher ratio rises from 0.014 to 0.022,
  and the share of theft rows with honest-majority neighbours falls from 90% to 23%.
  That cleaner training data does not carry over to better test-set ranking.

**Stress tests of the result** (`scripts/resampling_study.py`, `robustness.py`,
`history_length.py`, `deep_baseline.py`, `literature_metrics.py`; figures 5.21–5.25):

- **The cleaning step, not SMOTE+ENN, explains the proposed pipeline's deficit.**
  SMOTE+ENN on *raw* readings, tuned and scored like the others, reaches a test PR-AUC of
  0.504 (95% CI 0.465–0.544): level with standard XGBoost (0.513; −0.009, −0.026 to
  +0.008, Holm p = 0.31; McNemar p = 0.90) and significantly above the proposed pipeline on
  cleaned readings (+0.042, p < 0.001).
- **No resampling setting beats none.** In 5-fold CV with standard XGBoost's hyperparameters,
  every treatment is lower than no resampling (0.507), more so the more it rebalances:
  SMOTE 0.496 (ratio 0.2) to 0.445 (1.0), SMOTE+ENN 0.453 to 0.411, SMOTE-Tomek 0.468,
  Borderline-SMOTE 0.478, ADASYN 0.462, random undersampling 0.474.
- **The ranking of the five pipelines holds on six random splits** (tuned hyperparameters
  fixed; seed 42 reproduces the table above exactly): standard XGBoost is first on all six
  (PR-AUC 0.510 ± 0.007), SMOTE+ENN on raw readings 0.479 ± 0.015, the proposed pipeline
  0.458 ± 0.008. The proposed pipeline's recall lead on the study's split does not hold
  (mean recall 0.444 against 0.471).
- **Missing readings help, but behaviour carries the signal.** Without the eight
  missing-reading features, CV PR-AUC falls from 0.507 to 0.468; consumption behaviour alone
  (gaps filled, no missingness left) still reaches 0.400, and the missing-reading features
  alone 0.333, against 0.085 by chance.
- **History matters.** With only the most recent 3, 12 or 24 months of readings, test
  PR-AUC is 0.274, 0.319 and 0.396, against 0.513 with all 34.
- **A Wide & Deep CNN in the style of Zheng et al. (2018), reading the daily series
  directly, beats the feature-based XGBoost.** Trained on 4 CPU cores in about 4 minutes
  (0.27 ms per customer), it reaches PR-AUC 0.538 on the study's split (+0.025, not
  significant alone; ROC-AUC +0.029, p = 0.003; recall +0.054, p = 0.026) and is ahead on
  all six splits: PR-AUC 0.563 ± 0.017 against 0.510 ± 0.007 (mean +0.052), and higher
  ROC-AUC, recall, F1 and MCC on every split.
- **Top of the list:** standard XGBoost has MAP@100 0.960 and precision 0.938 in the top 1%
  (CNN 0.921 and 0.891); the CNN finds more thieves further down (recall 0.563 in the top
  10% against 0.518) and is worth more under the example budget (+1,698 per 1,000
  customers, 95% CI +157 to +3,130). MAP@N depends on the test set's size and base rate,
  so it compares with published SGCC values only roughly.

**Served model: the hybrid ensemble** (`python -m src.train` with PyTorch installed;
`scripts/hybrid_study.py`, `artifacts/hybrid_study.json`, Figure 5.26). The CNN and XGBoost
make different mistakes, so the console serves a blend of the two:

- **How it is built:** both parts are fitted on the training customers and Platt-calibrated
  on validation; the hybrid is `0.55 × CNN + 0.45 × XGBoost` (weight chosen on validation
  PR-AUC over a 0.05 grid), recalibrated with Platt scaling and thresholded at the F1
  maximum, all on validation. Training serves whichever pipeline has the highest validation
  PR-AUC: hybrid 0.636, CNN 0.561, XGBoost 0.501.
- **On the test customers** it reaches PR-AUC 0.617 and ROC-AUC 0.908, and at its threshold
  finds 331 of the 542 thieves (recall 0.611) with 300 false alarms (precision 0.525).
- **Significance** (paired bootstrap, 10,000 resamples, Holm; McNemar on the flags): against
  standard XGBoost, PR-AUC +0.104 (95% CI +0.078 to +0.129), ROC-AUC +0.057, recall +0.144,
  precision +0.038, F1 +0.088, MCC +0.093, all p < 0.05 (McNemar p = 0.039). Against the CNN
  alone, PR-AUC +0.078 (+0.057 to +0.100), ROC-AUC +0.028, recall +0.090, F1 +0.046 (all
  p < 0.001); precision (+0.009) and the flag decisions (McNemar p = 0.42) do not differ
  significantly.
- **On six random splits** the CNN weight is 0.55 every time and the hybrid has the higher
  PR-AUC on all six: 0.633 ± 0.012, against 0.563 ± 0.017 (CNN) and 0.510 ± 0.007 (XGBoost).
- **Serving needs no PyTorch:** training exports the CNN to ONNX (`models/sequence.onnx`,
  299,866 parameters) and the API scores it with onnxruntime. Scoring takes about 0.16 ms
  per customer. The hybrid needs daily readings, so feature-row uploads are refused.
- **Explanations:** the case page shows SHAP for the XGBoost part and, for the CNN part,
  how much each week raised its score in log-odds (the week replaced by the customer's
  typical day). `GET /api/customers/{id}/sequence-explanation` returns the weeks.
- **A caveat:** like XGBoost's missing-reading features, the CNN uses gaps. Around the
  missing 2016-09-18 release date, September 2016 is the week that most raises the score of
  9 of the 40 highest-risk customers in service.

Without PyTorch, training falls back to the study's five pipelines and serves standard
XGBoost, as before.

**Limitation.** Customers were split at random, so the test set measures performance
on unseen customers from the same utility and period. SGCC labels are per customer and
no second labelled utility is available, so there is no evidence yet of performance on
later periods or on another utility's customers. The research page and PDF say so.

## Run the app

The trained model (`models/xgb_best.ubj`), everything it was published with
(`artifacts/`, verified against `artifacts/manifest.json` at startup) and an unlabelled
population of 3,000 test customers (`data/sgcc_demo.csv.gz`) are committed, so no
training or downloads are needed.

Requirements: Python 3.11–3.13, Node.js 20+, Git. Works on Windows, Linux and
Apple Silicon Macs (Intel Macs lack a `llvmlite` wheel for SHAP).

**1. Get the code**

```bash
git clone https://github.com/MadScie254/sgcc.git
cd sgcc
```

**2. Backend** (first terminal)

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.lock
uvicorn backend.main:app --reload
```

Using an existing Anaconda environment instead (Command Prompt):

```bat
conda activate ml_env
pip install -r requirements.lock
uvicorn backend.main:app --reload
```

Install the lock file into whichever environment you use: it pins the versions the
model and the PDF reports need (for example `xgboost==3.2.0` and `fpdf2`). If
`/api/health` reports `"reports": {"available": false}`, the environment has an old
or missing fpdf2 and the message says how to fix it. If it reports `"status": "degraded"`
with `problems`, a published model file was changed after training (git must not
rewrite line endings in `artifacts/`; `.gitattributes` prevents that): restore the
files or retrain.

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
uvicorn backend.main:app --reload
```

Wait for `Application startup complete` (the first scoring run takes a few seconds).
`http://127.0.0.1:8000/api/health` should show `"model_loaded": true`.

**3. Frontend** (second terminal, from the repo folder)

```bash
cd frontend
npm ci
npm run dev
```

Open **http://localhost:5173**. It proxies `/api` to the backend on port 8000.
Without `ENV`/`API_KEYS` set the API runs in development mode: no key is needed and you
act as the supervisor "developer". State goes to SQLite and a local folder in
`artifacts/state/`.

**Single server instead:** run `npm run build` in `frontend/`, restart uvicorn, and open
`http://127.0.0.1:8000/`; the backend serves `frontend/dist`.

**Production-like run with named keys:** create a key per person, set `ENV=production`
and `API_KEYS`, and enter your key on the console's Settings page:

```bash
python scripts/make_api_key.py amina supervisor   # prints the key (once) and its API_KEYS entry
python scripts/make_api_key.py otieno analyst
```

PowerShell: `$env:ENV="production"; $env:API_KEYS='[{"name": "amina", "role": "supervisor", "sha256": "..."}]'`.

Settings can also go in a `.env` file in the repository root (gitignored; see
`.env.example`): the API reads it at startup, and variables set in the terminal win.

Docker is optional and never required: `docker compose up --build` (fill in `.env` first).

### Windows with PostgreSQL (no Docker)

With PostgreSQL installed locally (the Windows installer from postgresql.org, version
16–18, service running on port 5432), in Command Prompt:

```bat
conda activate ml_env
pip install --isolated -r requirements.lock
set PGPASSWORD=<the postgres password chosen at installation>
python scripts/setup_database.py
uvicorn backend.main:app --reload
```

`--isolated` makes pip ignore extra package indexes in your pip settings (an unreachable
one, such as NVIDIA's, otherwise retries every package). If pip stops with
`Cannot uninstall pytz ... no RECORD file` (a package conda installed), run
`pip install --ignore-installed --no-deps pytz==2026.3.post1` and install again: pip
stops at the first failure, so nothing else was installed.

If the setup script reports that the password was rejected, it also checks ports
5433–5435: a second PostgreSQL there usually means an older installation owns 5432 and
version 18 went to 5433 (`--port 5433`). It prints how to reset a forgotten password.
`PGPASSWORD` avoids the hidden password prompt, where Ctrl+V does not paste.

`setup_database.py` logs in as the `postgres` superuser (password from `PGPASSWORD`, or
asked for) and gives the app's own role the same password (without `PGPASSWORD` it asks,
and Enter reuses the same).
It creates the role `gridsentinel`, the database `gridsentinel` for the console and
`gridsentinel_test` for the tests, creates the tables, and writes `DATABASE_URL` and
`TEST_DATABASE_URL` into `.env`. It needs neither `psql` nor Docker, and running it
again is safe. Then `http://127.0.0.1:8000/api/health` shows `"database": "postgresql"`,
and cases, runs, the operating threshold and the audit log survive restarts. Uploads
and PDFs stay on local disk (`artifacts/state/blobs`) unless `S3_BUCKET` is set.
Use `--host`, `--port` or `--admin-user` for a different server.

`pytest tests/` then runs against `gridsentinel_test` and never touches `gridsentinel`;
without `TEST_DATABASE_URL` the tests use a temporary SQLite database.

### Roles and cases

- **Analysts** view scores and explanations, review and dispatch cases, write notes,
  upload datasets, generate reports and start scoring runs.
- **Supervisors** also resolve and reopen cases, publish the operating threshold,
  make an uploaded meter-data file the population in service, and delete files.

Cases follow `new → reviewing → dispatched → confirmed | cleared` (a case under review
can also be cleared). Resolving needs a supervisor, a finding and an evidence reference
(an inspection report number, a photo id); reopening needs a reason. Every change
records who made it, and every write, download and deletion goes to the audit log
(`GET /api/audit`, Settings page).

### State, files and automation

| Setting | Effect |
|---|---|
| `DATABASE_URL` | PostgreSQL (`scripts/setup_database.py` creates it and writes this line to `.env`) for cases, case events, scoring runs, settings (operating threshold, population), the upload and report catalogues and the audit log. Unset: SQLite in `SGCC_STATE_DIR` (one instance only) |
| `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION`, `S3_PREFIX`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3-compatible object storage (AWS S3, Cloudflare R2, MinIO) for uploads and PDF reports. Unset: `SGCC_STATE_DIR/blobs` |
| `RETENTION_DAYS` | Uploads and reports older than this are deleted at startup and every 6 hours (default 90; 0 keeps them). The upload in service as the population is kept |
| `SCORING_INTERVAL_MINUTES` | Also score on a schedule (the pipeline always runs at startup and on demand) |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute per key (default 600) |

Several instances can share one database: a scoring run holds a PostgreSQL advisory
lock, and the threshold and population are read from the database with a 5-second cache.

### Reports

`POST /api/reports` with `{"kind": "portfolio"}`, `{"kind": "dataset", "dataset_id": ...}`,
`{"kind": "case", "customer_id": ...}` or `{"kind": "research"}` renders a PDF;
`GET /api/reports/{id}/pdf` downloads it. Operations reports (portfolio, dataset, case)
contain no test-set figures; the research report contains nothing else. Uploads
(`POST /api/datasets`) and batch scoring (`POST /api/predict/batch`) accept either the
SGCC layout (`CONS_NO`, optional `FLAG`, one column per day with dates written year
first, `YYYY-MM-DD` or `YYYY/M/D`) or rows carrying all 87 model features. Repeated
customer ids, ambiguous dates and missing features are rejected with the reason.

### Test the integrated model

```bash
pytest tests/                                     # unit and API tests: TEST_DATABASE_URL (from .env) or SQLite
cd frontend && npm run e2e                        # Playwright: starts the API in production mode with two named keys
python scripts/evaluate_api.py --url http://127.0.0.1:8000 --key <supervisor key>
```

`evaluate_api.py` checks, through a running API: the manifest verification, the
test-set ROC-AUC, PR-AUC and calibration in the research record, that operations carry
no labels, validation precision at the trained threshold, that every endpoint scores a
customer identically, that SHAP values add up to the raw score, the LIME consistency
check, a threshold publish round trip, and that the operations and research PDFs download.

## Train

```bash
python scripts/download_data.py        # full SGCC dataset -> data/sgcc_full.csv (175 MB, no credentials)
python -m src.train --quick            # smoke run into artifacts/quick/ (never touches the served model)
python -m src.train                    # full run: 2 × 40 Optuna trials, 5-fold CV, the CNN and the hybrid (~25 min on 4 CPU cores)
python scripts/significance.py         # paired bootstrap + McNemar on the saved test predictions (~1 min)
python scripts/ablation.py             # one-change-at-a-time ablation (~20 min)
python scripts/data_quality.py         # proposal Appendix A tables -> docs/data-quality.md
python scripts/make_thesis_figures.py  # docs/thesis-figures (from the saved predictions)
```

Stress tests of the result (after training; each writes `artifacts/<name>.json` and a figure;
`--plot-only` redraws it):

```bash
python scripts/resampling_study.py     # SMOTE+ENN on raw readings, tuned; resampling grid (~20 min)
python scripts/robustness.py           # six random splits; reliance on missing readings (~10 min)
python scripts/history_length.py       # 3 to 34 months of history (~5 min)
pip install -r requirements-research.txt
python scripts/deep_baseline.py        # Wide & Deep CNN, CPU (~5 min); --splits: the other five splits (~20 min)
python scripts/literature_metrics.py   # MAP@100/200, top-share precision, budget value (~1 min)
python scripts/hybrid_study.py         # the hybrid on six random splits (~30 min); --significance: served split only
```

On an NVIDIA GPU (for example a Quadro P2000; Windows, conda `ml_env`), add `--device cuda`
to the commands that fit XGBoost:

```bat
conda activate ml_env
python -m src.train --device cuda
python scripts/ablation.py --device cuda
```

Only XGBoost moves to the GPU; cleaning, features, SMOTE+ENN and the baselines stay on
the CPU. `significance.py` and `make_thesis_figures.py` fit nothing (they read the saved
predictions), so they need no GPU. GPU results differ slightly from CPU (floating-point
order); the committed model is the CPU run, as the proposal targets CPU-only utilities.

The protocol follows the research proposal: series cleaning (section 3.7), features
grouped as statistical, temporal, trend and anomaly (3.8), a stratified 70/15/15
customer split, SMOTE+ENN on training rows only and redone inside every CV fold (3.9),
Optuna tuning of both XGBoost pipelines on the mean PR-AUC of five separately scored
folds, with early stopping on validation (3.10), Platt calibration and threshold choice
on validation, and one scoring of the untouched test customers for every pipeline
(3.11). The servable pipeline with the highest validation PR-AUC is served: the hybrid
when PyTorch is installed (`pip install -r requirements-research.txt`), otherwise the
XGBoost pipeline with the higher validation PR-AUC.
`docs/proposal-alignment.md` maps every objective and research question to its evidence.

Training stages every output, moves it into place and writes `artifacts/manifest.json`
(SHA-256 of each file) last:

- the model and `artifacts/pipeline.json`, which freezes how serving must see a
  customer (cleaning, feature settings and list, imputation, calibration, threshold)
  and where it came from (git commit, SHA-256 of the training data, library versions);
- `metrics.json`, `calibration.json`, `tuning.json`, `learning_curves.json`,
  `resampling.json`, `preprocessing_log.json`, `feature_importance.csv`,
  `best_params.json` and the pipeline comparison;
- `artifacts/predictions/{validation,test}.csv.gz`: every pipeline's raw score and
  calibrated probability for each validation and test customer, with the label;
- a fresh unlabelled population (`data/sgcc_demo.csv.gz`).

Settings live in `config.yaml`. Training replaces the served model, so it runs from the
command line only; afterwards press **Run scoring now** on the Pipeline page (or restart
the API) to load the new model.

## Security

- Outside `ENV=development` the API requires `API_KEYS`: named keys, stored as SHA-256
  hashes, each with a role. Every `/api/*` route except `/api/health` checks the
  `X-API-Key` header, and every request is rate-limited per key. The console keeps the
  key in `sessionStorage`, so it is gone when the tab closes, and never builds it into
  the JavaScript.
- Every write, download and deletion is audit-logged with the actor.
- The API cannot retrain or replace the model, and refuses to score if a published
  file differs from the manifest.
- Uploads are capped at `MAX_UPLOAD_MB` (default 25), 200,000 rows and 2,000 columns.
  Batch-score CSVs neutralise cells that a spreadsheet would run as formulas.
- Blob keys, report ids and dataset ids are generated by the API and validated before
  any file is touched.
- API docs (`/api/docs`) are only served in development.

## Deploy (Render)

`render.yaml` creates the web service and a PostgreSQL database (`DATABASE_URL` is
wired automatically). A free web service has no persistent disk, so uploads and PDFs
need object storage: create a bucket (Cloudflare R2's free tier or AWS S3) and fill in
`S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION`, `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`. Set `API_KEYS` from `scripts/make_api_key.py`. The database
plan in the blueprint (`basic-256mb`) is a paid Render plan; the web service itself can
stay on the free plan. Each push to `main` redeploys.

## Layout

- `src/` data loading, cleaning, features, resampling, tuning, calibration, statistics, training and publishing
- `backend/` FastAPI app (`routers/`, `services/` including `db.py` and `blobstore.py`, `schemas.py`)
- `frontend/` React + Vite console, with Playwright end-to-end tests in `frontend/e2e/`
- `models/`, `artifacts/` the published model, its manifest, saved predictions and results
- `data/sgcc_demo.csv.gz` the unlabelled population the API scores by default
- `scripts/` dataset download, API keys, significance, ablation, data quality, API evaluation, thesis figures
- `concept_note/` project concept note
