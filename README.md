# Engine Remaining Useful Life (RUL) Prediction

> 中文版：[README.zh-CN.md](README.zh-CN.md)

Predicts how many more cycles a turbofan engine can run, using NASA's C-MAPSS run-to-failure simulation data (FD001–FD004), and flags when maintenance should be scheduled before it fails. The analogy is predictive maintenance for mining haul trucks and equipment.

**Result**: on FD001's 100 official test engines, RMSE fell from 21.90 for the linear regression baseline to **13.92 cycles**, and the official score fell from 1314 to 274 (lower is better). The pipeline was extended to the other three subsets, which have multiple operating conditions and fault modes, and the whole selection process was evaluated with nested cross-validation.

The focus is not just the score but **preventing data leakage** and **being clear about where each gain comes from and what it costs**: data is split by engine, every change is checked with a paired comparison, and the mistakes made along the way are recorded as they happened.

![Maintenance alert for an engine close to failure](docs/app_alert.png)

The screenshot shows FD001 test engine 34: predicted RUL 4, true RUL 7. Its error is smaller than average, so it only illustrates the alert screen; the results tables below show overall accuracy. A screenshot of a healthy engine is in [docs/app_healthy.png](docs/app_healthy.png). The app's interface is in Chinese.

## Results

### Final models on all four subsets

| Subset | Conditions / fault modes | Default-config CV | **Nested CV** | Test RMSE (all) | Test RMSE (RUL ≤ 125) | Official score |
|---|---|---:|---:|---:|---:|---:|
| FD001 | 1 / 1 | 14.04 | **11.32 ± 0.83** | 13.92 | 12.83 | 274 |
| FD002 | 6 / 1 | 16.46 | **13.07 ± 0.67** | 25.82 | 12.02 | 6756 |
| FD003 | 1 / 2 | 14.16 | **12.15 ± 0.72** | 14.33 | 11.59 | 318 |
| FD004 | 6 / 2 | 16.55 | **13.76 ± 2.06** | 26.75 | 13.09 | 5091 |

- CV RMSE only counts cycles whose true RUL is ≤ 125. On all four subsets, nested CV beat the default configuration in 5 of 5 folds. With so few folds, this is not a significance test.
- **Nested CV**: each outer fold re-selects the feature set, hyperparameters and loss weight using only its outer training engines, then predicts the held-out engines. It estimates the selection process, not the single final model.
- **The official test set is not used for training or selection, but it is not fully independent evidence**: its results were seen during development, and they led to changing the validation metric and fixing the cap (see "Mistakes along the way").
- The CV and test sets have different RUL distributions and observed lengths, so the two RUL ≤ 125 columns are only a rough cross-check, not estimates of the same distribution.
- FD002 and FD004 have much higher all-engine RMSE mainly because 22–27% of their test engines have RUL above 125, and a model capped at 125 underestimates them by more than 40 cycles.

Details are in [notebooks/03_extensions.ipynb](notebooks/03_extensions.ipynb).

### Model comparison on FD001 (default parameters)

Models are chosen by cross-validation only (5 folds grouped by engine). All 14 experiments are in [results/FD001/summary.csv](results/FD001/summary.csv); see [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb) for details.

| Model | Features | RUL label | CV RMSE (RUL ≤ 125) | Test RMSE | Test official score |
|---|---|---|---:|---:|---:|
| **LightGBM** | **Window** | **Capped at 125** | **14.04 ± 0.82** | **14.57** | **309** |
| Random forest | Window | Capped at 125 | 14.44 ± 0.98 | 14.44 | 311 |
| Linear regression | Window | Capped at 125 | 19.39 ± 1.48 | 18.58 | 625 |
| LightGBM | Single reading | Capped at 125 | 20.76 ± 1.91 | 18.08 | 957 |
| Linear regression (baseline) | Single reading | Capped at 125 | 21.18 ± 1.73 | 21.90 | 1314 |
| LightGBM | Window | Uncapped | 24.98 ± 3.00 | 27.02 | 6625 |
| Linear regression | Single reading | Uncapped | 33.98 ± 4.76 | 31.90 | 13221 |

LightGBM and random forest are essentially tied: LightGBM is 0.40 better on CV and random forest is 0.13 better on the test set. The difference has no practical meaning.

### Where the gains come from

These are paired comparisons on the same 5 folds (change in CV RMSE). Changing only the model seed gives a CV RMSE standard deviation of about 0.03.

| Change | CV RMSE change | Folds improved (of 5) |
|---|---:|---:|
| Uncapped → capped at 125 (LightGBM, window) | -10.94 | 5/5 |
| Single reading → window features (LightGBM, capped) | -6.72 | 5/5 |
| Linear regression → LightGBM (window, capped) | -5.35 | 5/5 |
| Linear regression → LightGBM (single reading, capped) | -0.42 | 2/5 |
| Drop Nc and NRc → keep them (LightGBM, window, capped) | -1.16 | 5/5 |
| All columns → hand-picked columns (LightGBM, window, capped) | -0.07 | 3/5 |
| Default config → full selection pipeline, FD001 (nested CV) | -2.72 | 5/5 |

**Interpretation**

- **At high RUL the signal is weaker than the noise, so capping helps most**: as RUL falls from about 233 to about 125, the mean of the degradation indicator moves by only 0.44, while a single sample's standard deviation is 0.68. Without the cap, the model is asked to fit noise.
- **Single-reading noise is the bottleneck for every model**: with single readings only, switching to LightGBM barely helps. Once window features are added, tree models pull clearly ahead, because degradation accelerates and a linear model cannot fit that curve.
- **Window features account for about 89% of split gain**: means 64%, slopes 21%, standard deviations 3%. The top feature is the window mean of HPC outlet temperature (T30), which fits FD001's HPC degradation fault mode, but feature importance does not prove causation. These figures are for the default model.

### The asymmetric loss trade-off

The official score penalises late predictions (overestimating remaining life) more heavily, so I tried a weighted squared loss in which late errors are multiplied by a weight. The rule was fixed in advance: the CV score must fall by at least 10% and RMSE for RUL ≤ 125 may rise by at most 5%; among weights that meet both, pick the one with the lowest score. All four subsets ended up with weight 5.

| Subset | Test RMSE (RUL ≤ 125), symmetric → weight 5 | Test RMSE (all), symmetric → weight 5 | Official score, symmetric → weight 5 |
|---|---:|---:|---:|
| FD001 | 13.07 → 12.83 | 13.58 → 13.92 | 291 → 274 |
| FD002 | 12.71 → 12.02 | 24.42 → 25.82 | 4845 → **6756** |
| FD003 | 13.10 → 11.59 | 14.23 → 14.33 | 394 → 318 |
| FD004 | 13.98 → 13.09 | 25.43 → 26.75 | 4015 → **5091** |

**This is not a free gain.** The worse scores on FD002 and FD004 come almost entirely from engines with RUL > 125: on FD002 that group rose from 4238 to 6333, while all other engines together fell from 607 to 423. The weight shifts predictions down overall, which reduces late predictions for RUL ≤ 125, but engines that the cap already underestimates are underestimated even more.

The selection rule only looks at RUL ≤ 125, so this cost never enters the decision. The rule was set before seeing the results, and I did not overturn it because of the test results; I only record the outcome. If the use case cares about absolute remaining life during the healthy phase, this model should not be used.

## Findings from data exploration

See [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb). This section covers FD001 only.

- **7 columns are completely constant** (`setting_3` and sensors 1, 5, 10, 16, 18 and 19). FD001 has a single operating condition, so there is no reason for these to change.
- **12 sensors trend in a consistent direction**, with an absolute within-engine correlation to RUL of 0.63–0.81.
- **Nc and NRc move in different directions on different engines**: sensor_9 rises overall on 71 engines and falls on 29. Their overall correlation is weak, yet dropping them makes results worse, so they carry information.
- **There is no clear degradation knee point**: degradation accelerates continuously, so 125 is a reasonable engineering choice, not a physical constant.

## Alert threshold

The app's default alert fires when predicted RUL is ≤ 40. The table below uses the outer predictions from nested CV, so every engine's prediction comes from a full selection pipeline that never saw that engine. It is not the test set.

| Subset | Threshold | Engines never alerted | Median true RUL at first alert | Lowest RUL at first alert | Engines first alerted at RUL > 100 | False alarm rate on healthy cycles |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 40 | 0 | 40 | 27 | 0 | 0% |
| FD002 | 40 | 0 | 40 | 23 | 0 | 0% |
| FD003 | 40 | 0 | 39 | 28 | 0 | 0% |
| FD004 | 40 | 0 | 39 | 22 | 3 | 0.22% |

Other thresholds are under `alerts_nested` in `results/<subset>/final.json`.

Servicing a haul truck early only adds a little planned cost, while a breakdown mid-haul usually costs far more, so the threshold leans towards alerting early. But there is no maintenance cost model here, so the table does not prove that 40 is the best threshold. In practice the threshold should be set by maintenance intervals and downtime cost.

## Tech stack

Python 3.12 · pandas · scikit-learn · LightGBM · Jupyter · Streamlit · Altair · pytest

## Running it

The commands below are for Windows. On macOS or Linux, replace `.venv/Scripts/python` with `.venv/bin/python`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python scripts/download_data.py
```

Start the app (the first time each subset is opened, its model is trained, which usually takes under a minute):

```bash
.venv/Scripts/python -m streamlit run app/streamlit_app.py
```

You can open a specific engine in a subset directly with `http://localhost:8501/?subset=FD004&engine=5`.

Run the tests:

```bash
.venv/Scripts/python -X utf8 -m pytest -q
```

Re-running all experiments takes two steps in order, and results are written to `results/<subset>/`. The first step compares models with default parameters. The second reads the first step's results and takes over 2 hours (about 2 h 14 min and 2 h 22 min on two runs on the development laptop):

```bash
.venv/Scripts/python -X utf8 scripts/train_models.py
```

```bash
.venv/Scripts/python -X utf8 scripts/tune_models.py
```

To reproduce the exact dependency versions, use `requirements-lock.txt` instead.

## Project layout

```
├─ app/streamlit_app.py     App
├─ notebooks/
│  ├─ 01_eda.ipynb          Data exploration (FD001)
│  ├─ 02_modeling.ipynb     Model comparison and explanation (FD001, default parameters)
│  └─ 03_extensions.ipynb   Four subsets, search, asymmetric loss, alerts
├─ scripts/
│  ├─ download_data.py      Downloads the data
│  ├─ train_models.py       Default-parameter experiments
│  └─ tune_models.py        Nested CV and final selection
├─ src/rul/
│  ├─ data.py               Loading, RUL labels, splitting by engine
│  ├─ features.py           Sliding-window features, per-condition normalisation
│  ├─ models.py             Model pipelines
│  ├─ asymmetric.py         Late-weighted LightGBM, weight rule
│  ├─ experiments.py        Experiment configs, final models
│  ├─ tuning.py             Random search, selection pipeline, nested CV
│  ├─ evaluate.py           Cross-validation, test-set evaluation
│  ├─ alerts.py             Alert metrics
│  ├─ metrics.py            RMSE, official score
│  ├─ results.py            Writes result files
│  └─ inference.py          Prediction logic for the app
├─ results/                 Metrics for each subset (JSON, CSV)
├─ docs/review/             Three rounds of Codex review and how each point was handled
└─ tests/
```

## Design decisions

**Split by engine, not by row.** Readings from adjacent cycles of the same engine are nearly identical. A random row split lets the model see validation engines in advance and inflates the score. This project uses `GroupKFold`, with a test that training and validation engines never overlap, and a "spy model" that confirms training really cannot see the validation engines.

**Feature processing is fitted inside each fold.** Operating-condition clustering, normalisation, variance filtering and standardisation only look at the training fold. Constant columns are removed by the variance filter rather than hard-coded, because different subsets have different constant columns.

**Multi-condition subsets are normalised per condition first.** FD002 and FD004 have 6 operating conditions, and their raw readings are dominated by condition changes. The operating settings are clustered with KMeans, then z-scores are computed within each condition.

**Window features only look backwards.** Future readings are unknown at deployment time, and a test confirms that changing future cycles does not affect current features. The default window is 30 and the search range is 10–50; when a trajectory is shorter than the window, the available cycles are used.

**The validation metric only counts cycles with RUL ≤ 125.** This is the range that has signal and that drives maintenance decisions. All metrics compare against the uncapped true RUL; the cap only affects training labels. This metric favours capped models, so I also report full-trajectory RMSE, on which the uncapped model is actually better (37.32 vs 47.25), because a capped model never predicts above 125.

**The cap is fixed at 125 and not searched.** The main metric cannot see errors for RUL > 125, so letting it choose the cap would always push it lower. The cap trade-off is reported separately under `cap_sensitivity_cv` in `tuning.json`.

**Nested CV wraps the whole selection pipeline.** The feature set, hyperparameters and weight are all chosen within the outer training engines. If only the hyperparameter search were nested, the other choices would still have seen the validation engines and the results would be optimistic.

**Random search, not grid search.** 20 random configurations plus the default. Most parameters matter little, so random search is cheaper.

**A weighted squared loss instead of optimising the official score directly.** The official score is exponential, so its gradient explodes for large errors and training becomes unstable. The weighted squared loss has a single parameter and is easy to explain.

**No LSTM.** The project time was better spent explaining tree models and leakage prevention thoroughly. An LSTM could be future work.

## Mistakes along the way

- **The first validation method was flawed.** I randomly truncated validation engines to match the test set's trajectory lengths and got a CV RMSE of 39, while the test set gave only 21.9. Investigating showed that simulated RUL goes up to 313 and a single sample could add hundreds of thousands to the score, so the score was dominated by extreme points. I switched to the current metric. **This change was made after seeing test results**, and all later models were selected with this metric, so the test set is not fully independent.
- **With the cap included in the search, every subset picked 100.** CV improved a lot (FD001 11.69), but the official test results got worse (FD001 16.98, FD002 31.55), because the metric cannot see errors for RUL > 125. This was also found only after seeing test results. After the fix, the cap was fixed at 125, as agreed in the plan.
- **Nested CV did not wrap the full pipeline at first.** The feature set and weight were chosen outside it, after seeing all training engines. After the Codex review caught this, I merged the three steps into one `select_pipeline` that is re-run in every outer fold.
- **Hyperparameters were rounded before being read back for training.** The results file kept every float to 4 decimal places, so the final model was not actually using the parameters the search found. Model configs are now stored unrounded, with a test.
- **Weight 1 is not the same as plain LightGBM.** With column subsampling on, the custom objective uses a different random stream, so predictions differ point by point even though the error level is the same. Weights are therefore compared against weight 1 of the same wrapper.
- **Constant sensors produced noise features.** Floating-point error turned the rolling slope of a constant signal into values around 1e-14, which slipped past the variance filter. It showed up when the feature importance table had 4 more slope columns than the other kinds. After the fix, CV went from 13.83 to 14.04, so the fix did not improve the score, but those columns really were noise, so the fix stays.
- **Result files were not byte-for-byte reproducible.** Random forest's multithreaded prediction aggregation jittered at around 1e-13; rounding metrics to 4 decimal places fixed it. The first attempt at this fix did not actually take effect, which was caught only by running `diff` on the results of two runs.

A local Codex instance reviewed the project in three rounds. The reviews and how each point was handled are in [docs/review/](docs/review/) (in Chinese).

## Known limitations

- **Simulated data is not a real haul truck.** C-MAPSS is engine simulation data, and its sensor noise and degradation patterns are generated by a model. Real equipment also has missing maintenance records, changing operating conditions, broken sensors and more. The numbers here cannot be extrapolated to haul trucks; what transfers is the method.
- **The official test set has been seen**: it is a benchmark, not independent proof. A clean estimate would need the whole approach frozen and then checked on new engines.
- **The search ranges have limits**: window 50 and weight 5 are both the largest candidates. Larger values might be better and were not tested.
- **The selection rule cannot see the healthy phase**: the cost of the weight does not enter the decision.
- **Capping causes underestimation**: for engines with true RUL above 125, FD001 is underestimated by about 20 cycles on average, and FD002 and FD004 by about 45 cycles.
- **Mid-range error is still large**: on FD001, RMSE is 16.05 for true RUL 61–125, versus 4.48 for RUL ≤ 30.
- **Engines with short histories tend to be predicted late**: with few observed cycles the mean error is positive, for example +12.93 on FD004 for engines observed for 51–100 cycles.
- **Alerts can flicker**: predictions fluctuate, so after an alert fires the prediction may briefly rise back above the threshold. No smoothing is applied; the app shows a "needs attention" notice instead.
- **No maintenance cost model or significance test**: "better in 5/5 folds" only describes this split, and the alert table cannot show a lower total cost.
- **Small test sets**: FD001 and FD003 have only 100 engines each, and the official score is easily dominated by a few large errors, so small differences should not be over-interpreted.

## Data source and licence

- **Dataset**: Turbofan Engine Degradation Simulation Data Set (C-MAPSS), NASA Ames Prognostics Data Repository.
- **Paper**: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008 (the readme inside the zip also cites this paper).
- **Download**: `scripts/download_data.py` downloads a zip of about 12.4 MB from `phm-datasets.s3.amazonaws.com` and extracts it to `data/raw/`. Raw data is not committed to git.
- **Data licence**: the data.gov records for NASA C-MAPSS do not state clear licence terms, and I could not open the original NASA repository page to check. So this repository does not distribute the data; it only provides a download script and cites the source. If you want to use the data for anything beyond research, please confirm with NASA yourself.
- **Code licence**: the code in this repository is released under the [MIT licence](LICENSE). It does not cover the C-MAPSS data.
- **Data note**: the readme in the zip says FD004 has 248 training and 249 test engines, but the actual files have 249 training and 248 test engines (and 248 RUL rows). This project follows the actual files.

**Not checked against the original source**: the sensor name table comes from papers that cite Saxena 2008 (arXiv 2605.02507, 2503.07854), and the scoring function uses the definition commonly given in the C-MAPSS literature. Neither was checked directly against the original paper.
