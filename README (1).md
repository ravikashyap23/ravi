# Lights On — Problem 2 solution

## Files
- `solution.py`: complete Python implementation.
- `requirements.txt`: Python dependencies.
- `Problem_2_Solution.pdf`: solution explanation and implementation notes.

## Input data
The script expects training and test CSVs with `timestamp` and `building_id`; the training CSV must also contain `load_kwh`. It attempts to discover these files in `--data_dir`. To avoid ambiguity, you can pass explicit paths:

```bash
pip install -r requirements.txt
python solution.py --train ./data/train.csv --test ./data/test.csv --sample ./data/sample_submission.csv --output submission.csv
```

Or use automatic discovery:

```bash
python solution.py --data_dir ./data --output submission.csv
```

## Method
CatBoost quantile regression (default alpha 0.75), calendar/weather/building features, exact timestamp lags (1, 24, 168 hours), past-only rolling means/standard deviations, and recursive forecasting.

## Leakage prevention
No target interpolation or backfill is used. Training lags are joined using exact timestamp offsets. Rolling windows are closed on the left, excluding the current timestamp. Test forecasts are generated in chronological order, adding only prior predictions to each building's history.

## Important checks
- The script stops if `(building_id, timestamp)` duplicates exist; resolve them according to the challenge's data rules.
- It expects timestamps to be parseable by pandas.
- If your sample submission uses a prediction column name other than `load_kwh`, the script fills the first non-`row_id` column.
- The provided metric description suggests alpha=0.75 as a starting point (3:1 underprediction cost). Tune alpha using a chronological validation period and the official metric.
- This package does not include the competition CSV data.

## Output
`submission.csv` in sample-submission order when `row_id` is present.
