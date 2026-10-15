# Inventory Forecasting & Reorder Assistant

## Requirements
- Python 3.9+
- pandas
- matplotlib (optional, for chart output)

Install dependencies:
```bash
python -m pip install pandas matplotlib
```

## Input files
`sales.csv` columns:
`date, product_id, units_sold`

`inventory.csv` columns:
`product_id, product_name, current_stock, lead_time_days`
Optional columns: `unit_cost, supplier`

Sample CSV files are included.

## Run
```bash
python inventory_forecast.py --sales sales.csv --inventory inventory.csv
```

Use exponential smoothing:
```bash
python inventory_forecast.py --method exponential --alpha 0.3
```

Skip the chart:
```bash
python inventory_forecast.py --no-chart
```

## Output
- `reports/reorder_report.txt`: readable summary
- `reports/reorder_report.csv`: reorder list
- `reports/reorder_report.json`: structured report
- `reports/stock_vs_reorder.png`: chart (if matplotlib is installed)

## Formulas
Forecast: moving average or simple exponential smoothing.
Safety stock = z × standard deviation of daily demand × square root of lead time.
Reorder point = average daily demand × lead time + safety stock.
MAE = average absolute difference between actual and forecast demand.
MAPE = average absolute percentage error, excluding zero actual demand.

## Important assumptions
- Missing calendar dates are treated as zero sales for days in the observed history.
- MAPE is not calculated for zero actual demand values.
- The reorder quantity targets 30 days of forecast demand plus safety stock.
- The z factor is a simplified service-level assumption.
- This educational version does not model promotions, seasonality, supplier order minimums, or variable lead times.
