#!/usr/bin/env python3
"""
Inventory Forecasting & Reorder Assistant

Input CSV files:
1. sales.csv: date, product_id, units_sold
2. inventory.csv: product_id, product_name, current_stock, lead_time_days,
                  unit_cost (optional), supplier (optional)

Creates a prioritized reorder report, CSV export, JSON summary, and a demand
forecast chart when matplotlib is installed.
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd


REQUIRED_SALES = {"date", "product_id", "units_sold"}
REQUIRED_INVENTORY = {
    "product_id", "product_name", "current_stock", "lead_time_days"
}


def load_and_clean_data(sales_path, inventory_path):
    """Load CSV files, normalize columns, and clean invalid/duplicate rows."""
    sales = pd.read_csv(sales_path)
    inventory = pd.read_csv(inventory_path)

    sales.columns = sales.columns.str.strip().str.lower()
    inventory.columns = inventory.columns.str.strip().str.lower()

    missing_sales = REQUIRED_SALES - set(sales.columns)
    missing_inventory = REQUIRED_INVENTORY - set(inventory.columns)

    if missing_sales:
        raise ValueError(f"Sales CSV missing columns: {sorted(missing_sales)}")
    if missing_inventory:
        raise ValueError(
            f"Inventory CSV missing columns: {sorted(missing_inventory)}"
        )

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales["units_sold"] = pd.to_numeric(sales["units_sold"], errors="coerce")
    sales = sales.dropna(subset=["date", "product_id", "units_sold"])
    sales = sales[sales["units_sold"] >= 0]
    sales["product_id"] = sales["product_id"].astype(str).str.strip()
    sales = sales[sales["product_id"] != ""]
    sales = sales.drop_duplicates()

    inventory["product_id"] = inventory["product_id"].astype(str).str.strip()
    inventory["product_name"] = inventory["product_name"].fillna("Unknown product")
    inventory["current_stock"] = pd.to_numeric(
        inventory["current_stock"], errors="coerce"
    )
    inventory["lead_time_days"] = pd.to_numeric(
        inventory["lead_time_days"], errors="coerce"
    )

    inventory = inventory.dropna(
        subset=["product_id", "current_stock", "lead_time_days"]
    )
    inventory = inventory[
        (inventory["product_id"] != "")
        & (inventory["current_stock"] >= 0)
        & (inventory["lead_time_days"] >= 0)
    ]
    inventory = inventory.drop_duplicates(subset=["product_id"], keep="last")

    if "unit_cost" not in inventory.columns:
        inventory["unit_cost"] = 0.0
    inventory["unit_cost"] = pd.to_numeric(
        inventory["unit_cost"], errors="coerce"
    ).fillna(0.0)

    if "supplier" not in inventory.columns:
        inventory["supplier"] = "Not specified"
    inventory["supplier"] = inventory["supplier"].fillna("Not specified")

    if inventory.empty:
        raise ValueError("Inventory file contains no valid product rows.")

    logging.info(
        "Cleaned input: %d sales rows and %d inventory products",
        len(sales), len(inventory)
    )
    return sales, inventory


def build_daily_demand(sales, product_id, start_date, end_date):
    """Create a continuous daily demand series, including zero-sales days."""
    product_sales = sales[sales["product_id"] == product_id]
    if product_sales.empty:
        return pd.Series(dtype=float)

    daily = product_sales.groupby(
        product_sales["date"].dt.normalize()
    )["units_sold"].sum()

    calendar = pd.date_range(start=start_date, end=end_date, freq="D")
    return daily.reindex(calendar, fill_value=0).astype(float)


def forecast_demand(daily_demand, method="moving_average", window=7, alpha=0.3):
    """Estimate next-day demand with a moving average or simple exponential smoothing."""
    values = daily_demand.astype(float).tolist()

    if not values:
        return 0.0, 0.0

    if method == "moving_average":
        recent = values[-window:]
        forecast = sum(recent) / len(recent)
    elif method == "exponential":
        forecast = values[0]
        for actual in values[1:]:
            forecast = alpha * actual + (1 - alpha) * forecast
    else:
        raise ValueError(f"Unknown forecasting method: {method}")

    std_dev = float(daily_demand.std(ddof=1)) if len(values) > 1 else 0.0
    if math.isnan(std_dev):
        std_dev = 0.0

    return max(0.0, float(forecast)), max(0.0, std_dev)


def calculate_reorder_metrics(
    current_stock, average_daily_demand, demand_std, lead_time_days,
    service_z=1.65
):
    """
    Calculate safety stock and reorder point.

    Safety stock = z * daily demand standard deviation * sqrt(lead time)
    Reorder point = average daily demand * lead time + safety stock
    """
    lead_time = max(0.0, float(lead_time_days))
    average_daily_demand = max(0.0, float(average_daily_demand))
    demand_std = max(0.0, float(demand_std))

    safety_stock = service_z * demand_std * math.sqrt(lead_time)
    lead_time_demand = average_daily_demand * lead_time
    reorder_point = lead_time_demand + safety_stock

    # A simple target stock level: 30 days of forecast demand plus safety stock.
    target_stock = average_daily_demand * 30 + safety_stock
    reorder_quantity = max(0, math.ceil(target_stock - float(current_stock)))

    return {
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": math.ceil(safety_stock),
        "reorder_point": math.ceil(reorder_point),
        "reorder_quantity": reorder_quantity,
    }


def calculate_forecast_error(actual, predicted):
    """Return MAE and MAPE; MAPE ignores actual zero-demand observations."""
    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, dtype=float)

    if len(actual) == 0 or len(actual) != len(predicted):
        return {"mae": None, "mape_percent": None}

    mae = float((actual - predicted).abs().mean())
    nonzero = actual != 0
    if nonzero.any():
        mape = float(
            ((actual[nonzero] - predicted[nonzero]).abs()
             / actual[nonzero].abs()).mean() * 100
        )
    else:
        mape = None

    return {
        "mae": round(mae, 3),
        "mape_percent": round(mape, 3) if mape is not None else None,
    }


def evaluate_forecast(daily_demand, method, window, alpha):
    """One-step rolling evaluation using past data only."""
    values = daily_demand.astype(float).tolist()
    actual, predicted = [], []

    for index in range(1, len(values)):
        history = pd.Series(values[:index])
        forecast, _ = forecast_demand(
            history, method=method, window=window, alpha=alpha
        )
        actual.append(values[index])
        predicted.append(forecast)

    return calculate_forecast_error(actual, predicted)


def create_reorder_report(
    sales, inventory, method="moving_average", window=7, alpha=0.3,
    service_z=1.65, slow_days=30
):
    """Forecast each product, calculate reorder points, and rank priorities."""
    today = sales["date"].max().normalize() if not sales.empty else pd.Timestamp.today().normalize()
    history_start = sales["date"].min().normalize() if not sales.empty else today
    # Include all calendar days so days with no sales count as zero demand.
    daily_dates = pd.date_range(history_start, today, freq="D")

    results = []
    for _, product in inventory.iterrows():
        product_id = str(product["product_id"])
        demand = build_daily_demand(sales, product_id, history_start, today)

        if demand.empty and len(daily_dates):
            demand = pd.Series(0.0, index=daily_dates)

        avg_daily, demand_std = forecast_demand(
            demand, method=method, window=window, alpha=alpha
        )
        stock = float(product["current_stock"])
        lead_time = float(product["lead_time_days"])

        metrics = calculate_reorder_metrics(
            stock, avg_daily, demand_std, lead_time, service_z
        )

        if demand.empty or float(demand.sum()) == 0:
            status = "SLOW-MOVING / NO RECENT SALES"
        elif stock <= metrics["reorder_point"]:
            status = "REORDER NOW"
        elif stock <= metrics["reorder_point"] * 1.25:
            status = "WATCH"
        else:
            status = "HEALTHY"

        # Approximate days until stockout using forecast demand.
        days_until_stockout = (
            round(stock / avg_daily, 1) if avg_daily > 0 else None
        )
        stockout_prone = (
            days_until_stockout is not None
            and days_until_stockout <= lead_time
        )

        error = evaluate_forecast(demand, method, window, alpha)
        results.append({
            "product_id": product_id,
            "product_name": str(product["product_name"]),
            "supplier": str(product["supplier"]),
            "current_stock": round(stock, 2),
            "lead_time_days": round(lead_time, 2),
            "forecast_daily_demand": round(avg_daily, 3),
            "demand_std_dev": round(demand_std, 3),
            **metrics,
            "days_until_stockout": days_until_stockout,
            "stockout_prone": stockout_prone,
            "status": status,
            "unit_cost": round(float(product["unit_cost"]), 2),
            "estimated_order_cost": round(
                metrics["reorder_quantity"] * float(product["unit_cost"]), 2
            ),
            "mae": error["mae"],
            "mape_percent": error["mape_percent"],
        })

    priority = {"REORDER NOW": 0, "WATCH": 1,
                "SLOW-MOVING / NO RECENT SALES": 2, "HEALTHY": 3}
    results.sort(
        key=lambda item: (
            priority[item["status"]],
            not item["stockout_prone"],
            -item["reorder_quantity"],
        )
    )
    return results


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, rows, method):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "forecast_method": method,
        "product_count": len(rows),
        "reorder_now_count": sum(r["status"] == "REORDER NOW" for r in rows),
        "stockout_prone_count": sum(r["stockout_prone"] for r in rows),
        "products": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_text_report(path, rows, method):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "INVENTORY FORECASTING & REORDER ASSISTANT",
        "=" * 46,
        f"Forecast method: {method}",
        f"Products analyzed: {len(rows)}",
        f"Reorder now: {sum(r['status'] == 'REORDER NOW' for r in rows)}",
        f"Stockout-prone: {sum(r['stockout_prone'] for r in rows)}",
        "",
        "PRIORITIZED INVENTORY",
        "-" * 46,
    ]

    for index, item in enumerate(rows, start=1):
        lines.extend([
            f"{index}. {item['product_name']} (ID: {item['product_id']})",
            f"   Status: {item['status']}",
            f"   Current stock: {item['current_stock']}",
            f"   Forecast demand/day: {item['forecast_daily_demand']}",
            f"   Lead time: {item['lead_time_days']} days",
            f"   Safety stock: {item['safety_stock']}",
            f"   Reorder point: {item['reorder_point']}",
            f"   Suggested order quantity: {item['reorder_quantity']}",
            f"   Days until stockout: {item['days_until_stockout']}",
            f"   Stockout prone: {item['stockout_prone']}",
            f"   MAE: {item['mae']}; MAPE (%): {item['mape_percent']}",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def save_chart(path, rows):
    """Optional bar chart. The project still works without matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping chart.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["product_name"] for r in rows]
    stock = [r["current_stock"] for r in rows]
    reorder_points = [r["reorder_point"] for r in rows]

    positions = list(range(len(rows)))
    plt.figure(figsize=(max(8, len(rows) * 0.8), 5))
    plt.bar([p - 0.2 for p in positions], stock, width=0.4, label="Current stock")
    plt.bar([p + 0.2 for p in positions], reorder_points, width=0.4, label="Reorder point")
    plt.xticks(positions, names, rotation=45, ha="right")
    plt.ylabel("Units")
    plt.title("Current Stock vs Reorder Point")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Forecast inventory demand and create a reorder list."
    )
    parser.add_argument("--sales", default="sales.csv", help="Historical sales CSV")
    parser.add_argument("--inventory", default="inventory.csv", help="Inventory master CSV")
    parser.add_argument("--method", choices=["moving_average", "exponential"],
                        default="moving_average")
    parser.add_argument("--window", type=int, default=7, help="Moving average days")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Smoothing factor for exponential forecast (0 to 1)")
    parser.add_argument("--service-z", type=float, default=1.65,
                        help="Safety-stock service factor")
    parser.add_argument("--slow-days", type=int, default=30,
                        help="Reserved for configurable slow-moving analysis")
    parser.add_argument("--out", default="reports/reorder_report.txt")
    parser.add_argument("--csv", default="reports/reorder_report.csv")
    parser.add_argument("--json", default="reports/reorder_report.json")
    parser.add_argument("--chart", default="reports/stock_vs_reorder.png")
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.window < 1:
        parser.error("--window must be at least 1")
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be greater than 0 and at most 1")
    if args.service_z < 0:
        parser.error("--service-z cannot be negative")

    try:
        sales, inventory = load_and_clean_data(args.sales, args.inventory)
        rows = create_reorder_report(
            sales, inventory, args.method, args.window, args.alpha, args.service_z,
            args.slow_days
        )
        save_text_report(args.out, rows, args.method)
        save_csv(args.csv, rows)
        save_json(args.json, rows, args.method)

        if not args.no_chart:
            save_chart(args.chart, rows)

        print(f"Products analyzed: {len(rows)}")
        print(f"Reorder now: {sum(r['status'] == 'REORDER NOW' for r in rows)}")
        print(f"Stockout-prone: {sum(r['stockout_prone'] for r in rows)}")
        print(f"Text report: {args.out}")
        print(f"CSV report: {args.csv}")
        print(f"JSON report: {args.json}")
        if not args.no_chart:
            print(f"Chart: {args.chart}")

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
