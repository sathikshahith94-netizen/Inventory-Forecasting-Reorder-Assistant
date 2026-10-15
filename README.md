 "-" * 46,
 for index, item in enumerate(rows, start=1):
 lines.extend([
 f"{index}. {item['product_name']} (ID: {item['product_id']})",
 f" Status: {item['status']}",
 f" Current stock: {item['current_stock']}",
 f" Forecast demand/day: {item['forecast_daily_demand']}",
 f" Lead time: {item['lead_time_days']} days",
 f" Safety stock: {item['safety_stock']}",
 f" Reorder point: {item['reorder_point']}",
 f" Suggested order quantity: {item['reorder_quantity']}",
 f" Days until stockout: {item['days_until_stockout']}",
 f" Stockout prone: {item['stockout_prone']}",
 f" MAE: {item['mae']}; MAPE (%): {item['mape_percent']}",
 "",
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
