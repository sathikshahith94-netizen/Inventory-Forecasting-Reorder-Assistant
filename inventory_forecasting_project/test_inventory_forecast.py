import unittest
import pandas as pd

from inventory_forecast import (
    calculate_forecast_error,
    calculate_reorder_metrics,
    forecast_demand,
)


class InventoryForecastTests(unittest.TestCase):
    def test_moving_average_forecast(self):
        forecast, std_dev = forecast_demand(
            pd.Series([2, 4, 6, 8]), method="moving_average", window=2
        )
        self.assertEqual(forecast, 7.0)
        self.assertGreater(std_dev, 0)

    def test_exponential_forecast(self):
        forecast, _ = forecast_demand(
            pd.Series([10, 20, 30]), method="exponential", alpha=0.5
        )
        self.assertEqual(forecast, 22.5)

    def test_reorder_point(self):
        result = calculate_reorder_metrics(
            current_stock=5, average_daily_demand=3,
            demand_std=0, lead_time_days=4, service_z=1.65
        )
        self.assertEqual(result["lead_time_demand"], 12)
        self.assertEqual(result["safety_stock"], 0)
        self.assertEqual(result["reorder_point"], 12)
        self.assertGreater(result["reorder_quantity"], 0)

    def test_forecast_error(self):
        result = calculate_forecast_error([10, 20], [8, 18])
        self.assertEqual(result["mae"], 2.0)
        self.assertEqual(result["mape_percent"], 15.0)


if __name__ == "__main__":
    unittest.main()
