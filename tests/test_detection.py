import pandas as pd

from src.detection.alerts import to_alerts


def test_to_alerts_keeps_only_attacks():
    preds = pd.DataFrame({"prediction": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    alerts = to_alerts(preds)
    assert len(alerts) == 2
    assert set(alerts["prediction"]) == {1}
