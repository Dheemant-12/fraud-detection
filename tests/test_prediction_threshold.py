from src.config import PREDICTION_THRESHOLD


def test_prediction_threshold():
    assert PREDICTION_THRESHOLD == 0.5
    assert 0 <= PREDICTION_THRESHOLD <= 1