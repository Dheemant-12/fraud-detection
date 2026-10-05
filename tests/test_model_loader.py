from src.model_loader import load_model


def test_model_loads():
    saved_model = load_model()

    assert "model" in saved_model
    assert "features" in saved_model

    assert saved_model["model"] is not None
    assert len(saved_model["features"]) > 0