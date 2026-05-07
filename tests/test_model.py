import os
import pytest
import mlflow
from mlflow.tracking import MlflowClient

def test_best_model_accuracy():
    """Integration test to verify the best model in staging meets the accuracy threshold."""
    
    # Path to the run ID saved during training
    run_id_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_run_id.txt")
    
    assert os.path.exists(run_id_file), "best_run_id.txt not found. Train the model first."
    
    with open(run_id_file, "r") as f:
        run_id = f.read().strip()
        
    client = MlflowClient(tracking_uri="http://localhost:5000")
    
    # Get the run details
    run = client.get_run(run_id)
    metrics = run.data.metrics
    
    assert "accuracy" in metrics, "Accuracy metric not found in the best run"
    accuracy = metrics["accuracy"]
    
    print(f"\nBest model accuracy: {accuracy}")
    
    # Model validation: accuracy must be above 0.70
    threshold = 0.70
    assert accuracy > threshold, f"Model accuracy {accuracy} is below threshold {threshold}"
