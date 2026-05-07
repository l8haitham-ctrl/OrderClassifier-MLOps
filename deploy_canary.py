import os
import time
import random
from mlflow.tracking import MlflowClient

def test_canary_metrics(traffic_percentage, threshold=0.70):
    """
    Simulates checking the live metrics of the canary deployment.
    In a real scenario, this would query Prometheus, Datadog, or CloudWatch.
    """
    print(f"[{traffic_percentage}% Traffic] Monitoring live canary metrics...")
    time.sleep(1) # Simulate observation period
    
    # Simulate a metric score (e.g. accuracy or non-error rate)
    # We add a slight randomness but mostly keep it above threshold for success
    live_metric = random.uniform(0.72, 0.85)
    print(f"[{traffic_percentage}% Traffic] Live metric: {live_metric:.4f} (Threshold: {threshold})")
    
    return live_metric >= threshold

def main():
    print("=" * 60)
    print(" CANARY DEPLOYMENT SIMULATOR")
    print("=" * 60)
    
    run_id_file = os.path.join(os.path.dirname(__file__), "best_run_id.txt")
    if not os.path.exists(run_id_file):
        print(" Error: best_run_id.txt not found. Train the model first.")
        return
        
    with open(run_id_file, "r") as f:
        run_id = f.read().strip()
        
    client = MlflowClient(tracking_uri="http://localhost:5000")
    
    # 1. Identify the model in Staging
    try:
        versions = client.search_model_versions(f"run_id='{run_id}'")
        if not versions:
            print(" Error: Could not find model version associated with the run ID.")
            return
            
        model_version = versions[0]
        model_name = model_version.name
        version_num = model_version.version
        current_stage = model_version.current_stage
        
        if current_stage != "Staging":
            print(f" Error: Expected model {model_name} version {version_num} to be in 'Staging', but found '{current_stage}'")
            return
            
        print(f"\n Identified Model for Canary: {model_name} (Version {version_num}) in Staging.")
        
    except Exception as e:
        print(f" Error connecting to MLflow: {e}")
        return

    # 2. Canary Rollout Phases
    rollout_phases = [5, 25, 50, 100]
    
    print("\n Starting Canary Rollout Strategy: 5% -> 25% -> 50% -> 100%")
    
    for traffic in rollout_phases:
        print(f"\n>>> Routing {traffic}% of traffic to new model v{version_num}...")
        success = test_canary_metrics(traffic_percentage=traffic)
        
        if not success:
            print(f"\n [ROLLBACK INITIATED] Metrics fell below threshold at {traffic}% traffic.")
            print(" Rolling back 100% traffic to previous v1 (old) model.")
            
            # Transition to Archived instead of Production
            print(f" Archiving failed model version {version_num}...")
            client.transition_model_version_stage(
                name=model_name,
                version=version_num,
                stage="Archived"
            )
            print(" Rollback complete. Deployment aborted.")
            return
            
        print(" Metrics stable. Proceeding to next phase.")
        time.sleep(1)

    # 3. Finalize Deployment
    print("\n Canary Rollout completed successfully at 100% traffic!")
    print(f" Promoting {model_name} version {version_num} from Staging to Production...")
    
    client.transition_model_version_stage(
        name=model_name,
        version=version_num,
        stage="Production",
        archive_existing_versions=True
    )
    
    print("=" * 60)
    print(" DEPLOYMENT SUCCESSFUL ")
    print("=" * 60)

if __name__ == "__main__":
    # Seed for reproducibility in simulation
    random.seed()
    main()
