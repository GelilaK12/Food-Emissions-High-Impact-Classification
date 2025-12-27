import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

with mlflow.start_run(run_name="sanity_check"):
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.88)

print("Logged sanity_check run")
