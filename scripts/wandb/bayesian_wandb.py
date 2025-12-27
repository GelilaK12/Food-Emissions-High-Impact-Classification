if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import pymc as pm
    import arviz as az
    import matplotlib.pyplot as plt
    import joblib
    import wandb
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path

    # =====================
    # Initialize W&B
    # =====================
    wandb.init(
        project="food_emissions_classification",
        entity="gelilakassaye6-vsco",
        config={
            "model_type": "bayesian_logistic",
            "draws": 300,
            "tune": 300,
            "target_accept": 0.9 } )

    # =====================
    # Load & Clean Data
    # =====================
    df = pd.read_csv("data/Food_Production.csv")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["high_impact"] = (
        df["total_emissions"] >= df["total_emissions"].quantile(0.75)).astype(int)

    features = [
        "land_use_change",
        "animal_feed",
        "farm",
        "processing",
        "transport",
        "packaging",
        "retail"
    ]

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df["high_impact"]

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].values.astype("float64")
    y = y.loc[mask].astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # =====================
    # Bayesian Logistic Model
    # =====================
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        coeffs = pm.Normal("coeffs", mu=0, sigma=2, shape=X.shape[1])
        logits = pm.math.clip(intercept + pm.math.dot(X, coeffs), -15, 15)

        y_obs = pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        trace = pm.sample(
            draws=300,
            tune=300,
            chains=1,
            cores=1,
            target_accept=0.9,
            return_inferencedata=True
        )

    # =====================
    # Posterior Predictive
    # =====================
    with model:
        ppc = pm.sample_posterior_predictive(trace)

    y_pred = ppc.posterior_predictive["y_obs"].mean(("chain", "draw")).values
    y_pred_class = (y_pred > 0.5).astype(int)

    accuracy = (y_pred_class == y).mean()
    wandb.log({"accuracy": accuracy})
    print(f"\nAccuracy: {accuracy:.4f}")

    # =====================
    # Posterior Summary
    # =====================
    summary = az.summary(trace, var_names=["intercept", "coeffs"])
    coeff_table = wandb.Table(columns=["feature", "mean", "sd", "hdi_3%", "hdi_97%"])

    for i, f in enumerate(features):
        coeff_table.add_data(
            f,
            summary.loc[f"coeffs[{i}]", "mean"],
            summary.loc[f"coeffs[{i}]", "sd"],
            summary.loc[f"coeffs[{i}]", "hdi_3%"],
            summary.loc[f"coeffs[{i}]", "hdi_97%"]
        )

    wandb.log({"posterior_coefficients": coeff_table})

    # =====================
    # Plots
    # =====================
    az.plot_trace(trace, var_names=["intercept", "coeffs"])
    plt.tight_layout()
    plt.savefig("trace_plot.png")
    wandb.log({"trace_plot": wandb.Image("trace_plot.png")})

    plt.figure(figsize=(6, 4))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.xlabel("True High Impact")
    plt.ylabel("Predicted Probability")
    plt.savefig("posterior_pred.png")
    wandb.log({"posterior_predictive_plot": wandb.Image("posterior_pred.png")})

    # =====================
    # Save Artifact
    # =====================
    Path("artifacts").mkdir(exist_ok=True)
    model_path = "artifacts/bayesian_logreg_trace.pkl"
    joblib.dump(trace, model_path)

    artifact = wandb.Artifact("bayesian_logreg_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()
