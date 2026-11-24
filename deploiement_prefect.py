"""Étape 05 - Déploiements Prefect planifiés."""

from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, install_flow, code_flow

if __name__ == "__main__":
    # Names match the handout: prefect deployment run 'all/ml-pipeline-all'
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # every day at 02:00
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["code", "mlops"]),
    )
