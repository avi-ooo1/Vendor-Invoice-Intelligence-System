from pathlib import Path

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
    save_model
)

def main():
    # Dynamic path to data/inventory.db relative to this file
    base_dir = Path(__file__).resolve().parent.parent
    db_path = base_dir / "data" / "inventory.db"
    
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(str(db_path))

    # Prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest Regression")

    models = [
        ("Linear Regression", lr_model, lr_metrics),
        ("Decision Tree Regression", dt_model, dt_metrics),
        ("Random Forest Regression", rf_model, rf_metrics),
    ]

    # Select best model based on highest R² score
    best_model_name, best_model, _ = max(models, key=lambda item: item[2]["r2"])

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    save_model(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: models/{model_path.name}")

if __name__ == "__main__":
    main()
