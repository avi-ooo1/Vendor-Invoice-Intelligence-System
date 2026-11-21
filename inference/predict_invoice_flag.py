from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

def load_model(model_path=MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path=SCALER_PATH):
    """
    Load trained feature scaler if available.
    """
    if Path(scaler_path).exists():
        with open(scaler_path, "rb") as f:
            return joblib.load(f)
    return None

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    scaler = load_scaler()
    input_df = pd.DataFrame(input_data)
    
    features = input_df[FEATURES] if all(col in input_df.columns for col in FEATURES) else input_df
    
    if scaler is not None:
        scaled_features = scaler.transform(features)
        input_df['Predicted_Flag'] = model.predict(scaled_features).round()
    else:
        input_df['Predicted_Flag'] = model.predict(features).round()
        
    return input_df

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.0]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)