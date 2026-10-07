from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import joblib

CAR_PRICE_API_DIR = Path(__file__).resolve().parent
DATA_PATH = CAR_PRICE_API_DIR / "cardekho_data (1).csv"
MODEL_PATH = CAR_PRICE_API_DIR / "random_forest_model.pkl"
COLS_PATH = CAR_PRICE_API_DIR / "feature_columns.pkl"

_model = None
_feature_columns = None
_cached_df = None


def load_artifacts():
    global _model, _feature_columns
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _feature_columns is None:
        _feature_columns = joblib.load(COLS_PATH)


def get_dataset() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        if DATA_PATH.exists():
            _cached_df = pd.read_csv(DATA_PATH)
        else:
            _cached_df = pd.DataFrame()
    return _cached_df


def get_known_cars() -> List[str]:
    df = get_dataset()
    if not df.empty and "Car_Name" in df.columns:
        return sorted(df["Car_Name"].unique().tolist())
    return ["swift", "city", "ciaz", "verna", "i20", "fortuner", "innova", "brio"]


def get_car_defaults(car_name: str) -> Dict[str, float]:
    df = get_dataset()
    if not df.empty and "Car_Name" in df.columns:
        match = df[df["Car_Name"].str.lower() == car_name.strip().lower()]
        if not match.empty:
            return {
                "present_price": round(float(match["Present_Price"].median()), 2),
                "avg_selling_price": round(float(match["Selling_Price"].median()), 2),
                "avg_kms": int(match["Kms_Driven"].median()),
                "year": int(match["Year"].mode()[0]),
                "fuel_type": str(match["Fuel_Type"].mode()[0]),
                "transmission": str(match["Transmission"].mode()[0]),
                "seller_type": str(match["Seller_Type"].mode()[0]),
            }
    return {
        "present_price": 6.5,
        "avg_selling_price": 4.5,
        "avg_kms": 35000,
        "year": 2016,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "seller_type": "Dealer",
    }


def preprocess(payload: dict) -> pd.DataFrame:
    """
    Converts raw input into the SAME one-hot encoded column structure used in training.
    """
    load_artifacts()
    df = pd.DataFrame([payload])

    categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission", "Owner", "Car_Name"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns to training columns without fragmentation
    df_encoded = df_encoded.reindex(columns=_feature_columns, fill_value=0)

    return df_encoded


def predict_price(payload: dict) -> Tuple[float, dict]:
    load_artifacts()
    X = preprocess(payload)
    pred_raw = float(_model.predict(X)[0])
    # Selling price should not be negative or unrealistically 0
    pred = max(0.1, round(pred_raw, 2))

    present_price = float(payload.get("Present_Price", 0.0))
    depreciation = 0.0
    if present_price > 0:
        depreciation = round(max(0.0, (1 - (pred / present_price)) * 100), 1)

    # Estimate fair market range (+/- 6%)
    fair_low = round(max(0.05, pred * 0.94), 2)
    fair_high = round(pred * 1.06, 2)

    meta = {
        "prediction_price": pred,
        "fair_price_low": fair_low,
        "fair_price_high": fair_high,
        "depreciation_pct": depreciation,
        "currency": "Lakhs (INR)",
        "inr_formatted": f"Rs. {pred * 100000:,.0f}",
    }
    return pred, meta


def get_feature_importances() -> List[Dict[str, float]]:
    load_artifacts()
    if hasattr(_model, "feature_importances_"):
        fi = list(zip(_feature_columns, _model.feature_importances_))
        fi.sort(key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(imp), 4)} for f, imp in fi[:10]]
    return []
