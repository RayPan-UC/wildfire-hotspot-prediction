from wildfire_hotspot_prediction.build_prediction_data.predictor import WildfirePredictor
from wildfire_hotspot_prediction.build_prediction_data.feature_builder import build_prediction_features
from wildfire_hotspot_prediction.build_prediction_data.era5_check import ensure_era5_coverage


def run_prediction_pipeline(study, t1, delta_t_h, predictor, threshold):
    """Build features, run prediction, and return results with fire context.

    Args:
        study:      Study instance.
        t1:         Current overpass timestamp (t1).
        delta_t_h:  Prediction horizon in hours (e.g. 3, 6, 12).
        predictor:  WildfirePredictor instance.
        threshold:  Decision threshold for binary prediction.

    Returns:
        Tuple of (result_df, intermediates):
            result_df:     DataFrame with prob/pred columns (empty if no features).
            intermediates: Dict with fire context (t0/t1, fire metrics, weather,
                           fwi, wind_forecast). Empty dict if no features.
    """
    import pandas as pd

    features_df, intermediates = build_prediction_features(study, t1=t1, delta_t_h=float(delta_t_h))
    if features_df.empty:
        return pd.DataFrame(), intermediates

    result_df = predictor.predict(features_df, threshold=threshold)
    return result_df, intermediates
