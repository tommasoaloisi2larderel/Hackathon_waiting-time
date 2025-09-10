
import pandas as pd
import numpy as np
from per_attraction_modeling import train_per_attraction
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# --- Helper to robustly get a Per-Attraction model instance ---
def get_per_attraction_model():
    """Return an object with .train_attraction_models(df) and .predict(df).
    Tries several known patterns so this file works with different versions
    of per_attraction_modeling.
    """
    # 1) Prefer a direct class export `PerAttractionModel`
    try:
        from per_attraction_modeling import PerAttractionModel  # type: ignore
        inst = PerAttractionModel()
        if hasattr(inst, 'train_attraction_models') and hasattr(inst, 'predict'):
            return inst
    except Exception:
        pass

    # 2) Try factory: train_per_attraction(return_model=True)
    try:
        inst = train_per_attraction(return_model=True)  # type: ignore
        if hasattr(inst, 'train_attraction_models') and hasattr(inst, 'predict'):
            return inst
    except TypeError:
        # signature may not support return_model
        pass
    except Exception:
        pass

    # 3) Try calling train_per_attraction() and verify it returns a model
    try:
        inst = train_per_attraction()
        if hasattr(inst, 'train_attraction_models') and hasattr(inst, 'predict'):
            return inst
    except Exception:
        pass

    raise RuntimeError(
        "Could not obtain a per-attraction model. Ensure `per_attraction_modeling` "
        "exposes `PerAttractionModel` or supports `train_per_attraction(return_model=True)` "
        "to return a model instance."
    )

def compare_models():
    """Compare global ensemble vs per-attraction models"""
    print("Model Comparison: Global vs Per-Attraction")
    print("=" * 50)
    
    # Load training data
    train_df = pd.read_csv('waiting_times_train.csv')
    train_df['DATETIME'] = pd.to_datetime(train_df['DATETIME'])
    train_df = train_df.sort_values(['ENTITY_DESCRIPTION_SHORT', 'DATETIME'])
    
    # Time series split for each attraction
    attractions = ['Water Ride', 'Pirate Ship', 'Flying Coaster']
    results = {}
    
    for attraction in attractions:
        print(f"\nEvaluating {attraction}...")
        
        # Filter data for this attraction
        attraction_data = train_df[train_df['ENTITY_DESCRIPTION_SHORT'] == attraction].copy()
        
        if len(attraction_data) < 20:
            print(f"  Insufficient data ({len(attraction_data)} samples)")
            continue
            
        # Use last 20% as test set for comparison
        split_idx = int(len(attraction_data) * 0.8)
        train_subset = attraction_data.iloc[:split_idx]
        test_subset = attraction_data.iloc[split_idx:]
        
        print(f"  Train: {len(train_subset)}, Test: {len(test_subset)}")
        
        # Train per-attraction model on train subset
        pa_model = get_per_attraction_model()
        pa_model.train_attraction_models(train_subset)
        
        # Make predictions on test subset
        pa_predictions = pa_model.predict(test_subset)
        
        if not pa_predictions.empty:
            # Calculate RMSE
            y_true = test_subset['WAIT_TIME_IN_2H'].values
            y_pred = pa_predictions['y_pred'].values
            
            # Align predictions with true values
            test_aligned = test_subset.reset_index(drop=True)
            pred_aligned = pa_predictions.reset_index(drop=True)
            
            if len(y_true) == len(y_pred):
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                results[attraction] = {
                    'per_attraction_rmse': rmse,
                    'n_samples': len(y_true),
                    'y_true_mean': np.mean(y_true),
                    'y_pred_mean': np.mean(y_pred)
                }
                print(f"  Per-attraction RMSE: {rmse:.3f}")
                print(f"  True mean: {np.mean(y_true):.1f}, Pred mean: {np.mean(y_pred):.1f}")
            else:
                print(f"  Length mismatch: true={len(y_true)}, pred={len(y_pred)}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    total_samples = 0
    weighted_rmse = 0
    
    for attraction, metrics in results.items():
        rmse = metrics['per_attraction_rmse']
        n_samples = metrics['n_samples']
        
        print(f"{attraction:15s}: RMSE = {rmse:6.3f} (n={n_samples})")
        
        total_samples += n_samples
        weighted_rmse += rmse * n_samples
    
    if total_samples > 0:
        weighted_avg_rmse = weighted_rmse / total_samples
        print(f"{'Weighted Average':15s}: RMSE = {weighted_avg_rmse:6.3f}")
        
        print(f"\nComparison with your current RMSE of 9.0:")
        improvement = 9.0 - weighted_avg_rmse
        print(f"Expected improvement: {improvement:+.3f} RMSE points")
        if improvement > 0:
            print("✓ Per-attraction modeling shows improvement!")
        else:
            print("⚠ Current global model performs better on this test")

def quick_train_and_predict():
    """Quick pipeline to generate new predictions"""
    print("\nQuick Training & Prediction Pipeline")
    print("=" * 40)
    
    # Load data
    train_df = pd.read_csv('waiting_times_train.csv')
    val_df = pd.read_csv('waiting_times_X_test_val.csv')
    
    # Train per-attraction models
    pa_model = get_per_attraction_model()
    pa_model.train_attraction_models(train_df)
    
    # Generate predictions
    predictions = pa_model.predict(val_df)
    
    if not predictions.empty:
        # Save for submission
        predictions.to_csv('per_attraction_submission_val.csv', index=False)
        print(f"✓ Predictions saved to per_attraction_submission_val.csv")
        print(f"  Shape: {predictions.shape}")
        
        # Show prediction stats by attraction
        print("\nPrediction Statistics by Attraction:")
        for attraction in predictions['ENTITY_DESCRIPTION_SHORT'].unique():
            attr_preds = predictions[predictions['ENTITY_DESCRIPTION_SHORT'] == attraction]['y_pred']
            print(f"  {attraction:15s}: {len(attr_preds):3d} predictions, "
                  f"mean={attr_preds.mean():5.1f}, std={attr_preds.std():5.1f}")
        
        return predictions
    else:
        print("❌ No predictions generated")
        return None

if __name__ == "__main__":
    # Run comparison
    compare_models()
    
    # Generate new predictions
    predictions = quick_train_and_predict()