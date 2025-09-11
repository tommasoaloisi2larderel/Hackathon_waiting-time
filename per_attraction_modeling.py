# Simple Per-Attraction Modeling Script
# Run this directly: python train_per_attraction.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """Create features from the dataset"""
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    
    # Time features
    df['hour'] = df['DATETIME'].dt.hour
    df['day_of_week'] = df['DATETIME'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['DATETIME'].dt.month
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Fill missing event times
    event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
    for col in event_cols:
        if col in df.columns:
            df[col] = df[col].fillna(999)
            
    # Capacity utilization
    df['wait_per_capacity'] = df['CURRENT_WAIT_TIME'] / (df['ADJUST_CAPACITY'] + 1)
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in 
                   ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'WAIT_TIME_IN_2H']]
    
    return df, feature_cols

def train_per_attraction():
    """Train separate models for each attraction"""
    print("Loading data...")
    train_df = pd.read_csv('waiting_times_train.csv')
    val_df = pd.read_csv('waiting_times_X_test_val.csv')
    
    # Prepare features
    train_df, feature_cols = prepare_features(train_df)
    val_df, _ = prepare_features(val_df)
    
    attractions = ['Water Ride', 'Pirate Ship', 'Flying Coaster']
    all_predictions = []
    
    print(f"Using {len(feature_cols)} features")
    
    for attraction in attractions:
        print(f"\n--- Training {attraction} ---")
        
        # Filter training data
        attr_train = train_df[train_df['ENTITY_DESCRIPTION_SHORT'] == attraction].copy()
        attr_train = attr_train.sort_values('DATETIME')
        
        print(f"Training samples: {len(attr_train)}")
        
        if len(attr_train) < 10:
            print(f"Insufficient data for {attraction}")
            continue
            
        # Prepare training data
        X = attr_train[feature_cols]
        y = attr_train['WAIT_TIME_IN_2H']
        
        # Models to try
        models = {
            'hgb': HistGradientBoostingRegressor(random_state=42, max_iter=100),
            'rf': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'lr': LinearRegression()
        }
        
        # Cross-validation to get model weights
        tscv = TimeSeriesSplit(n_splits=3)
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            avg_score = np.mean(scores)
            model_scores[name] = avg_score
            print(f"  {name}: RMSE = {avg_score:.3f}")
            
            # Retrain on full data
            model.fit(X, y)
            trained_models[name] = model
        
        # Calculate weights (inverse of RMSE)
        weights = {}
        total_weight = 0
        for name, score in model_scores.items():
            weight = 1 / (score + 0.1)
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_weight
        
        print(f"  Weights: {', '.join([f'{k}:{v:.3f}' for k, v in weights.items()])}")
        
        # Make predictions on validation set
        attr_val = val_df[val_df['ENTITY_DESCRIPTION_SHORT'] == attraction].copy()
        
        if len(attr_val) > 0:
            X_val = attr_val[feature_cols]
            
            # Weighted ensemble prediction
            ensemble_pred = np.zeros(len(X_val))
            for name, model in trained_models.items():
                pred = model.predict(X_val)
                ensemble_pred += weights[name] * pred
            
            # Store predictions
            pred_df = attr_val[['DATETIME', 'ENTITY_DESCRIPTION_SHORT']].copy()
            pred_df['y_pred'] = ensemble_pred
            all_predictions.append(pred_df)
            
            print(f"  Predictions: {len(pred_df)} samples")
            print(f"  Pred range: {ensemble_pred.min():.1f} - {ensemble_pred.max():.1f}")
    
    # Combine all predictions
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Save predictions
        output_file = 'per_attraction_predictions.csv'
        final_predictions.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved {len(final_predictions)} predictions to {output_file}")
        
        # Show summary
        print("\nPrediction Summary:")
        for attraction in attractions:
            attr_preds = final_predictions[
                final_predictions['ENTITY_DESCRIPTION_SHORT'] == attraction
            ]['y_pred']
            if len(attr_preds) > 0:
                print(f"  {attraction:15s}: {len(attr_preds):3d} predictions, "
                      f"mean={attr_preds.mean():5.1f}, std={attr_preds.std():5.1f}")
        
        return final_predictions
    else:
        print("\n❌ No predictions generated!")
        return None

def compare_with_current():
    """Quick comparison with your current approach"""
    print("\n" + "="*50)
    print("COMPARISON WITH CURRENT ENSEMBLE")
    print("="*50)
    
    # Try to load your current predictions
    try:
        current_pred = pd.read_csv('TEST_waiting_times_ENSEMBLE.csv')
        new_pred = pd.read_csv('per_attraction_predictions.csv')
        
        print(f"Current predictions: {len(current_pred)}")
        print(f"New predictions: {len(new_pred)}")
        
        # Compare prediction distributions
        print("\nPrediction distributions:")
        print(f"Current - Mean: {current_pred['y_pred'].mean():.1f}, "
              f"Std: {current_pred['y_pred'].std():.1f}")
        print(f"New     - Mean: {new_pred['y_pred'].mean():.1f}, "
              f"Std: {new_pred['y_pred'].std():.1f}")
        
    except FileNotFoundError:
        print("Could not find current predictions for comparison")

if __name__ == "__main__":
    print("Per-Attraction Modeling Pipeline")
    print("="*40)
    
    # Train and predict
    predictions = train_per_attraction()
    
    # Compare if possible
    if predictions is not None:
        compare_with_current()
        
        print(f"\n🎯 Next step: Submit 'per_attraction_predictions.csv' to see RMSE improvement!")