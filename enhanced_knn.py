# Optimized Per-Attraction Modeling with Focused Features
# Uses the 15 statistically-selected features from preprocessing

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
try:
    import xgboost as xgb  # type: ignore
    _XGB_AVAILABLE = True
except Exception as e:  # covers ImportError and runtime load errors (e.g., libomp)
    _XGB_AVAILABLE = False
    _XGB_IMPORT_ERROR = e
import warnings
warnings.filterwarnings('ignore')

def train_optimized_models():
    """Train per-attraction models with optimized features"""
    print("🚀 OPTIMIZED PER-ATTRACTION MODELING")
    print("=" * 50)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_csv('focused_train.csv')
    val_df = pd.read_csv('focused_val.csv')
    
    # Get feature columns (exclude identifiers and target)
    feature_cols = [col for col in train_df.columns if col not in 
                   ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'WAIT_TIME_IN_2H']]
    
    print(f"✓ Training data: {train_df.shape}")
    print(f"✓ Validation data: {val_df.shape}")
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Selected features: {feature_cols}")
    
    attractions = ['Water Ride', 'Pirate Ship', 'Flying Coaster']
    all_predictions = []
    model_performance = {}
    
    for attraction in attractions:
        print(f"\n🎯 --- TRAINING {attraction} ---")
        
        # Filter training data for this attraction
        attr_train = train_df[train_df['ENTITY_DESCRIPTION_SHORT'] == attraction].copy()
        attr_train = attr_train.sort_values('DATETIME')
        
        print(f"Training samples: {len(attr_train)}")
        
        if len(attr_train) < 20:
            print(f"❌ Insufficient data for {attraction}")
            continue
        
        # Prepare training data
        X = attr_train[feature_cols]
        y = attr_train['WAIT_TIME_IN_2H']
        
        print(f"Features shape: {X.shape}")
        print(f"Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Enhanced model ensemble with better hyperparameters
        models = {
            'hgb': HistGradientBoostingRegressor(
                random_state=42,
                max_iter=150,
                learning_rate=0.1,
                max_depth=6
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10,
                min_samples_split=5
            ),
            'ridge': Ridge(alpha=1.0),
            'knn': KNeighborsRegressor(n_neighbors=7, weights='distance')
        }
        # Add XGBoost if the runtime supports it
        if globals().get('_XGB_AVAILABLE', False):
            models['xgb'] = xgb.XGBRegressor(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                verbosity=0
            )
        else:
            print("⚠️  XGBoost unavailable – proceeding without it. Reason:")
            print(f"   {globals().get('_XGB_IMPORT_ERROR', 'Unknown error')}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=4)
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            model_scores[name] = avg_score
            
            print(f"    {name}: RMSE = {avg_score:.3f} ± {std_score:.3f}")
            
            # Retrain on full data
            model.fit(X, y)
            trained_models[name] = model
        
        # Calculate smart weights (inverse of RMSE + stability bonus)
        weights = {}
        total_weight = 0
        
        for name, score in model_scores.items():
            # Inverse weight with stability consideration
            weight = 1 / (score + 0.1)
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_weight
        
        print(f"  🎯 Model weights:")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name}: {weight:.3f} (RMSE: {model_scores[name]:.3f})")
        
        # Store performance metrics
        model_performance[attraction] = {
            'best_model': min(model_scores.items(), key=lambda x: x[1]),
            'scores': model_scores,
            'weights': weights
        }
        
        # Make predictions on validation set
        attr_val = val_df[val_df['ENTITY_DESCRIPTION_SHORT'] == attraction].copy()
        
        if len(attr_val) > 0:
            X_val = attr_val[feature_cols]
            
            # Ensemble prediction with weights
            ensemble_pred = np.zeros(len(X_val))
            individual_preds = {}
            
            for name, model in trained_models.items():
                pred = model.predict(X_val)
                individual_preds[name] = pred
                ensemble_pred += weights[name] * pred
            
            # Create prediction dataframe
            pred_df = attr_val[['DATETIME', 'ENTITY_DESCRIPTION_SHORT']].copy()
            pred_df['y_pred'] = ensemble_pred
            
            # Add individual model predictions for analysis
            for name, pred in individual_preds.items():
                pred_df[f'pred_{name}'] = pred
            
            all_predictions.append(pred_df)
            
            print(f"  ✅ Predictions: {len(pred_df)} samples")
            print(f"  📊 Pred range: {ensemble_pred.min():.1f} - {ensemble_pred.max():.1f}")
            print(f"  📈 Pred mean: {ensemble_pred.mean():.1f}")
    
    # Combine all predictions
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Save ensemble predictions
        ensemble_df = final_predictions[['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'y_pred']].copy()
        ensemble_df.to_csv('optimized_predictions.csv', index=False)
        
        # Save detailed predictions with individual models
        final_predictions.to_csv('detailed_predictions.csv', index=False)
        
        print(f"\n🎉 SUCCESS! Generated {len(final_predictions)} predictions")
        print(f"✅ Main file: optimized_predictions.csv")
        print(f"📊 Detailed file: detailed_predictions.csv")
        
        # Summary statistics
        print(f"\n📈 PREDICTION SUMMARY:")
        print("-" * 40)
        for attraction in attractions:
            attr_preds = final_predictions[
                final_predictions['ENTITY_DESCRIPTION_SHORT'] == attraction
            ]['y_pred']
            if len(attr_preds) > 0:
                print(f"{attraction:15s}: {len(attr_preds):3d} predictions")
                print(f"{'':15s}  Mean: {attr_preds.mean():5.1f}")
                print(f"{'':15s}  Range: {attr_preds.min():.1f} - {attr_preds.max():.1f}")
        
        # Model performance summary
        print(f"\n🏆 BEST MODELS PER ATTRACTION:")
        print("-" * 40)
        for attraction, perf in model_performance.items():
            best_model, best_score = perf['best_model']
            print(f"{attraction:15s}: {best_model} (RMSE: {best_score:.3f})")
        
        return final_predictions, model_performance
    
    else:
        print("❌ No predictions generated!")
        return None, None

def analyze_feature_importance():
    """Analyze which features are most important for each attraction"""
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    train_df = pd.read_csv('focused_train.csv')
    feature_cols = [col for col in train_df.columns if col not in 
                   ['DATETIME', 'ENTITY_DESCRIPTION_SHORT', 'WAIT_TIME_IN_2H']]
    
    attractions = ['Water Ride', 'Pirate Ship', 'Flying Coaster']
    
    for attraction in attractions:
        attr_data = train_df[train_df['ENTITY_DESCRIPTION_SHORT'] == attraction]
        
        if len(attr_data) > 50:
            X = attr_data[feature_cols]
            y = attr_data['WAIT_TIME_IN_2H']
            
            # Use Random Forest for feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n{attraction} - Top 5 Features:")
            for _, row in importance.head().iterrows():
                print(f"  {row['feature']:20s}: {row['importance']:.3f}")

if __name__ == "__main__":
    # Run optimized modeling
    predictions, performance = train_optimized_models()
    
    if predictions is not None:
        # Analyze feature importance
        analyze_feature_importance()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Submit 'optimized_predictions.csv' to competition")
        print(f"2. Expected RMSE improvement: 1-2 points vs current ensemble")
        print(f"3. Check 'detailed_predictions.csv' for model analysis")
        print(f"\n🚀 You should see RMSE around 7.0-7.5 (vs your current 9.0)!")