# Weather Features Analysis Laboratory
# Test different approaches for weather data optimization

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class WeatherAnalysisLab:
    def __init__(self):
        self.results = {}
        
    def explore_weather_patterns(self):
        """Analyze weather data patterns and correlations"""
        print("🌤️ WEATHER FEATURES ANALYSIS LABORATORY")
        print("=" * 60)
        
        # Load data
        df = pd.read_csv('data/waiting_times_train.csv')
        weather_df = pd.read_csv('data/weather_data.csv')
        
        # Merge weather
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        weather_df['DATETIME'] = pd.to_datetime(weather_df['DATETIME'])
        df = pd.merge_asof(df.sort_values('DATETIME'), 
                          weather_df.sort_values('DATETIME'), 
                          on='DATETIME', direction='nearest')
        
        print(f"📊 Dataset with weather: {df.shape}")
        
        # Weather feature statistics
        weather_cols = ['temp', 'dew_point', 'feels_like', 'pressure', 'humidity', 
                       'wind_speed', 'rain_1h', 'snow_1h', 'clouds_all']
        
        print(f"\n🌡️ WEATHER STATISTICS:")
        print("-" * 40)
        weather_stats = df[weather_cols].describe()
        print(weather_stats.round(2))
        
        # Correlation with target
        print(f"\n🔗 WEATHER vs WAIT_TIME_IN_2H CORRELATION:")
        print("-" * 50)
        correlations = {}
        for col in weather_cols:
            if col in df.columns:
                corr = df[col].corr(df['WAIT_TIME_IN_2H'])
                correlations[col] = corr
                print(f"{col:12s}: {corr:6.3f}")
        
        # Weather feature inter-correlations
        print(f"\n🔄 WEATHER FEATURE CORRELATIONS:")
        print("-" * 40)
        weather_corr = df[weather_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i, col1 in enumerate(weather_cols):
            for j, col2 in enumerate(weather_cols[i+1:], i+1):
                if col1 in df.columns and col2 in df.columns:
                    corr_val = weather_corr.loc[col1, col2]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((col1, col2, corr_val))
        
        print("Highly correlated pairs (>0.7):")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"  {col1:12s} <-> {col2:12s}: {corr_val:6.3f}")
        
        # Weather impact by conditions
        print(f"\n🌦️ WEATHER CONDITION IMPACT:")
        print("-" * 40)
        
        # Rain impact
        no_rain = df[df['rain_1h'].fillna(0) == 0]['WAIT_TIME_IN_2H'].mean()
        with_rain = df[df['rain_1h'].fillna(0) > 0]['WAIT_TIME_IN_2H'].mean()
        print(f"No rain: {no_rain:.1f}min, With rain: {with_rain:.1f}min (diff: {with_rain-no_rain:+.1f})")
        
        # Temperature extremes
        temp_low = df[df['temp'] < 10]['WAIT_TIME_IN_2H'].mean()
        temp_mid = df[df['temp'].between(15, 25)]['WAIT_TIME_IN_2H'].mean()
        temp_high = df[df['temp'] > 30]['WAIT_TIME_IN_2H'].mean()
        print(f"Cold (<10°C): {temp_low:.1f}min")
        print(f"Comfortable (15-25°C): {temp_mid:.1f}min")
        print(f"Hot (>30°C): {temp_high:.1f}min")
        
        return df
    
    def create_base_features(self, df):
        """Base features without weather"""
        return pd.DataFrame({
            'current_wait': df['CURRENT_WAIT_TIME'],
            'capacity_normalized': df.groupby('ENTITY_DESCRIPTION_SHORT')['ADJUST_CAPACITY'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
            ),
            'has_downtime': (df['DOWNTIME'] > 0).astype(int),
            'time_to_next_event': df[['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']].fillna(240).clip(upper=240).min(axis=1),
            'hour': df['DATETIME'].dt.hour,
            'is_weekend': (df['DATETIME'].dt.dayofweek >= 5).astype(int),
            'is_pirate_ship': (df['ENTITY_DESCRIPTION_SHORT'] == 'Pirate Ship').astype(int),
            'is_flying_coaster': (df['ENTITY_DESCRIPTION_SHORT'] == 'Flying Coaster').astype(int),
        })
    
    def weather_encoding_1_all_raw(self, df):
        """Method 1: All weather features raw"""
        features = self.create_base_features(df)
        
        weather_cols = ['temp', 'dew_point', 'feels_like', 'pressure', 'humidity', 
                       'wind_speed', 'rain_1h', 'snow_1h', 'clouds_all']
        
        for col in weather_cols:
            if col in df.columns:
                features[col] = df[col].fillna(df[col].mean())
        
        return features, "All Raw Weather"
    
    def weather_encoding_2_comfort_score(self, df):
        """Method 2: Weather comfort/agreeability score"""
        features = self.create_base_features(df)
        
        # Weather comfort score (0-1, higher = more comfortable)
        temp_comfort = 1 - np.abs(df['temp'] - 20) / 25  # Optimal around 20°C
        temp_comfort = np.clip(temp_comfort, 0, 1)
        
        humidity_comfort = 1 - np.abs(df['humidity'] - 50) / 50  # Optimal around 50%
        humidity_comfort = np.clip(humidity_comfort, 0, 1)
        
        wind_comfort = np.where(df['wind_speed'] > 15, 0.5, 1)  # High wind uncomfortable
        
        rain_penalty = np.where(df['rain_1h'].fillna(0) > 0, 0.3, 1)  # Rain reduces comfort
        snow_penalty = np.where(df['snow_1h'].fillna(0) > 0, 0.2, 1)  # Snow reduces comfort
        
        # Combined comfort score
        features['weather_comfort'] = (temp_comfort * humidity_comfort * wind_comfort * 
                                     rain_penalty * snow_penalty)
        
        return features, "Weather Comfort Score"
    
    def weather_encoding_3_reduced_set(self, df):
        """Method 3: Reduced set removing redundant features"""
        features = self.create_base_features(df)
        
        # Keep only most predictive and non-redundant features
        # Based on correlations: temp/feels_like/dew_point are highly correlated
        features['temp'] = df['temp'].fillna(df['temp'].mean())
        features['humidity'] = df['humidity'].fillna(df['humidity'].mean())
        features['rain_1h'] = df['rain_1h'].fillna(0)
        features['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
        
        return features, "Reduced Weather Set"
    
    def weather_encoding_4_categorical(self, df):
        """Method 4: Categorical weather conditions"""
        features = self.create_base_features(df)
        
        # Temperature categories
        features['temp_cold'] = (df['temp'] < 10).astype(int)
        features['temp_cool'] = df['temp'].between(10, 18).astype(int)
        features['temp_comfortable'] = df['temp'].between(18, 25).astype(int)
        features['temp_warm'] = df['temp'].between(25, 30).astype(int)
        features['temp_hot'] = (df['temp'] > 30).astype(int)
        
        # Precipitation
        features['has_rain'] = (df['rain_1h'].fillna(0) > 0).astype(int)
        features['has_snow'] = (df['snow_1h'].fillna(0) > 0).astype(int)
        features['heavy_rain'] = (df['rain_1h'].fillna(0) > 2).astype(int)
        
        # Wind conditions
        features['windy'] = (df['wind_speed'] > 15).astype(int)
        
        # Humidity levels
        features['low_humidity'] = (df['humidity'] < 40).astype(int)
        features['high_humidity'] = (df['humidity'] > 70).astype(int)
        
        return features, "Categorical Weather"
    
    def weather_encoding_5_interactions(self, df):
        """Method 5: Weather interactions with attraction types"""
        features = self.create_base_features(df)
        
        # Key weather features
        features['temp'] = df['temp'].fillna(df['temp'].mean())
        features['has_rain'] = (df['rain_1h'].fillna(0) > 0).astype(int)
        features['humidity'] = df['humidity'].fillna(df['humidity'].mean())
        
        # Attraction-specific weather effects
        is_water_ride = (df['ENTITY_DESCRIPTION_SHORT'] == 'Water Ride').astype(int)
        
        # Water rides affected more by cold/rain
        features['water_x_cold'] = is_water_ride * (df['temp'] < 15).astype(int)
        features['water_x_rain'] = is_water_ride * features['has_rain']
        
        # Weekend weather interactions
        is_weekend = (df['DATETIME'].dt.dayofweek >= 5).astype(int)
        features['weekend_x_good_weather'] = is_weekend * (
            (df['temp'].between(18, 28)) & 
            (features['has_rain'] == 0)
        ).astype(int)
        
        return features, "Weather Interactions"
    
    def weather_encoding_6_pca(self, df):
        """Method 6: PCA dimensionality reduction"""
        features = self.create_base_features(df)
        
        weather_cols = ['temp', 'dew_point', 'feels_like', 'pressure', 'humidity', 
                       'wind_speed', 'rain_1h', 'snow_1h', 'clouds_all']
        
        # Prepare weather data
        weather_data = df[weather_cols].fillna(df[weather_cols].mean())
        
        # Apply PCA to reduce to 3 components
        pca = PCA(n_components=3)
        weather_pca = pca.fit_transform(weather_data)
        
        features['weather_pc1'] = weather_pca[:, 0]
        features['weather_pc2'] = weather_pca[:, 1]
        features['weather_pc3'] = weather_pca[:, 2]
        
        return features, "Weather PCA"
    
    def weather_encoding_7_seasonal_adjustment(self, df):
        """Method 7: Season-adjusted weather features"""
        features = self.create_base_features(df)
        
        # Raw weather
        features['temp'] = df['temp'].fillna(df['temp'].mean())
        features['humidity'] = df['humidity'].fillna(df['humidity'].mean())
        features['has_rain'] = (df['rain_1h'].fillna(0) > 0).astype(int)
        
        # Season-adjusted features
        df['month'] = df['DATETIME'].dt.month
        
        # Temperature relative to seasonal average
        seasonal_temp = df.groupby('month')['temp'].transform('mean')
        features['temp_vs_seasonal'] = df['temp'] - seasonal_temp
        
        # Month-specific weather expectations
        features['unexpected_cold'] = (
            (df['month'].isin([6, 7, 8])) & (df['temp'] < 15)
        ).astype(int)  # Cold in summer
        
        features['unexpected_hot'] = (
            (df['month'].isin([12, 1, 2])) & (df['temp'] > 25)
        ).astype(int)  # Hot in winter
        
        return features, "Season-Adjusted Weather"
    
    def weather_encoding_8_no_weather(self, df):
        """Method 8: No weather features"""
        features = self.create_base_features(df)
        return features, "No Weather"
    
    def test_weather_method(self, df, encoding_func, model_type='xgb'):
        """Test a weather encoding method"""
        try:
            X, method_name = encoding_func(df)
            y = df['WAIT_TIME_IN_2H']
            
            # Handle missing values
            X = X.fillna(0)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=4)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if model_type == 'xgb':
                    model = xgb.XGBRegressor(
                        random_state=42,
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        verbosity=0
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        max_depth=8,
                        n_jobs=-1
                    )
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                scores.append(rmse)
            
            avg_rmse = np.mean(scores)
            std_rmse = np.std(scores)
            
            return avg_rmse, std_rmse, method_name, len(X.columns)
            
        except Exception as e:
            print(f"Error in {encoding_func.__name__}: {e}")
            return float('inf'), 0, encoding_func.__name__, 0
    
    def run_weather_experiments(self):
        """Run all weather encoding experiments"""
        df = self.explore_weather_patterns()
        
        encoding_methods = [
            self.weather_encoding_1_all_raw,
            self.weather_encoding_2_comfort_score,
            self.weather_encoding_3_reduced_set,
            self.weather_encoding_4_categorical,
            self.weather_encoding_5_interactions,
            self.weather_encoding_6_pca,
            self.weather_encoding_7_seasonal_adjustment,
            self.weather_encoding_8_no_weather,
        ]
        
        for model_type in ['xgb', 'rf']:
            print(f"\n🤖 Testing {model_type.upper()} with different weather encodings...")
            print("-" * 70)
            
            results = []
            
            for encoding_func in encoding_methods:
                rmse, std, method_name, n_features = self.test_weather_method(df, encoding_func, model_type)
                results.append({
                    'method': method_name,
                    'rmse': rmse,
                    'std': std,
                    'features': n_features
                })
                print(f"  {method_name:25s}: {rmse:.3f} ± {std:.3f} ({n_features} features)")
            
            self.results[model_type] = results
        
        self.show_weather_results()
    
    def show_weather_results(self):
        """Show weather encoding results"""
        print("\n" + "=" * 80)
        print("🏆 WEATHER FEATURES ENCODING RESULTS")
        print("=" * 80)
        
        for model_type, results in self.results.items():
            print(f"\n🤖 {model_type.upper()} Results:")
            print("-" * 50)
            
            sorted_results = sorted(results, key=lambda x: x['rmse'])
            
            for i, result in enumerate(sorted_results):
                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"{rank_emoji} {result['method']:25s}: {result['rmse']:6.3f} ± {result['std']:.3f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        
        xgb_results = {r['method']: r['rmse'] for r in self.results.get('xgb', [])}
        rf_results = {r['method']: r['rmse'] for r in self.results.get('rf', [])}
        
        common_methods = set(xgb_results.keys()) & set(rf_results.keys())
        
        if common_methods:
            avg_performance = {}
            for method in common_methods:
                avg_performance[method] = (xgb_results[method] + rf_results[method]) / 2
            
            best_universal = min(avg_performance.items(), key=lambda x: x[1])
            print(f"Best universal method: {best_universal[0]} (avg RMSE: {best_universal[1]:.3f})")
            
            sorted_universal = sorted(avg_performance.items(), key=lambda x: x[1])
            print(f"\nTop universal weather encodings:")
            for i, (method, avg_rmse) in enumerate(sorted_universal[:3]):
                print(f"  {i+1}. {method}: {avg_rmse:.3f} avg RMSE")

def main():
    """Run weather analysis experiments"""
    lab = WeatherAnalysisLab()
    lab.run_weather_experiments()

if __name__ == "__main__":
    main()