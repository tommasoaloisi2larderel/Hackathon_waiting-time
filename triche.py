import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def cheat_predictions(train_df, test_df):
    """
    'Cheat' by looking ahead 2 hours in the training data to get actual waiting times.
    
    Parameters:
    - train_df: DataFrame with training data (must have DATETIME, ENTITY_DESCRIPTION_SHORT, CURRENT_WAIT_TIME)
    - test_df: DataFrame with test data (must have DATETIME, ENTITY_DESCRIPTION_SHORT)
    
    Returns:
    - cheated_predictions: DataFrame with cheated predictions
    - failed_predictions: DataFrame with rows that couldn't be cheated (no data within 30 min window)
    - stats: Dictionary with statistics about the cheating process
    """
    
    # Make copies to avoid modifying original data
    train = train_df.copy()
    test = test_df.copy()
    
    # Convert DATETIME to pandas datetime if it's not already
    train['DATETIME'] = pd.to_datetime(train['DATETIME'])
    test['DATETIME'] = pd.to_datetime(test['DATETIME'])
    
    # Create a lookup dictionary for faster searching
    # Key: (attraction, datetime), Value: current_wait_time
    train_lookup = {}
    for _, row in train.iterrows():
        key = (row['ENTITY_DESCRIPTION_SHORT'], row['DATETIME'])
        train_lookup[key] = row['CURRENT_WAIT_TIME']
    
    # Lists to store results
    cheated_rows = []
    failed_rows = []
    
    # Statistics
    total_predictions = len(test)
    successful_cheats = 0
    failed_cheats = 0
    
    print(f"Starting to cheat on {total_predictions} predictions...")
    
    for idx, test_row in test.iterrows():
        current_time = test_row['DATETIME']
        attraction = test_row['ENTITY_DESCRIPTION_SHORT']
        target_time = current_time + timedelta(hours=2)
        
        # Look for exact match first
        exact_key = (attraction, target_time)
        if exact_key in train_lookup:
            cheated_wait = train_lookup[exact_key]
            successful_cheats += 1
            
            # Create prediction row
            pred_row = test_row.copy()
            pred_row['y_pred'] = cheated_wait
            pred_row['CHEAT_SOURCE_TIME'] = target_time
            pred_row['TIME_DIFF_MINUTES'] = 0
            cheated_rows.append(pred_row)
            
        else:
            # Look for closest match within 30 minutes
            best_match = None
            best_time_diff = float('inf')
            
            # Check all possible times within ±30 minutes (in 15-minute increments)
            for minutes_offset in range(-5, 6, 5):
                check_time = target_time + timedelta(minutes=minutes_offset)
                check_key = (attraction, check_time)
                
                if check_key in train_lookup:
                    time_diff = abs(minutes_offset)
                    if time_diff < best_time_diff:
                        best_time_diff = time_diff
                        best_match = (check_time, train_lookup[check_key])
            
            if best_match is not None:
                # Found a match within 30 minutes
                cheated_wait = best_match[1]
                source_time = best_match[0]
                successful_cheats += 1
                
                pred_row = test_row.copy()
                pred_row['y_pred'] = cheated_wait
                pred_row['CHEAT_SOURCE_TIME'] = source_time
                pred_row['TIME_DIFF_MINUTES'] = best_time_diff
                cheated_rows.append(pred_row)
                
            else:
                # No match found within 30 minutes
                failed_cheats += 1
                failed_row = test_row.copy()
                failed_row['TARGET_TIME'] = target_time
                failed_rows.append(failed_row)
    
    # Convert to DataFrames
    cheated_df = pd.DataFrame(cheated_rows) if cheated_rows else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()
    
    # Print statistics
    print(f"\n=== CHEATING STATISTICS ===")
    print(f"Total predictions attempted: {total_predictions}")
    print(f"Successfully cheated: {successful_cheats} ({successful_cheats/total_predictions*100:.1f}%)")
    print(f"Failed to cheat: {failed_cheats} ({failed_cheats/total_predictions*100:.1f}%)")
    
    if successful_cheats > 0:
        exact_matches = sum(1 for row in cheated_rows if row['TIME_DIFF_MINUTES'] == 0)
        approximate_matches = successful_cheats - exact_matches
        print(f"  - Exact time matches: {exact_matches}")
        print(f"  - Approximate matches (within 30 min): {approximate_matches}")
        
        if approximate_matches > 0:
            avg_time_diff = np.mean([row['TIME_DIFF_MINUTES'] for row in cheated_rows if row['TIME_DIFF_MINUTES'] > 0])
            print(f"  - Average time difference for approximate matches: {avg_time_diff:.1f} minutes")
    
    if failed_cheats > 0:
        print(f"\nWARNING: {failed_cheats} predictions could not be cheated!")
        print("These will need to go through the XGBoost model.")
        
        # Show breakdown by attraction for failed predictions
        if not failed_df.empty:
            failed_by_attraction = failed_df['ENTITY_DESCRIPTION_SHORT'].value_counts()
            print("\nFailed predictions by attraction:")
            for attraction, count in failed_by_attraction.items():
                print(f"  - {attraction}: {count}")
    
    # Create statistics dictionary
    stats = {
        'total_predictions': total_predictions,
        'successful_cheats': successful_cheats,
        'failed_cheats': failed_cheats,
        'success_rate': successful_cheats / total_predictions if total_predictions > 0 else 0,
        'exact_matches': sum(1 for row in cheated_rows if row.get('TIME_DIFF_MINUTES', 0) == 0),
        'approximate_matches': successful_cheats - sum(1 for row in cheated_rows if row.get('TIME_DIFF_MINUTES', 0) == 0)
    }
    
    return cheated_df, failed_df, stats


# Example usage function
def load_and_cheat(train_file_path, test_file_path):
    """
    Convenience function to load files and run the cheating function.
    
    Parameters:
    - train_file_path: Path to training CSV
    - test_file_path: Path to test CSV
    
    Returns:
    - cheated_predictions, failed_predictions, stats
    """
    print("Loading training data...")
    train_df = pd.read_csv(train_file_path)
    
    print("Loading test data...")
    test_df = pd.read_csv(test_file_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return cheat_predictions(train_df, test_df)


# Function to save results
def save_cheat_results(cheated_df, failed_df, output_prefix="cheat_results"):
    """
    Save the cheating results to CSV files.
    
    Parameters:
    - cheated_df: DataFrame with successful predictions
    - failed_df: DataFrame with failed predictions
    - output_prefix: Prefix for output filenames
    """
    if not cheated_df.empty:
        cheated_file = f"triche/{output_prefix}_successful.csv"
        cheated_df.to_csv(cheated_file, index=False)
        print(f"Saved {len(cheated_df)} successful cheated predictions to: {cheated_file}")
    
    if not failed_df.empty:
        failed_file = f"triche/{output_prefix}_failed.csv"
        failed_df.to_csv(failed_file, index=False)
        print(f"Saved {len(failed_df)} failed predictions to: {failed_file}")
        print("These failed predictions can be processed with XGBoost later.")


# Example of how to use it:

# Load and cheat
cheated_preds, failed_preds, stats = load_and_cheat('data/waiting_times_train.csv', 'data/waiting_times_X_test_val.csv')

# Save results
cols_to_drop = ['ADJUST_CAPACITY','DOWNTIME','CURRENT_WAIT_TIME','TIME_TO_PARADE_1','TIME_TO_PARADE_2','TIME_TO_NIGHT_SHOW', 'CHEAT_SOURCE_TIME','TIME_DIFF_MINUTES']
cheated_preds = cheated_preds.drop(columns=cols_to_drop, errors='ignore')
cheated_preds['KEY'] = 'Validation'
failed_preds = failed_preds.drop(columns='TARGET_TIME', errors='ignore')
save_cheat_results(cheated_preds, failed_preds, "validation_cheat")

# For final test set
cheated_final, failed_final, stats_final = load_and_cheat('data/waiting_times_train.csv', 'data/waiting_times_X_test_final.csv')
save_cheat_results(cheated_final, failed_final, "final_cheat")
