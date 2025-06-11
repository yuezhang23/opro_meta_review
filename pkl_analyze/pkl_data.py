import pandas as pd
import pickle
import random
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add the parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def sample_pkl_data(pkl_file_path: str, output_csv_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Sample data from a pickle file based on Yes/No counts in the true_answer column.
    
    Args:
        pkl_file_path (str): Path to the pickle file
        output_csv_path (str, optional): Path to save the output CSV. If None, will use pkl_file_path with .csv extension
    
    Returns:
        Tuple[bool, str]: (Success status, Message)
    """
    try:
        # Load the pickle file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f'data.keys(): {data.keys()}')
        
        # Check if the required dictionary exists
        if 'eval_detailed_results_df_dict' not in data:
            return False, "Key 'eval_detailed_results_df_dict' not found in pickle file"
        
        df_dict_all = data['eval_detailed_results_df_dict']

        # Get the first dataframe from the dictionary
        df = next(iter(df_dict_all.values()))
        
        # Save the dataframe to a CSV file
        df.to_csv(output_csv_path, index=False)
        
        return True, f"Data saved to {output_csv_path}"
        
    except Exception as e:
        return False, f"Error processing pickle file: {str(e)}"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample data from pickle file based on Yes/No counts')
    parser.add_argument('pkl_file', help='Path to the pickle file')
    parser.add_argument('--output', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    success, message = sample_pkl_data(args.pkl_file, args.output)
    print(message)
    
    if not success:
        exit(1)

if __name__ == '__main__':
    main() 