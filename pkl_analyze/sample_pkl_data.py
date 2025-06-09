import pandas as pd
import pickle
import random
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

def sample_pkl_data(pkl_file_path: str, number_yes: int, number_no: int, output_csv_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Sample data from a pickle file based on Yes/No counts in the true_answer column.
    
    Args:
        pkl_file_path (str): Path to the pickle file
        number_yes (int): Number of 'Yes' samples to retrieve
        number_no (int): Number of 'No' samples to retrieve
        output_csv_path (str, optional): Path to save the output CSV. If None, will use pkl_file_path with .csv extension
    
    Returns:
        Tuple[bool, str]: (Success status, Message)
    """
    try:
        # Load the pickle file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if the required dictionary exists
        if 'eval_detailed_results_df_dict' not in data:
            return False, "Key 'eval_detailed_results_df_dict' not found in pickle file"
        
        df_dict_all = data['eval_detailed_results_df_dict']

        # exampler
        example_key = "Conduct a detailed examination of the reviews pertaining to the academic paper, emphasizing the paper's originality, methodological excellence, and empirical validations. Evaluate whether the strengths highlighted by the reviewers distinctly surpass the identified weaknesses, particularly in relation to clarity of presentation and its contribution to the research area. Make a final decision regarding acceptance (Yes) or rejection (No), grounded in a balanced yet optimistic interpretation of the paper’s potential significance and implications on future research directions. Use affirmative language that reflects positivity and reassurance about the paper’s capabilities to advance its field."
        df = df_dict_all[example_key]
        
        # Check if 'true_answer' exists in the dictionary
        if 'true_answer' not in df:
            return False, "Key 'true_answer' not found in eval_detailed_results_df_dict"
        
        # Verify that values are either 'Yes' or 'No'
        unique_values = df['true_answer'].unique()
        if not all(val in ['Yes', 'No'] for val in unique_values):
            return False, f"Unexpected values in true_answer column: {unique_values}"
        
        # Sample the data
        yes_samples = df[df['true_answer'] == 'Yes'].sample(n=min(number_yes, len(df[df['true_answer'] == 'Yes'])))
        no_samples = df[df['true_answer'] == 'No'].sample(n=min(number_no, len(df[df['true_answer'] == 'No'])))
        
        # Combine samples
        sampled_df = pd.concat([yes_samples, no_samples])
        
        # Generate output path if not provided
        if output_csv_path is None:
            output_csv_path = str(Path(pkl_file_path).with_suffix('.csv'))
        
        # Save to CSV
        sampled_df.to_csv(output_csv_path, index=False)
        
        return True, f"Successfully sampled and saved data to {output_csv_path}. Sampled {len(yes_samples)} 'Yes' and {len(no_samples)} 'No' entries."
        
    except Exception as e:
        return False, f"Error processing pickle file: {str(e)}"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample data from pickle file based on Yes/No counts')
    parser.add_argument('pkl_file', help='Path to the pickle file')
    parser.add_argument('number_yes', type=int, help='Number of Yes samples to retrieve')
    parser.add_argument('number_no', type=int, help='Number of No samples to retrieve')
    parser.add_argument('--output', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    success, message = sample_pkl_data(args.pkl_file, args.number_yes, args.number_no, args.output)
    print(message)
    
    if not success:
        exit(1)

if __name__ == '__main__':
    main() 