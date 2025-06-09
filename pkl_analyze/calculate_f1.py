import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

def calculate_f1_from_csv():
    # Read the CSV file
    try:
        df = pd.read_csv('sample_test.csv')
        
        # Check if required columns exist
        required_columns = ['raw_answer_second_round', 'true_answer']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Column '{col}' not found in the CSV file")
                return
        
        # Get the predictions and true values
        predictions = df['raw_answer_second_round'].values
        true_values = df['true_answer'].values
        
        # Calculate F1 score
        # Note: If the values are not binary, we might need to adjust the average parameter
        f1 = f1_score(true_values, predictions, average='micro')
        
        print(f"F1 Score: {f1:.4f}")
        
        # Print some additional statistics
        print("\nAdditional Statistics:")
        print(f"Number of samples: {len(df)}")
        print("\nValue distribution in predictions:")
        print(pd.Series(predictions).value_counts())
        print("\nValue distribution in true values:")
        print(pd.Series(true_values).value_counts())
        
    except Exception as e:
        print(f"Error reading or processing the file: {str(e)}")

if __name__ == "__main__":
    calculate_f1_from_csv() 