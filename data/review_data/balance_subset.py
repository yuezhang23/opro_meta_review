import pandas as pd
import numpy as np

# Read the CSV file with semicolon separator
df = pd.read_csv('review_train.csv', sep=';')

# Verify the columns
print("Columns in the dataset:", df.columns.tolist())

# Count samples per label
label_counts = df['label'].value_counts()
print(f"Original label distribution:\n{label_counts}")

# Define different class count configurations
# Each tuple represents (positive_class_count, negative_class_count, output_filename)
configurations = [
    # (100, 100, '100+100_neurips_2023_test.csv'),
    # (40, 160, '40+160_neurips_2023_test.csv'),
    # (160, 40, '160+40_neurips_2023_test.csv'),
    # (178, 22, '178+22_neurips_2023_test.csv'),
    (100, 100, '100+100_neurips_2024_train.csv')
]

for pos_count, neg_count, output_file in configurations:
    print(f"\nGenerating balanced dataset with {pos_count} positive and {neg_count} negative samples...")
    
    balanced_samples = []
    # Get positive samples
    pos_df = df[df['label'] == 1]
    pos_samples = pos_df.drop_duplicates(subset=['id']).sample(n=pos_count)
    balanced_samples.append(pos_samples)
    
    # Get negative samples
    neg_df = df[df['label'] == 0]
    neg_samples = neg_df.drop_duplicates(subset=['id']).sample(n=neg_count)
    balanced_samples.append(neg_samples)

    # Combine the balanced samples
    balanced_df = pd.concat(balanced_samples)

    # Shuffle the rows
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    # Save to CSV file
    balanced_df.to_csv(output_file, sep=';', index=False)
    print(f"Saved balanced dataset to {output_file}")

# Get the remaining data (data not in balanced_df)
# remaining_df = df[~df['id'].isin(balanced_df['id'])]

# Save to two separate CSV files
# remaining_output = '700+700_iclr_2025_train.csv'

# remaining_df.to_csv(remaining_output, sep=';', index=False)
