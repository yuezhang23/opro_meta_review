import pandas as pd
import glob

# Read the original CSV files
input_files = ['train.csv']  # This will match train.csv, train_*.csv, etc.
starter = '''# Task
Carry out a comprehensive assessment of the provided reviews focusing on aspects such as soundness, methodology, clarity, relevance, and impact of the research and its contributions as highlighted by the reviewers. Determine whether the accumulated feedback indicates a trend favoring acceptance over rejection, taking into consideration if significant strengths and advancements stated in the reviews supersede the weaknesses and limitations. Conclude your evaluation by stating 'Yes' for acceptance or 'No' for rejection clearly. 

# Output format
Answer Yes or No as labels

# Prediction
Text: 
'''
dfs = []
for file in input_files:
    try:
        temp_df = pd.read_csv(file, sep=',')
        temp_df = temp_df[['raw_prompt', 'true_answer']]

        # for each row string  in df['raw_prompt'], if it starts with starter, remove the starter
        temp_df['raw_prompt'] = temp_df['raw_prompt'].str.replace(starter, '', case=False)
        # Check for duplicates in raw_prompt
        duplicates = temp_df['raw_prompt'].value_counts()
        duplicates = duplicates[duplicates > 1]
        if not duplicates.empty:
            print(f"\nDuplicate raw_prompts found in {file}:")
            print(duplicates)
        dfs.append(temp_df)
    except Exception as e:
        print(f"Error reading {file}: {str(e)}")


# Combine all input DataFrames
df = pd.concat(dfs, ignore_index=True)
# count total number of rows in df
total_rows = len(df)
print(f"Total number of rows in df: {total_rows}")

# for value under true_answer, if it is 'Yes', change it to 1, if it is 'No', change it to 0, make it case insensitive
df['true_answer'] = df['true_answer'].str.lower().map({'yes': 1, 'no': 0})

# change the column name of raw_prompt to text
df.rename(columns={'raw_prompt': 'text'}, inplace=True)

# change the column name of true_answer to label
df.rename(columns={'true_answer': 'label'}, inplace=True)

# Read all comparison CSV files
comparison_files = ['test_200.csv', 'train_800.csv']
comparison_dfs = []

for file in comparison_files:
    try:
        temp_df = pd.read_csv(file, sep=';')
        comparison_dfs.append(temp_df)
    except Exception as e:
        print(f"Error reading {file}: {str(e)}")

# Combine all comparison DataFrames
comparison_df = pd.concat(comparison_dfs, ignore_index=True)

# Print label counts in the filtered comparison DataFrame
print("\nLabel counts before filtering:")
print(comparison_df['label'].value_counts())

# Create a dictionary mapping labels to sets of texts for faster lookup
label_to_texts = {}
for text, label in zip(df['text'], df['label']):
    if label not in label_to_texts:
        label_to_texts[label] = set()
    label_to_texts[label].add(text)

# Filter out rows from comparison_df that exist in df (including substring matches)
mask = ~comparison_df.apply(lambda row: any(
    row['text'] in main_text for main_text in label_to_texts.get(row['label'], set())
), axis=1)
comparison_df = comparison_df[mask]

# Print label counts in the filtered comparison DataFrame
print("\nLabel counts in filtered comparison DataFrame:")
print(comparison_df['label'].value_counts())

# Save the filtered comparison DataFrame back to the file
comparison_df.to_csv('comparison.csv', index=False, sep=';')

