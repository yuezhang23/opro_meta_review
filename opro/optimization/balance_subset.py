import pandas as pd
import numpy as np

def create_balanced_subset(train_exs, cnt: int) -> list:
    # Convert train_exs to DataFrame
    data = {
        'id': [ex['id'] for ex in train_exs],
        'text': [ex['text'] for ex in train_exs],
        'label': [ex['label'] for ex in train_exs]
    }
    df = pd.DataFrame(data)

    # Verify the columns
    print("Columns in the dataset:", df.columns.tolist())

    # Count samples per label
    label_counts = df['label'].value_counts()
    print(f"Original label distribution:\n{label_counts}")

    # Select 100 unique samples for each label (0 and 1) based on unique IDs
    balanced_samples = []
    for label in [1, 0]:
        # Get samples for current label
        label_df = df[df['label'] == label]
        # Select 100 unique samples based on ID
        selected_samples = label_df.drop_duplicates(subset=['id']).sample(n=100)
        balanced_samples.append(selected_samples)

    # Combine the balanced samples
    balanced_df = pd.concat(balanced_samples)

    # Shuffle the rows
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    # Convert back to train_exs format
    balanced_exs = []
    for _, row in balanced_df.iterrows():
        # Create a new example with the same dictionary format as input train_exs
        balanced_exs.append({
            'id': row['id'],
            'text': row['text'],
            'label': row['label']
        })
    
    return balanced_exs

# # Example usage:
# if __name__ == "__main__":
#     # This is just for testing - in practice, train_exs will be passed from main.py
#     from tasks import MetareviewerBinaryTask
#     task = MetareviewerBinaryTask('data/', 8)
#     train_exs = task.get_train_examples('data/metareviewer_data_train_800.csv')
#     balanced_exs = create_balanced_subset(train_exs, cnt=1)
#     print(f"Created balanced subset with {len(balanced_exs)} examples")
