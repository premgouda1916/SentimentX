import pandas as pd
import os
import glob

data_dir = r"c:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data"
output_path = os.path.join(data_dir, "dataset.csv")

# List of all individual language CSVs to merge
language_files = glob.glob(os.path.join(data_dir, "*.csv"))

# Remove the output file from the list if it already exists
if output_path in language_files:
    language_files.remove(output_path)

all_dfs = []

print("Compiling final dataset...")
for file in language_files:
    lang_name = os.path.basename(file).replace(".csv", "")
    df = pd.read_csv(file)
    
    # Ensure standard schema: text, label, language
    if 'language' not in df.columns:
        df['language'] = lang_name
        
    all_dfs.append(df)
    print(f"Loaded {len(df)} rows from {lang_name}")

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Optional: shuffle the dataset for better training distribution
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.to_csv(output_path, index=False)
    print("\n==================================")
    print(f"SUCCESS! Combined {len(final_df)} rows into dataset.csv")
    print("==================================")
    print("\nClass Distribution:")
    print(final_df['label'].value_counts())
    print("\nLanguage Distribution:")
    print(final_df['language'].value_counts())
else:
    print("No CSV files found to compile.")
