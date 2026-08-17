import pandas as pd
import os
import sys

def merge_kannada_xlsx():
    files = {
        r"C:\Users\premg\OneDrive\Desktop\Major project\testangry.xlsx": "Anger",
        r"C:\Users\premg\OneDrive\Desktop\Major project\testfear.xlsx": "Fear",
        r"C:\Users\premg\OneDrive\Desktop\Major project\testjoy.XLSX": "Happiness",
        r"C:\Users\premg\OneDrive\Desktop\Major project\testsad.xlsx": "Sadness"
    }

    target_csv = r"C:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data\Kannada.csv"

    print("Loading existing Kannada.csv...")
    try:
        df_kan = pd.read_csv(target_csv)
    except Exception as e:
        print(f"Error reading Kannada.csv: {e}")
        df_kan = pd.DataFrame(columns=['text', 'label'])

    original_len = len(df_kan)
    new_rows = []

    # Install openpyxl if needed for excel opening
    try:
        import openpyxl
    except ImportError:
        print("openpyxl is not installed. Installing silently...")
        os.system(f"{sys.executable} -m pip install openpyxl")

    import shutil
    import tempfile

    for file, emotion in files.items():
        print(f"Processing '{os.path.basename(file)}' as Emotion: {emotion}...")
        try:
            # Copy to temp to bypass Windows file locks if User has Excel open
            tmp_path = os.path.join(tempfile.gettempdir(), f"temp_{os.path.basename(file)}")
            shutil.copy2(file, tmp_path)
            
            # Load without header to inspect the data cleanly
            df = pd.read_excel(tmp_path, header=None)
            os.remove(tmp_path)
            
            # If the first cell is purely a header like 'text' or 'sentences', drop the first row
            first_val = str(df.iloc[0, 0]).lower()
            if first_val in ['text', 'sentence', 'sentences', 'data', 'kannada']:
                df = df.iloc[1:]
                
            # Treat the very first column as the text
            for item in df.iloc[:, 0].dropna():
                text_clean = str(item).strip()
                if text_clean:
                    new_rows.append({'text': text_clean, 'label': emotion})
                    
        except Exception as e:
             print(f"Error reading {file}: {e}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Append to the original Kannada dataset
        df_merged = pd.concat([df_kan, df_new], ignore_index=True)
        # Drop duplicates if there are identical sentences
        df_merged = df_merged.drop_duplicates(subset=['text'])
        
        df_merged.to_csv(target_csv, index=False)
        print(f"\nMerge Complete! Successfully augmented Kannada.csv.")
        print(f"Original Kannada Rows: {original_len}")
        print(f"Total rows added from Excel: {len(df_new)}")
        print(f"Total Unique Rows Now: {len(df_merged)}")
        
        print("\nDistribution of the new unified Kannada dataset:")
        print(df_merged['label'].value_counts())
    else:
        print("\nNo rows added. Check file contents.")

if __name__ == "__main__":
    merge_kannada_xlsx()
