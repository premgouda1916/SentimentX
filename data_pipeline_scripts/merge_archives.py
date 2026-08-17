import os
import glob
import pandas as pd
import pyarrow.ipc as ipc
import pyarrow as pa

emotion_map = {
    'HAPPY': 'Happiness',
    'SAD': 'Sadness',
    'ANGER': 'Anger',
    'FEAR': 'Fear',
    'SURPRISE': 'Surprise',
    'DISGUST': 'Disgust'
}

def extract_from_archive(archive_path, target_csv):
    print(f"\n--- Extracting from {archive_path} to {target_csv} ---")
    
    # Load existing
    try:
        df_target = pd.read_csv(target_csv)
    except Exception:
        df_target = pd.DataFrame(columns=['text', 'label'])
        
    original_len = len(df_target)
    
    texts = []
    labels = []
    
    for split in ['train', 'test']:
        split_dir = os.path.join(archive_path, split)
        if not os.path.exists(split_dir): continue
        
        arrow_files = glob.glob(os.path.join(split_dir, "*.arrow"))
        for file in arrow_files:
            file_name = os.path.basename(file)
            with pa.memory_map(file, 'r') as source:
                reader = ipc.RecordBatchStreamReader(source)
                try:
                    for batch in reader:
                        # Select only text and style explicitly to save RAM
                        b_text = batch['text'].to_pylist()
                        b_style = batch['style'].to_pylist()
                        
                        for t, s in zip(b_text, b_style):
                            t_clean = str(t).strip()
                            s_clean = emotion_map.get(str(s).strip().upper(), None)
                            if t_clean and s_clean:
                                texts.append(t_clean)
                                labels.append(s_clean)
                except Exception as e:
                    print(f"Error reading batch in {file_name}: {e}")
                    
    if not texts:
        print("No matches detected!")
        return
        
    df_new = pd.DataFrame({'text': texts, 'label': labels})
    df_merged = pd.concat([df_target, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['text'])
    df_merged.to_csv(target_csv, index=False)
    
    print(f"Extracted rows: {len(df_new)}")
    print(f"Original {os.path.basename(target_csv)} Rows: {original_len}")
    print(f"New Extrapolated Unique Total: {len(df_merged)}")

# Telugu
extract_from_archive(r"c:\Users\premg\OneDrive\Desktop\Major project\archive (1)", 
                     r"c:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data\Telugu.csv")

# Marathi
extract_from_archive(r"c:\Users\premg\OneDrive\Desktop\Major project\archive (2)", 
                     r"c:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data\Marathi.csv")

# Kannada
extract_from_archive(r"c:\Users\premg\OneDrive\Desktop\Major project\archive (3)", 
                     r"c:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data\Kannada.csv")

# Archive 4 (Malayalam), Archive 5 (Hindi), Archive 6 (English) 
# are handled by merge_archives_456.py (different formats: Arrow, GoEmotions CSV, HF emotion CSV)
