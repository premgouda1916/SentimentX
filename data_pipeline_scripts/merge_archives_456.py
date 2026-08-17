"""
Merge archive 4, 5, 6 into respective language CSVs.
- Archive 4 (Arrow, Malayalam)  → data/Malayalam.csv
- Archive 5 (CSV, Hindi emoHi)  → data/Hindi.csv
- Archive 6 (CSV, English)      → data/English.csv
"""
import os
import glob
import pandas as pd
import pyarrow.ipc as ipc
import pyarrow as pa

# ═══════════════════════════════════════════════
# ARCHIVE 4 → Malayalam.csv  (Arrow format, same as archives 1-3)
# ═══════════════════════════════════════════════

emotion_map_arrow = {
    'HAPPY': 'Happiness',
    'SAD': 'Sadness',
    'ANGER': 'Anger',
    'FEAR': 'Fear',
    'SURPRISE': 'Surprise',
    'DISGUST': 'Disgust'
}

def extract_from_arrow_archive(archive_path, target_csv):
    print(f"\n{'='*60}")
    print(f"Extracting from {archive_path}")
    print(f"  → Target: {target_csv}")
    print(f"{'='*60}")
    
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
                        b_text = batch['text'].to_pylist()
                        b_style = batch['style'].to_pylist()
                        
                        for t, s in zip(b_text, b_style):
                            t_clean = str(t).strip()
                            s_clean = emotion_map_arrow.get(str(s).strip().upper(), None)
                            if t_clean and s_clean:
                                texts.append(t_clean)
                                labels.append(s_clean)
                except Exception as e:
                    print(f"  Error reading batch in {file_name}: {e}")
                    
    if not texts:
        print("  No matches detected!")
        return
        
    df_new = pd.DataFrame({'text': texts, 'label': labels})
    df_merged = pd.concat([df_target, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['text'])
    df_merged.to_csv(target_csv, index=False)
    
    print(f"  Extracted rows: {len(df_new)}")
    print(f"  Original rows:  {original_len}")
    print(f"  New unique total: {len(df_merged)}")
    print(f"  Label distribution:")
    print(f"  {df_merged['label'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════
# ARCHIVE 5 → Hindi.csv  (GoEmotions Hindi - emoHi)
# Labels are GoEmotions 28-class → map to Ekman 6
# ═══════════════════════════════════════════════

# GoEmotions label indices → Ekman mapping
go_to_ekman = {
    0: 'Happiness',    # admiration
    1: 'Happiness',    # amusement
    2: 'Anger',        # anger
    3: 'Anger',        # annoyance
    4: 'Happiness',    # approval
    5: 'Happiness',    # caring
    6: None,           # confusion (skip)
    7: None,           # curiosity (skip)
    8: None,           # desire (skip)
    9: 'Sadness',      # disappointment
    10: 'Anger',       # disapproval
    11: 'Disgust',     # disgust
    12: None,          # embarrassment (skip)
    13: 'Happiness',   # excitement
    14: 'Fear',        # fear
    15: 'Happiness',   # gratitude
    16: 'Sadness',     # grief
    17: 'Happiness',   # joy
    18: 'Happiness',   # love
    19: 'Fear',        # nervousness
    20: 'Happiness',   # optimism
    21: 'Happiness',   # pride
    22: 'Surprise',    # realization
    23: 'Happiness',   # relief
    24: 'Sadness',     # remorse
    25: 'Sadness',     # sadness
    26: 'Surprise',    # surprise
    27: None,          # neutral (skip)
}

def parse_go_label(label_str):
    """Parse a GoEmotions label like '[27]' or '[ 1 17]' and return the primary Ekman emotion."""
    label_str = label_str.strip().strip('[]')
    nums = label_str.split()
    # Use first valid mapped label
    for n in nums:
        try:
            idx = int(n.strip())
            ekman = go_to_ekman.get(idx, None)
            if ekman is not None:
                return ekman
        except ValueError:
            continue
    return None

def extract_emohi(archive_dir, target_csv):
    print(f"\n{'='*60}")
    print(f"Extracting from {archive_dir} (emoHi Hindi)")
    print(f"  → Target: {target_csv}")
    print(f"{'='*60}")
    
    try:
        df_target = pd.read_csv(target_csv)
    except Exception:
        df_target = pd.DataFrame(columns=['text', 'label'])
    
    original_len = len(df_target)
    texts = []
    labels = []
    
    for csv_file in ['emoHi-train.csv', 'emoHi-test.csv', 'emoHi-valid.csv']:
        path = os.path.join(archive_dir, csv_file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"  Processing {csv_file}: {len(df)} rows")
        
        for _, row in df.iterrows():
            text = str(row['text']).strip()
            ekman = parse_go_label(str(row['labels']))
            if text and ekman:
                texts.append(text)
                labels.append(ekman)
    
    if not texts:
        print("  No matches detected!")
        return
    
    df_new = pd.DataFrame({'text': texts, 'label': labels})
    df_merged = pd.concat([df_target, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['text'])
    df_merged.to_csv(target_csv, index=False)
    
    print(f"  Extracted rows: {len(df_new)}")
    print(f"  Original rows:  {original_len}")
    print(f"  New unique total: {len(df_merged)}")
    print(f"  Label distribution:")
    print(f"  {df_merged['label'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════
# ARCHIVE 6 → English.csv  (HuggingFace emotion dataset)
# Labels: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
# ═══════════════════════════════════════════════

emotion_map_int = {
    0: 'Sadness',
    1: 'Happiness',
    2: 'Happiness',   # love → Happiness
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise',
}

def extract_emotion_csv(archive_dir, target_csv):
    print(f"\n{'='*60}")
    print(f"Extracting from {archive_dir} (English emotion)")
    print(f"  → Target: {target_csv}")
    print(f"{'='*60}")
    
    try:
        df_target = pd.read_csv(target_csv)
    except Exception:
        df_target = pd.DataFrame(columns=['text', 'label'])
    
    original_len = len(df_target)
    texts = []
    labels = []
    
    for csv_file in ['training.csv', 'test.csv', 'validation.csv']:
        path = os.path.join(archive_dir, csv_file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"  Processing {csv_file}: {len(df)} rows")
        
        for _, row in df.iterrows():
            text = str(row['text']).strip()
            ekman = emotion_map_int.get(row['label'], None)
            if text and ekman:
                texts.append(text)
                labels.append(ekman)
    
    if not texts:
        print("  No matches detected!")
        return
    
    df_new = pd.DataFrame({'text': texts, 'label': labels})
    df_merged = pd.concat([df_target, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['text'])
    df_merged.to_csv(target_csv, index=False)
    
    print(f"  Extracted rows: {len(df_new)}")
    print(f"  Original rows:  {original_len}")
    print(f"  New unique total: {len(df_merged)}")
    print(f"  Label distribution:")
    print(f"  {df_merged['label'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════

# Archive 4 → Malayalam
extract_from_arrow_archive(
    r"c:\Users\premg\OneDrive\Desktop\Major project\archive 4",
    r"c:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data\Malayalam.csv"
)

# Archive 5 → Hindi  
extract_emohi(
    r"c:\Users\premg\OneDrive\Desktop\Major project\archive 5",
    r"c:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data\Hindi.csv"
)

# Archive 6 → English
extract_emotion_csv(
    r"c:\Users\premg\OneDrive\Desktop\Major project\archive 6",
    r"c:\Users\premg\OneDrive\Desktop\Major project\SentimentX\data\English.csv"
)

print(f"\n{'='*60}")
print("All archives merged successfully!")
print(f"{'='*60}")
