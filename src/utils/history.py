import pandas as pd
import os
import random
from datetime import datetime, timedelta

HISTORY_FILE = "data/history.csv"
DATASET_FILE = "data/dataset/dataset.csv"

def load_history():
    """
    Loads prediction history. 
    If history is empty, it attempts to import from the dataset.csv first.
    """
    # If history doesn't exist, try to create it from the dataset
    if not os.path.exists(HISTORY_FILE):
        import_dataset_to_history()
        
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    except Exception:
        pass
        
    return pd.DataFrame(columns=["Date", "Subject", "Snippet", "Prediction", "Confidence"])

def import_dataset_to_history():
    """
    Reads the training dataset and converts it into a history file
    with simulated timestamps for visualization purposes.
    """
    if not os.path.exists(DATASET_FILE):
        return

    try:
        # Read dataset (Assuming columns: Category, Message)
        # We try standard encodings for spam datasets
        try:
            df_source = pd.read_csv(DATASET_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            df_source = pd.read_csv(DATASET_FILE, encoding='latin-1')

        # Rename columns if needed to match expected format
        if 'v1' in df_source.columns: 
            df_source = df_source.rename(columns={'v1': 'Category', 'v2': 'Message'})
        
        # Ensure we have the required columns
        if 'Category' not in df_source.columns or 'Message' not in df_source.columns:
            return

        # Prepare new data
        history_data = []
        base_date = datetime.now()
        
        # We'll take a sample if the dataset is huge (e.g., limit to 2000 for performance)
        # or take all if you prefer. Let's take up to 3000 rows.
        sample_df = df_source.sample(min(len(df_source), 3000), random_state=42)

        for _, row in sample_df.iterrows():
            # Simulate a date within the last 30 days
            days_ago = random.randint(0, 30)
            minutes_ago = random.randint(0, 1440)
            fake_date = base_date - timedelta(days=days_ago, minutes=minutes_ago)
            
            # Map Category to Prediction format
            category = row['Category'].lower()
            prediction = "Spam" if category == "spam" else "Ham"
            
            # Simulate confidence (Spam usually has higher confidence in datasets)
            confidence = random.uniform(85, 99) if prediction == "Spam" else random.uniform(70, 99)

            history_data.append({
                "Date": fake_date,
                "Subject": "Imported from Dataset", # Placeholder subject
                "Snippet": str(row['Message'])[:100], # First 100 chars
                "Prediction": prediction,
                "Confidence": round(confidence, 1)
            })
            
        # Save to history.csv
        new_df = pd.DataFrame(history_data)
        new_df = new_df.sort_values(by="Date", ascending=False)
        
        os.makedirs("data", exist_ok=True)
        new_df.to_csv(HISTORY_FILE, index=False)
        
    except Exception as e:
        print(f"Error importing dataset: {e}")

def save_prediction(subject, snippet, prediction, confidence, date=None):
    """Saves a new prediction to the history file."""
    if date is None:
        date = datetime.now()
        
    new_data = {
        "Date": date,
        "Subject": subject,
        "Snippet": snippet,
        "Prediction": prediction,
        "Confidence": confidence
    }
    
    # Load existing (which might trigger the import if missing)
    df = load_history()
    df = pd.concat([pd.DataFrame([new_data]), df], ignore_index=True)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv(HISTORY_FILE, index=False)