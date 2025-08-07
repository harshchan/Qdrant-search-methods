import pandas as pd
import re
import numpy as np
from typing import List, Dict, Any
import string

def clean_text(text: str) -> str:
    """Clean text by lowercasing, removing punctuation, and normalizing whitespace"""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the e-commerce dataset"""
    df = pd.read_csv(csv_path)
    
    # Combine relevant text fields
    text_columns = ['title', 'description', 'attributes']
    available_columns = [col for col in text_columns if col in df.columns]
    
    df['combined_text'] = df[available_columns].fillna('').apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1
    )
    
    # Clean the combined text
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    return df

if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data("data/shoe_products_dataset.csv")
    print(f"Loaded {len(df)} products")
    print(f"Sample cleaned text: {df['cleaned_text'].iloc[0]}")