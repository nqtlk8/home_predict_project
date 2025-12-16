# Split dataset into train and validation sets
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/cleaned/cleaned_data.csv')
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the datasets
    train_df.to_csv('data/processed/train_cleaned.csv', index=False)
    val_df.to_csv('data/processed/val_cleaned.csv', index=False)
    print("Dataset split into train and validation sets.")