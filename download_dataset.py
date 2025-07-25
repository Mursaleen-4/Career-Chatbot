"""
Download the actual Career Guidance Dataset from Hugging Face
"""

from datasets import load_dataset
import pandas as pd

def download_career_dataset():
    """
    Download the career guidance dataset from Hugging Face and save as CSV
    """
    print("Downloading career guidance dataset from Hugging Face...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("Pradeep016/career-guidance-qa-dataset")
        
        # Convert to pandas DataFrame
        df = dataset['train'].to_pandas()
        
        print(f"Dataset downloaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Display sample data
        print("\nSample data:")
        print(df.head())
        
        # Check unique roles
        if 'Role' in df.columns:
            print(f"\nNumber of unique roles: {df['Role'].nunique()}")
            print("Roles:", df['Role'].unique()[:10])  # Show first 10 roles
        
        # Save as CSV
        df.to_csv('career_guidance_dataset.csv', index=False)
        print(f"\nDataset saved as 'career_guidance_dataset.csv'")
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_career_dataset()
