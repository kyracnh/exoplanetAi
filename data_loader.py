import pandas as pd
import os
from pathlib import Path

def load_and_preprocess_data(input_path='./Data/real_data.csv', 
                             output_path='./preprocessed/preprocessed.csv'):
    """
    Load exoplanet data from CSV and extract relevant features for model training.
    
    Parameters:
    -----------
    input_path : str
        Path to the raw data CSV file
    output_path : str
        Path where preprocessed data will be saved
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe with selected features
    """
    
    # Define the features to extract
    feature_columns = [
        "koi_disposition",  # Exoplanet Archive Disposition (target variable)
        "koi_score",        # Disposition Score
        "koi_depth",        # Transit Depth
        "koi_model_snr",    # Signal-to-Noise Ratio
        "koi_period",       # Orbital Period
        "koi_duration",     # Transit Duration
        "koi_prad",         # Planet Radius
        "koi_srad",         # Stellar Radius
        "koi_kepmag",       # Kepler Magnitude
        "koi_teq"           # Equilibrium Temperature
    ]
    
    try:
        # Load the raw data, skipping comment lines that start with #
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path, comment='#', delimiter=',')
        print(f"Original data shape: {df.shape}")
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Extract only the columns we need
        # Check which columns exist in the dataset
        available_columns = [col for col in feature_columns if col in df.columns]
        missing_columns = [col for col in feature_columns if col not in df.columns]
        
        if missing_columns:
            print(f"\nWarning: The following columns are missing: {missing_columns}")
        
        if not available_columns:
            raise ValueError("None of the specified columns were found in the dataset!")
        
        # Select only available columns
        df_preprocessed = df[available_columns].copy()
        print(f"\nExtracted {len(available_columns)} features")
        
        # Display basic statistics
        print(f"Preprocessed data shape: {df_preprocessed.shape}")
        print(f"\nMissing values per column:")
        print(df_preprocessed.isnull().sum())
        
        # Display disposition distribution if available
        if 'koi_disposition' in df_preprocessed.columns:
            print(f"\nDisposition distribution:")
            print(df_preprocessed['koi_disposition'].value_counts())
        
        # Display data types
        print(f"\nData types:")
        print(df_preprocessed.dtypes)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed data
        df_preprocessed.to_csv(output_path, index=False)
        print(f"\nPreprocessed data saved to {output_path}")
        
        return df_preprocessed
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        raise
    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the preprocessing
    preprocessed_df = load_and_preprocess_data()
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nFirst few rows of preprocessed data:")
    print(preprocessed_df.head())
    print(f"\nBasic statistics:")
    print(preprocessed_df.describe())