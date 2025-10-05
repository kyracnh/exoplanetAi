import pandas as pd
import numpy as np
import joblib
import json
import os

class ExoplanetPredictor:
    """Make predictions using the trained Random Forest model."""
    
    def __init__(self, model_dir='./models'):
        """
        Initialize the predictor by loading the trained model.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing the trained model files
        """
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and associated artifacts."""
        print("Loading trained model...")
        
        # Load model
        model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load label encoder
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        self.label_encoder = joblib.load(encoder_path)
        print(f"✓ Label encoder loaded")
        
        # Load feature columns
        features_path = os.path.join(self.model_dir, 'feature_columns.json')
        with open(features_path, 'r') as f:
            self.feature_columns = json.load(f)
        print(f"✓ Feature columns loaded ({len(self.feature_columns)} features)")
        
        print(f"\nModel is ready! Can predict: {list(self.label_encoder.classes_)}")
        
    def predict(self, data):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        data : pd.DataFrame or dict
            Input data with the same features used for training
            
        Returns:
        --------
        dict
            Predictions with labels and probabilities
        """
        # Convert dict to DataFrame if necessary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features correctly
        X = data[self.feature_columns]
        
        # Handle missing values (fill with median from training would be better)
        X = X.fillna(X.median())
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Decode predictions
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        results = []
        for i in range(len(data)):
            result = {
                'prediction': predicted_labels[i],
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.label_encoder.classes_, probabilities[i])
                },
                'confidence': float(np.max(probabilities[i]))
            }
            results.append(result)
        
        return results if len(results) > 1 else results[0]
    
    def predict_from_csv(self, csv_path, output_path=None):
        """
        Make predictions on data from a CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file with input data
        output_path : str, optional
            Path to save predictions (if None, only returns results)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        print(f"Loading data from {csv_path}...")
        data = pd.read_csv(csv_path)
        print(f"Loaded {len(data)} samples")
        
        # Make predictions
        print("Making predictions...")
        results = self.predict(data)
        
        # If single result, convert to list
        if not isinstance(results, list):
            results = [results]
        
        # Create results DataFrame
        predictions_df = pd.DataFrame([
            {
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                **{f'prob_{label}': r['probabilities'][label] 
                   for label in self.label_encoder.classes_}
            }
            for r in results
        ])
        
        # Combine with original data
        output_df = pd.concat([data, predictions_df], axis=1)
        
        # Save if output path provided
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return output_df


def predict_single_example():
    """Example of making a single prediction."""
    print("=" * 60)
    print("SINGLE PREDICTION EXAMPLE")
    print("=" * 60)
    
    predictor = ExoplanetPredictor()
    
    # Example data point
    example = {
        'koi_score': 0.9,
        'koi_depth': 1000.0,
        'koi_model_snr': 50.0,
        'koi_period': 10.5,
        'koi_duration': 3.2,
        'koi_prad': 2.1,
        'koi_srad': 1.0,
        'koi_kepmag': 14.5,
        'koi_teq': 500.0
    }
    
    print("\nInput data:")
    for key, value in example.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict(example)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Predicted class: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print("\nClass probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")


def predict_from_file(input_csv, output_csv='./predictions/predictions.csv'):
    """Example of making predictions from a CSV file."""
    print("=" * 60)
    print("BATCH PREDICTION FROM CSV")
    print("=" * 60)
    
    predictor = ExoplanetPredictor()
    results = predictor.predict_from_csv(input_csv, output_csv)
    
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\nTotal predictions: {len(results)}")
    print(f"\nPrediction distribution:")
    print(results['prediction'].value_counts())
    print(f"\nAverage confidence: {results['confidence'].mean():.4f}")
    
    return results


if __name__ == "__main__":
    # Run single prediction example
    predict_single_example()
    
    print("\n" + "=" * 60)
    print("\nTo make predictions on new data:")
    print("  1. For single prediction: use predict_single_example()")
    print("  2. For batch predictions: use predict_from_file('your_data.csv')")
    print("  3. Or use the ExoplanetPredictor class directly in your code")
    print("=" * 60)