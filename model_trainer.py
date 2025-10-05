import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from pathlib import Path
import json

class ExoplanetModelTrainer:
    """Train a Random Forest classifier for exoplanet disposition prediction."""
    
    def __init__(self, data_path='./preprocessed/preprocessed.csv',
                 model_output_dir='./models'):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        data_path : str
            Path to preprocessed CSV file
        model_output_dir : str
            Directory to save trained models and artifacts
        """
        self.data_path = data_path
        self.model_output_dir = model_output_dir
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare the preprocessed data."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        df = pd.read_csv(self.data_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Separate features and target
        if 'koi_disposition' not in df.columns:
            raise ValueError("Target column 'koi_disposition' not found in data!")
        
        # Feature columns (all except the target)
        self.feature_columns = [col for col in df.columns if col != 'koi_disposition']
        
        X = df[self.feature_columns]
        y = df['koi_disposition']
        
        print(f"\nTarget distribution:")
        print(y.value_counts())
        
        # Handle missing values
        print(f"\nMissing values before handling:")
        print(X.isnull().sum())
        
        # Fill missing values with median for numeric columns
        X = X.fillna(X.median())
        
        print(f"\nMissing values after handling:")
        print(X.isnull().sum())
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nLabel encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label} -> {i}")
        
        return X, y_encoded
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        
    def train_model(self, n_estimators=100, max_depth=None, random_state=42):
        """Train the Random Forest model."""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        print(f"Training Random Forest with {n_estimators} trees...")
        self.model.fit(self.X_train, self.y_train)
        print("Training complete!")
        
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Training accuracy
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        
        # Testing accuracy
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print("-" * 60)
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_test_pred, 
                                   target_names=target_names))
        
        # Confusion matrix
        print("\nConfusion Matrix (Test Set):")
        print("-" * 60)
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(cm)
        
        # Feature importance
        print("\nTop 10 Most Important Features:")
        print("-" * 60)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def save_model(self):
        """Save the trained model and associated artifacts."""
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        
        # Create output directory
        Path(self.model_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(self.model_output_dir, 'random_forest_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save the label encoder
        encoder_path = os.path.join(self.model_output_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Label encoder saved to: {encoder_path}")
        
        # Save feature columns
        features_path = os.path.join(self.model_output_dir, 'feature_columns.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        print(f"Feature columns saved to: {features_path}")
        
        # Save model metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist(),
            'training_samples': len(self.X_train),
            'testing_samples': len(self.X_test)
        }
        
        metadata_path = os.path.join(self.model_output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to: {metadata_path}")
        
    def run_full_pipeline(self, n_estimators=100, max_depth=None, 
                         test_size=0.2, random_state=42):
        """Run the complete training pipeline."""
        print("\n" + "=" * 60)
        print("EXOPLANET CLASSIFICATION MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        self.split_data(X, y, test_size=test_size, random_state=random_state)
        
        # Train model
        self.train_model(n_estimators=n_estimators, max_depth=max_depth, 
                        random_state=random_state)
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Save model
        self.save_model()
        
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nFinal Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
        print(f"\nModel and artifacts saved to: {self.model_output_dir}")
        
        return self.model, metrics


if __name__ == "__main__":
    # Initialize trainer
    trainer = ExoplanetModelTrainer()
    
    # Run full training pipeline
    # You can adjust hyperparameters here
    model, metrics = trainer.run_full_pipeline(
        n_estimators=100,      # Number of trees in the forest
        max_depth=None,        # Maximum depth of trees (None = unlimited)
        test_size=0.2,         # 20% of data for testing
        random_state=42        # For reproducibility
    )
    
    print("\n✓ Model training complete!")
    print("✓ Model saved and ready to use!")