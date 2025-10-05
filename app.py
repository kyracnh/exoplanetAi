from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from model_predictor import ExoplanetPredictor
from pathlib import Path
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)

# Initialize predictor
try:
    predictor = ExoplanetPredictor()
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model_loaded = False

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and make predictions."""
    
    if not model_loaded:
        flash('Model not loaded! Please train the model first.', 'error')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a CSV file.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and validate CSV
        df = pd.read_csv(filepath)
        
        # Check if required features are present
        required_features = predictor.feature_columns
        missing_features = set(required_features) - set(df.columns)
        
        if missing_features:
            flash(f'Missing required columns: {", ".join(missing_features)}', 'error')
            os.remove(filepath)
            return redirect(url_for('index'))
        
        # Make predictions
        results = predictor.predict_from_csv(filepath)
        
        # Save results
        result_filename = f"predictions_{filename}"
        result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        results.to_csv(result_filepath, index=False)
        
        # Get summary statistics
        summary = {
            'total_samples': len(results),
            'predictions': results['prediction'].value_counts().to_dict(),
            'avg_confidence': float(results['confidence'].mean()),
            'min_confidence': float(results['confidence'].min()),
            'max_confidence': float(results['confidence'].max())
        }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('results.html', 
                             summary=summary,
                             results=results.head(50).to_html(classes='table table-striped', index=False),
                             results_data=results.head(50),
                             result_filename=result_filename,
                             total_rows=len(results))
        
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """Download prediction results."""
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        flash('File not found', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)."""
    
    if not model_loaded:
        return {'error': 'Model not loaded'}, 500
    
    try:
        data = request.get_json()
        
        if not data:
            return {'error': 'No data provided'}, 400
        
        # Make prediction
        result = predictor.predict(data)
        
        return result, 200
        
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/model-info')
def model_info():
    """Display model information."""
    
    if not model_loaded:
        flash('Model not loaded', 'error')
        return redirect(url_for('index'))
    
    # Load metadata
    metadata_path = os.path.join(predictor.model_dir, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return render_template('model_info.html', metadata=metadata)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)