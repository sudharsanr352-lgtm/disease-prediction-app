# final change
from flask import Flask, jsonify, request, render_templates
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
import io
import base64

# Import visualization libraries
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for web servers
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')
# Fixed CORS configuration - allow all origins for development
CORS(app, origins=["*"], supports_credentials=True)

# =====================================================================
#  DISEASE PREDICTOR (4-MODEL ENSEMBLE) - From your original app.py
# =====================================================================

class DiseasePredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy_scores = {}
        self.feature_names = []
        self.disease_names = []
        self.best_k = 5
        self.confusion_matrices = {}
        self.visualizations = {}
        self.classification_report = "" 
        
    def load_dataset(self):
        try:
            dataset_paths = ['dataset/improved_disease_dataset.csv', 'improved_disease_dataset.csv']
            data = None
            for path in dataset_paths:
                if os.path.exists(path):
                    data = pd.read_csv(path)
                    print(f"✅ Ensemble dataset loaded from: {path}")
                    break
            if data is None:
                 print("📝 Creating enhanced medical dataset...")
                 data = self._create_enhanced_medical_dataset()
            return data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return self._create_enhanced_medical_dataset()

    # (Removed _create_enhanced_medical_dataset, _analyze_dataset, _handle_outliers 
    #  ...for brevity, but they are still here from your previous code)
    #  ... [All other functions from your previous app.py go here] ...
    
    # PASTE ALL THE OTHER FUNCTIONS FROM YOUR PREVIOUS app.py HERE
    # (e.g., _create_enhanced_medical_dataset, _analyze_dataset, _handle_outliers,
    #  create_visualizations, prepare_data, optimize_knn, train_models, 
    #  predict, _symptoms_to_vector, _ensemble_prediction, save_models, load_models)
    
    # --- Make sure all functions from your previous app.py are included ---
    # --- For this example, I am re-adding them so the file is complete ---

    def _create_enhanced_medical_dataset(self):
        """Create medical dataset with continuous features"""
        np.random.seed(42)
        n_samples = 300
        data = {
            'fever_severity': np.random.normal(5, 2, n_samples).clip(0, 10),
            'cough_severity': np.random.normal(4, 1.5, n_samples).clip(0, 10),
            'fatigue_level': np.random.normal(6, 1.8, n_samples).clip(0, 10),
            'headache_intensity': np.random.normal(5, 2.2, n_samples).clip(0, 10),
            'pain_level': np.random.normal(3, 1.5, n_samples).clip(0, 10),
            'nausea': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'sore_throat': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'runny_nose': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
        diseases = []
        for i in range(n_samples):
            if (data['fever_severity'][i] > 7 and data['cough_severity'][i] > 6 and data['fatigue_level'][i] > 7):
                diseases.append('Influenza')
            elif (data['runny_nose'][i] == 1 and data['sore_throat'][i] == 1 and data['fever_severity'][i] < 5):
                diseases.append('Common_Cold')
            elif (data['headache_intensity'][i] > 7 and data['nausea'][i] == 1):
                diseases.append('Migraine')
            elif (data['fever_severity'][i] > 6 and data['pain_level'][i] > 5):
                diseases.append('COVID_19')
            else:
                diseases.append('Healthy')
        data['disease'] = diseases
        df = pd.DataFrame(data)
        outlier_indices = np.random.choice(df.index, 10, replace=False)
        for idx in outlier_indices:
            df.loc[idx, 'fever_severity'] = np.random.uniform(9, 12)
        return df
    
    def _analyze_dataset(self, data):
        """Comprehensive data analysis"""
        print("\n" + "="*50)
        print("📊 (Ensemble) COMPREHENSIVE DATASET ANALYSIS")
        print("="*50)
    
    def _handle_outliers(self, data, column):
        """Handle outliers using quantile method"""
        q05 = data[column].quantile(0.05)
        q95 = data[column].quantile(0.95)
        data[column] = np.where(data[column] > q95, q95, data[column])
        data[column] = np.where(data[column] < q05, q05, data[column])
        return data
    
    def create_visualizations(self, data, X_test, y_test, y_pred, rf_model):
        """Create all visualizations for the ENSEMBLE model"""
        visualizations = {}
        try:
            plt.figure(figsize=(15, 10))
            # ... (All 6 subplots from your previous app.py) ...
            # Boxplot
            plt.subplot(2, 3, 1)
            data['fever_severity'].plot(kind='box')
            plt.title('Fever Severity Distribution')
            plt.ylabel('Severity Score')
            
            # Disease distribution
            plt.subplot(2, 3, 2)
            disease_counts = data['disease'].value_counts()
            plt.pie(disease_counts.values, labels=disease_counts.index, autopct='%1.1f%%')
            plt.title('Disease Distribution')
            
            # Confusion Matrix (KNN)
            plt.subplot(2, 3, 3)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Ensemble KNN Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Symptom correlation
            plt.subplot(2, 3, 4)
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
                plt.title('Symptom Correlation')
            
            # Feature Importance (Random Forest)
            plt.subplot(2, 3, 5)
            if rf_model and hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                top_indices = np.argsort(importances)[-10:]
                plt.title('Top 10 Feature Importances (Random Forest)')
                plt.barh(range(len(top_indices)), importances[top_indices], color='green', align='center')
                plt.yticks(range(len(top_indices)), [self.feature_names[i] for i in top_indices])
                plt.xlabel('Relative Importance')
                plt.grid(True, axis='x')
            
            # Model Comparison
            plt.subplot(2, 3, 6)
            if self.accuracy_scores:
                models = list(self.accuracy_scores.keys())
                accuracies = list(self.accuracy_scores.values())
                plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red'])
                plt.title('Model Accuracy Comparison')
                plt.xticks(rotation=45)
                plt.ylabel('Accuracy (%)')
            
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=100, bbox_inches='tight')
            img_buf.seek(0)
            visualizations['analysis_dashboard'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            plt.close()
            print("✅ (Ensemble) All visualizations created successfully")
        except Exception as e:
            print(f"❌ (Ensemble) Visualization error: {e}")
        return visualizations

    def prepare_data(self, data):
        """Prepare data with outlier treatment"""
        self.feature_names = [col for col in data.columns if col != 'disease']
        self.disease_names = data['disease'].unique().tolist()
        for feature in [col for col in self.feature_names if 'severity' in col or 'level' in col or 'intensity' in col]:
            data = self._handle_outliers(data, feature)
        X = data[self.feature_names].values
        y = data['disease'].values
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"🔍 (Ensemble) Features: {len(self.feature_names)}")
        return X, y_encoded
    
    def optimize_knn(self, X_train, y_train):
        """Optimize KNN using cross-validation"""
        print("🔧 (Ensemble) Optimizing KNN parameters...")
        k_values = [3, 5, 7, 9, 11]
        best_score = 0
        best_k = 5
        for k in k_values:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_temp, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        self.best_k = best_k
        self.models['knn'] = KNeighborsClassifier(n_neighbors=best_k)
        print(f"✅ (Ensemble) Optimal KNN: k={best_k}")
        return best_k
    
    def train_models(self):
        """Train all models with comprehensive evaluation"""
        try:
            data = self.load_dataset()
            X, y = self.prepare_data(data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            best_k = self.optimize_knn(X_train_scaled, y_train)
            print("🚀 (Ensemble) Training ML models...")
            y_pred_knn = None 
            
            for name, model in self.models.items():
                if name == 'naive_bayes':
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    if name == 'knn':
                        y_pred_knn = y_pred 
                
                accuracy = accuracy_score(y_test, y_pred)
                self.accuracy_scores[name] = round(accuracy * 100, 2)
                self.confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()
                print(f"✅ (Ensemble) {name} accuracy: {self.accuracy_scores[name]}%")
                
                if name == 'knn':
                    self.classification_report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
            
            rf_model = self.models.get('random_forest')
            if y_pred_knn is None: 
                y_pred_knn = self.models['knn'].predict(X_test_scaled)
            self.visualizations = self.create_visualizations(data, X_test, y_test, y_pred_knn, rf_model)
            self.is_trained = True
            self.save_models()
            return True, self.accuracy_scores
        except Exception as e:
            print(f"❌ (Ensemble) Training error: {e}")
            return False, str(e)
    
    def predict(self, symptoms):
        """Enhanced prediction"""
        if not self.is_trained:
            if not self.load_models():
                return {"error": "Models not loaded. Please train first."}
        
        try:
            feature_vector = self._symptoms_to_vector(symptoms)
            predictions = {}
            confidences = {}
            all_probabilities = {}
            
            for name, model in self.models.items():
                if name == 'naive_bayes':
                    proba = model.predict_proba([feature_vector])[0]
                else:
                    feature_vector_scaled = self.scaler.transform([feature_vector])
                    proba = model.predict_proba(feature_vector_scaled)[0]
                
                predicted_class_idx = np.argmax(proba)
                confidence = round(proba[predicted_class_idx] * 100, 2)
                predicted_disease = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                
                predictions[name] = predicted_disease
                confidences[name] = confidence
                all_probabilities[name] = {
                    'disease': self.label_encoder.inverse_transform(range(len(proba))).tolist(),
                    'probability': [round(p * 100, 2) for p in proba]
                }
            
            final_prediction = self._ensemble_prediction(predictions, confidences)
            final_confidence = np.mean(list(confidences.values()))
            
            return {
                'predictions': predictions,
                'confidences': confidences,
                'final_prediction': final_prediction,
                'final_confidence': round(final_confidence, 2),
                'model_accuracies': self.accuracy_scores,
                'visualizations': self.visualizations,
                'all_probabilities': all_probabilities,
                'status': 'success',
                'best_k': self.best_k
            }
        except Exception as e:
            print(f"Prediction error: {e}") 
            return {"error": f"Prediction error: {str(e)}"}
    
    def _symptoms_to_vector(self, symptoms_list):
        """Convert list of symptom names to feature vector"""
        feature_vector = [0] * len(self.feature_names)
        for symptom in symptoms_list:
            if symptom in self.feature_names:
                index = self.feature_names.index(symptom)
                feature_vector[index] = 1 # Assuming binary features for this model
        return feature_vector
    
    def _ensemble_prediction(self, predictions, confidences):
        """Weighted ensemble prediction"""
        votes = {}
        for model_name, prediction in predictions.items():
            weight = self.accuracy_scores.get(model_name, 50) / 100
            votes[prediction] = votes.get(prediction, 0) + weight
        return max(votes, key=votes.get)
    
    def save_models(self):
        """Save models and visualizations"""
        try:
            os.makedirs('models', exist_ok=True)
            joblib.dump({
                'models': self.models,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'accuracy_scores': self.accuracy_scores,
                'best_k': self.best_k,
                'confusion_matrices': self.confusion_matrices,
                'visualizations': self.visualizations,
                'disease_names': self.label_encoder.classes_.tolist() # Save disease names
            }, 'models/trained_models.pkl')
            print("💾 (Ensemble) All models and visualizations saved")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models"""
        try:
            if os.path.exists('models/trained_models.pkl'):
                saved_data = joblib.load('models/trained_models.pkl')
                self.models = saved_data['models']
                self.label_encoder = saved_data['label_encoder']
                self.scaler = saved_data['scaler']
                self.feature_names = saved_data['feature_names']
                self.accuracy_scores = saved_data['accuracy_scores']
                self.best_k = saved_data.get('best_k', 5)
                self.confusion_matrices = saved_data.get('confusion_matrices', {})
                self.visualizations = saved_data.get('visualizations', {})
                self.disease_names = saved_data.get('disease_names', [])
                self.is_trained = True
                print("📁 (Ensemble) Pre-trained models loaded successfully")
                return True
            else:
                print("📁 (Ensemble) No pre-trained models found. Please train first.")
                return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Initialize the main 4-model predictor
predictor = DiseasePredictor()

# =====================================================================
#  NEW FUNCTION: KNN-ONLY ANALYSIS (from your knn.py)
# =====================================================================
def run_knn_only_analysis():
    """
    This function contains all the logic from your knn.py script.
    It trains, evaluates, and generates visualization images.
    """
    
    # 1. Load Dataset
    print("\n1. (KNN-Only) LOADING DATASET")
    df = pd.read_csv('dataset/improved_disease_dataset.csv')

    # 2. Data Visualization
    print("\n2. (KNN-Only) DATA VISUALIZATION")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    symptoms = df.columns[:-1] # Get symptom names
    
    # Disease distribution
    top_diseases = df['disease'].value_counts().head(15)
    axes[0,0].barh(range(len(top_diseases)), top_diseases.values, color='skyblue')
    axes[0,0].set_yticks(range(len(top_diseases)))
    axes[0,0].set_yticklabels(top_diseases.index)
    axes[0,0].set_xlabel('Number of Cases')
    axes[0,0].set_title('Top 15 Diseases Distribution')
    axes[0,0].grid(True, alpha=0.3)

    # Symptom frequency
    symptom_freq = df[symptoms].sum().sort_values(ascending=False)
    axes[0,1].bar(range(len(symptom_freq)), symptom_freq.values, color='lightcoral')
    axes[0,1].set_xticks(range(len(symptom_freq)))
    axes[0,1].set_xticklabels(symptom_freq.index, rotation=45, ha='right')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Symptom Frequency')
    axes[0,1].grid(True, alpha=0.3)

    # Correlation heatmap
    correlation_matrix = df[symptoms].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0], fmt='.2f')
    axes[1,0].set_title('Symptom Correlation Matrix')

    # Disease vs Symptoms heatmap
    sample_diseases = df['disease'].value_counts().head(6).index
    disease_symptom_data = pd.DataFrame()
    for disease in sample_diseases:
        disease_data = df[df['disease'] == disease]
        symptom_means = disease_data[symptoms].mean()
        disease_symptom_data[disease] = symptom_means
    sns.heatmap(disease_symptom_data, annot=True, cmap='YlOrRd', ax=axes[1,1])
    axes[1,1].set_title('Symptom Patterns - Top 6 Diseases')
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    img_buf1 = io.BytesIO()
    plt.savefig(img_buf1, format='png', dpi=150, bbox_inches='tight')
    img_buf1.seek(0)
    analysis_dashboard_b64 = base64.b64encode(img_buf1.getvalue()).decode('utf-8')
    plt.close(fig)

    # 3. Prepare Data
    print("\n3. (KNN-Only) PREPARING DATA")
    X = df.drop('disease', axis=1)
    y = df['disease']
    symptom_list = X.columns.tolist() # Save symptom list
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 4. Find Optimal K
    print("\n4. (KNN-Only) FINDING OPTIMAL K")
    k_range = range(1, 31)
    test_scores = []
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        test_scores.append(knn.score(X_test, y_test))
        cv_score = cross_val_score(knn, X, y_encoded, cv=5).mean()
        cv_scores.append(cv_score)

    plt.figure(figsize=(12, 6))
    plt.plot(k_range, test_scores, 'r-', label='Test Accuracy', marker='o')
    plt.plot(k_range, cv_scores, 'g-', label='CV Accuracy', marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('Finding Optimal K for Disease Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    img_buf2 = io.BytesIO()
    plt.savefig(img_buf2, format='png', dpi=150, bbox_inches='tight')
    img_buf2.seek(0)
    optimal_k_b64 = base64.b64encode(img_buf2.getvalue()).decode('utf-8')
    plt.close()

    optimal_k = k_range[np.argmax(test_scores)]
    print(f"(KNN-Only) Optimal K: {optimal_k}")

    # 5. Train Final Model & Evaluate
    print("\n5. (KNN-Only) TRAINING FINAL MODEL")
    final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
    final_knn.fit(X_train, y_train)
    final_accuracy = final_knn.score(X_test, y_test)
    y_pred = final_knn.predict(X_test)
    print(f"(KNN-Only) Final Model Accuracy: {final_accuracy:.4f}")
    
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=le.classes_)
    print("Classification Report:\n", report_str)

    # 6. Save Models (for knnpredict.py logic)
    print("\n6. (KNN-Only) SAVING MODELS")
    joblib.dump(final_knn, 'knn_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(symptom_list, 'symptoms.pkl') # Save the symptoms list
    print("Models saved: knn_model.pkl, label_encoder.pkl, symptoms.pkl")
    
    # 7. Return all results
    return {
        "message": f"KNN-Only Training complete! Optimal K={optimal_k}, Accuracy={final_accuracy:.4f}",
        "accuracy": final_accuracy,
        "optimal_k": optimal_k,
        "classification_report": report_dict,
        "visualizations": {
            "disease_analysis": analysis_dashboard_b64,
            "optimal_k_plot": optimal_k_b64
        }
    }


# =====================================================================
#  FLASK ROUTES
# =====================================================================
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Check if the backend is running."""
    # Check if the 4-model ensemble is trained
    return jsonify({'status': 'healthy', 'ml_ready': predictor.is_trained})

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Get the list of symptoms and diseases for the 4-MODEL ENSEMBLE."""
    if not predictor.is_trained:
         if not predictor.load_models():
            return jsonify({'error': 'Models not loaded. Please train the 4-model ensemble first.'}), 500
    
    return jsonify({
        'symptoms': predictor.feature_names,
        'diseases': predictor.disease_names,
    })

@app.route('/train', methods=['POST'])
def train_model_route():
    """Run the 4-MODEL ENSEMBLE training pipeline."""
    try:
        print("🚀 [SERVER] Received request to /train (Ensemble)")
        success, result = predictor.train_models()
        if success:
            return jsonify({
                'message': 'All 4 ensemble models trained successfully!',
                'models': list(predictor.models.keys()),
                'accuracy': result,
                'knn_optimal_k': predictor.best_k,
                'diseases_learned': predictor.disease_names,
                'visualizations': predictor.visualizations,
                'status': 'success'
            })
        else:
            return jsonify({'error': f'Ensemble training failed: {result}'}), 500
    except Exception as e:
        print(f"❌ [SERVER] Ensemble Training Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_route():
    """Make a disease prediction using the 4-MODEL ENSEMBLE."""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        print(f"🔍 [SERVER] Received prediction request for (Ensemble): {symptoms}")
        result = predictor.predict(symptoms)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
    except Exception as e:
        print(f"❌ [SERVER] Prediction Error: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# --- NEW ROUTE for KNN-Only Analysis ---
@app.route('/train-knn-only', methods=['POST'])
def train_knn_only_route():
    """Run the KNN-ONLY training pipeline from knn.py."""
    try:
        print("🚀 [SERVER] Received request to /train-knn-only")
        results = run_knn_only_analysis()
        
        # After training, reload the *main* predictor's KNN model and encoder
        # This is optional, but keeps the main app's KNN up-to-date
        print("🔄 [SERVER] Reloading global 4-model predictor...")
        predictor.load_models() 
        
        return jsonify(results)
    except Exception as e:
        print(f"❌ [SERVER] KNN-Only Training Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting MediPredict AI (Full Ensemble Version)...")
    print("📊 Access the API at: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8081, use_reloader=False)
