class DiseasePredictionApp {
    constructor() {
    this.baseUrl = '';
    this.selectedSymptoms = new Set();
    this.isBackendConnected = false;
    this.isMLTrained = false; // This now tracks the 4-model ensemble
    this.initializeEventListeners();
    this.updateStatus('Checking system connection...', 'loading');
    
    // Check connection on startup
    setTimeout(() => {
        this.testBackendConnection();
    }, 1000); // Check after 1 sec
}

    initializeEventListeners() {
        // 4-Model Ensemble buttons
        document.getElementById('loadSymptomsBtn').addEventListener('click', () => this.loadSymptoms());
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModel());
        document.getElementById('predictBtn').addEventListener('click', () => this.predictDisease());
        
        // New KNN-Only button
        document.getElementById('knnAnalysisBtn').addEventListener('click', () => this.runKnnAnalysis());
        
        // Shared buttons
        document.getElementById('clearSymptomsBtn').addEventListener('click', () => this.clearSymptoms());
        document.getElementById('symptomSearch').addEventListener('input', (e) => this.filterSymptoms(e.target.value));
        document.getElementById('retryConnection').addEventListener('click', () => this.testBackendConnection());
        
        // --- THE ERROR WAS HERE ---
        // The line trying to access 'debugBtn' has been removed.
    }

    async testBackendConnection() {
        try {
            this.updateStatus('Testing backend connection...', 'loading');
            console.log('🔌 Testing connection to:', this.baseUrl);
            
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: {'Accept': 'application/json', 'Content-Type': 'application/json'},
                mode: 'cors'
            });

            if (response.ok) {
                const data = await response.json();
                this.isBackendConnected = true;
                this.isMLTrained = data.ml_ready || false;
                
                if (this.isMLTrained) {
                    this.updateStatus('✅ System Ready - Click "Load Symptoms"', 'ready');
                    await this.loadModelInfo(); // Load counts if already trained
                } else {
                    this.updateStatus('⚠️ Backend Connected - Please Train Ensemble', 'ready');
                }
                
                document.getElementById('retryConnection').style.display = 'none';
                this.showNotification('Backend connected successfully!', 'success');
                
            } else {
                throw new Error(`HTTP ${response.status} - ${response.statusText}`);
            }
        } catch (error) {
            console.error('Backend connection failed:', error);
            this.isBackendConnected = false;
            this.updateStatus('❌ Backend Not Available - Start app.py!', 'error');
            this.showNotification(`Cannot connect to backend. Make sure app.py is running.`, 'error');
            document.getElementById('retryConnection').style.display = 'block';
        }
    }

    async loadModelInfo() {
        // This function now just updates the stats from the /symptoms endpoint
        try {
            const response = await fetch(`${this.baseUrl}/symptoms`);
            if (response.ok) {
                const data = await response.json();
                document.getElementById('diseaseCount').textContent = data.diseases?.length || 0;
                document.getElementById('symptomCount').textContent = data.symptoms?.length || 0;
                
                document.getElementById('trainBtn').innerHTML = '<i class="fas fa-sync-alt"></i> Retrain Ensemble';
                document.getElementById('trainBtn').classList.add('btn-success');
            }
        } catch (error) {
            console.error('Failed to load model info:', error);
        }
    }

    async loadSymptoms() {
        if (!this.isBackendConnected) {
            this.showNotification('Backend not connected. Please start the backend server first.', 'error');
            return;
        }
        if (!this.isMLTrained) {
            this.showNotification('Please train the Ensemble model first!', 'warning');
            return;
        }

        try {
            this.updateStatus('Loading symptoms...', 'loading');
            
            const response = await fetch(`${this.baseUrl}/symptoms`);
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            this.displaySymptoms(data.symptoms);
            this.updateStatus('Symptoms loaded successfully', 'ready');
            this.showNotification(`Loaded ${data.symptoms?.length || 0} symptoms`, 'success');
            
            // Update counts
            document.getElementById('symptomCount').textContent = data.symptoms?.length || 0;
            document.getElementById('diseaseCount').textContent = data.diseases?.length || 0;
            
        } catch (error) {
            console.error('Load symptoms error:', error);
            this.updateStatus(`Error: ${error.message}`, 'error');
            this.showNotification(`Failed to load symptoms: ${error.message}`, 'error');
        }
    }

    displaySymptoms(symptoms) {
        const container = document.getElementById('symptomsContainer');
        
        if (!symptoms || symptoms.length === 0) {
            container.innerHTML = '<div class="placeholder">No symptoms available</div>';
            return;
        }

        container.innerHTML = symptoms.map(symptom => `
            <div class="symptom-checkbox">
                <input type="checkbox" id="symptom-${symptom}" value="${symptom}">
                <label for="symptom-${symptom}">
                    <i class="fas fa-plus-circle"></i>
                    ${this.formatSymptomName(symptom)}
                </label>
            </div>
        `).join('');

        container.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const symptom = e.target.value;
                if (e.target.checked) {
                    this.selectedSymptoms.add(symptom);
                } else {
                    this.selectedSymptoms.delete(symptom);
                }
                this.updateSelectedSymptomsDisplay();
            });
        });
    }

    updateSelectedSymptomsDisplay() {
        const container = document.getElementById('selectedSymptoms');
        const countElement = document.getElementById('selectedCount');
        
        countElement.textContent = this.selectedSymptoms.size;
        
        if (this.selectedSymptoms.size === 0) {
            container.innerHTML = '<div class="placeholder">No symptoms selected</div>';
            return;
        }

        container.innerHTML = Array.from(this.selectedSymptoms).map(symptom => `
            <span class="selected-symptom">
                ${this.formatSymptomName(symptom)}
                <i class="fas fa-times" onclick="app.removeSymptom('${symptom}')"></i>
            </span>
        `).join('');
    }

    removeSymptom(symptom) {
        this.selectedSymptoms.delete(symptom);
        this.updateSelectedSymptomsDisplay();
        
        const checkbox = document.querySelector(`input[value="${symptom}"]`);
        if (checkbox) checkbox.checked = false;
    }

    clearSymptoms() {
        this.selectedSymptoms.clear();
        this.updateSelectedSymptomsDisplay();
        
        document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        
        document.getElementById('symptomSearch').value = '';
        this.filterSymptoms('');
        
        this.updateStatus('Symptoms cleared', 'ready');
        this.showNotification('All symptoms cleared', 'info');
    }

    filterSymptoms(searchTerm) {
        const symptoms = document.querySelectorAll('.symptom-checkbox');
        const term = searchTerm.toLowerCase();
        
        symptoms.forEach(symptom => {
            const label = symptom.querySelector('label').textContent.toLowerCase();
            symptom.style.display = label.includes(term) ? 'flex' : 'none';
        });
    }

    // --- This function trains the 4-MODEL ENSEMBLE ---
    async trainModel() {
        if (!this.isBackendConnected) {
            this.showNotification('Backend not connected', 'error');
            return;
        }

        try {
            this.updateStatus('🚀 Training Ensemble models... This may take a minute.', 'loading');
            document.getElementById('trainBtn').disabled = true;
            document.getElementById('trainBtn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
            this.showNotification('Starting 4-model ensemble training... Please wait.', 'info');
            
            const response = await fetch(`${this.baseUrl}/train`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Training failed');

            this.updateStatus('✅ Ensemble Training complete! Click "Load Symptoms".', 'ready');
            this.showNotification(data.message || 'Ensemble Models Trained!', 'success');
            
            // --- Show Ensemble visualization ---
            const vizContainer = document.getElementById('visualizationContainer');
            const vizImg1 = document.getElementById('analysisDashboardImg');
            
            if (data.visualizations && data.visualizations.analysis_dashboard) {
                vizImg1.src = 'data:image/png;base64,' + data.visualizations.analysis_dashboard;
                vizContainer.style.display = 'block';
                this.showNotification('Ensemble dashboard updated!', 'info');
            } else {
                vizContainer.style.display = 'none';
            }
            // --- END ---

            this.isMLTrained = true;
            document.getElementById('trainBtn').innerHTML = '<i class="fas fa-sync-alt"></i> Retrain Ensemble';
            document.getElementById('trainBtn').classList.add('btn-success');
            document.getElementById('trainBtn').disabled = false;
            
            await this.loadModelInfo(); // Reload stats
            
        } catch (error) {
            this.updateStatus(`❌ Ensemble Training error: ${error.message}`, 'error');
            this.showNotification(`Training failed: ${error.message}`, 'error');
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('trainBtn').innerHTML = '<i class="fas fa-brain"></i> Train Ensemble';
        }
    }

    // --- This function predicts using the 4-MODEL ENSEMBLE ---
    async predictDisease() {
        if (!this.isBackendConnected) {
            this.showNotification('Backend not connected', 'error');
            return;
        }

        if (!this.isMLTrained) {
            this.showNotification('Please train the Ensemble model first', 'warning');
            return;
        }

        if (this.selectedSymptoms.size === 0) {
            this.showNotification('Please select at least one symptom', 'warning');
            return;
        }

        try {
            this.updateStatus('🔍 Analyzing symptoms (Ensemble)...', 'loading');
            document.getElementById('predictBtn').disabled = true;
            document.getElementById('predictBtn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            
            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symptoms: Array.from(this.selectedSymptoms)
                })
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Prediction failed');

            this.displayPredictionResults(data);
            this.updateStatus('✅ Ensemble Prediction completed', 'ready');
            this.showNotification('Ensemble Prediction completed!', 'success');

        } catch (error) {
            this.updateStatus(`❌ Prediction error: ${error.message}`, 'error');
            this.showNotification(`Prediction failed: ${error.message}`, 'error');
        } finally {
            document.getElementById('predictBtn').disabled = false;
            document.getElementById('predictBtn').innerHTML = '<i class="fas fa-diagnoses"></i> Predict Disease';
        }
    }
    
    // --- This function displays the 4-MODEL ENSEMBLE results ---
    displayPredictionResults(data) {
        const resultsContainer = document.getElementById('results');
        
        // --- Build the 4-model list ---
        let modelsHTML = '';
        if (data.predictions) {
            for (const [model, prediction] of Object.entries(data.predictions)) {
                modelsHTML += `
                    <div class="model-prediction">
                        <span class="model-name">
                            <i class="${this.getModelIcon(model)}"></i>
                            ${this.formatModelName(model)}
                        </span>
                        <span class="prediction-value">${prediction} (${data.confidences[model]}%)</span>
                    </div>
                `;
            }
        }

        // --- Get the list of analyzed symptoms ---
        const symptomsString = Array.from(this.selectedSymptoms)
                                    .map(s => this.formatSymptomName(s))
                                    .join(', ');

        // --- Build the Final HTML Layout ---
        let resultsHTML = `
            <div class="prediction-results" style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px;">
                
                <div class="left-column">
                    
                    <div class="model-results" style="display: flex; flex-direction: column; gap: 15px; margin-bottom: 25px;">
                        ${modelsHTML}
                    </div>

                    <div class="analysis-details" style="padding: 20px; background: var(--light); border-radius: 15px;">
                        <h4 style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px; color: var(--primary);">
                            <i class="fas fa-info-circle"></i> Analysis Details
                        </h4>
                        <p class="symptoms-list" style="color: var(--gray); margin-bottom: 15px;">
                            <strong>Symptoms analyzed:</strong> ${symptomsString}
                        </p>
                        <div class="disclaimer" style="font-size: 0.85rem; color: var(--gray); padding: 15px; background: white; border-radius: 10px; border-left: 4px solid var(--warning);">
                            <i class="fas fa-exclamation-triangle" style="color: var(--warning); margin-right: 8px;"></i>
                            AI-assisted prediction. Consult healthcare professionals for medical diagnosis.
                        </div>
                    </div>
                </div>

                <div class="right-column">
                    <div class="final-prediction" style="background: var(--bg-gradient); color: white; padding: 30px; border-radius: 20px; text-align: center; box-shadow: var(--card-shadow); height: 100%; display: flex; flex-direction: column; justify-content: center;">
                        <h3 style="font-size: 1.5rem; margin-bottom: 15px; display: flex; align-items: center; justify-content: center; gap: 10px;">
                            <i class="fas fa-user-md"></i> Final Diagnosis
                        </h3>
                        <div class="diagnosis" style="font-size: 2.2rem; font-weight: 700; margin: 20px 0; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">
                            ${data.final_prediction}
                        </div>
                        <span class="confidence-badge" style="background: rgba(255, 255, 255, 0.2); padding: 12px 25px; border-radius: 50px; font-size: 1.1rem; display: inline-flex; align-items: center; gap: 8px; backdrop-filter: blur(10px); margin: 0 auto;">
                            <i class="fas fa-shield-alt"></i>
                            Confidence: ${data.final_confidence}%
                        </span>
                    </div>
                </div>

            </div>
        `;
        
        resultsContainer.innerHTML = resultsHTML;

        // --- Show visualization ---
        const vizContainer = document.getElementById('visualizationContainer');
        const vizImg = document.getElementById('analysisDashboardImg');
        
        if (data.visualizations && data.visualizations.analysis_dashboard) {
            vizImg.src = 'data:image/png;base64,' + data.visualizations.analysis_dashboard;
            vizContainer.style.display = 'block';
        } else {
            vizContainer.style.display = 'none';
        }
        
        // Hide the KNN-Only results
        document.getElementById('knnResultsSection').style.display = 'none';
    }
    
    // --- NEW FUNCTION: Runs the KNN-Only Analysis ---
    async runKnnAnalysis() {
        if (!this.isBackendConnected) {
            this.showNotification('Backend not connected', 'error');
            return;
        }

        try {
            this.updateStatus('🚀 Running KNN-Only Analysis... This may take a minute.', 'loading');
            document.getElementById('knnAnalysisBtn').disabled = true;
            document.getElementById('knnAnalysisBtn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            this.showNotification('Starting KNN-Only analysis... Please wait.', 'info');
            
            const response = await fetch(`${this.baseUrl}/train-knn-only`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'KNN analysis failed');

            this.updateStatus('✅ KNN Analysis Complete!', 'ready');
            this.showNotification(data.message || 'KNN Analysis Complete!', 'success');
            
            // --- Show BOTH KNN visualizations ---
            const knnVizContainer = document.getElementById('knnResultsSection');
            const vizImg1 = document.getElementById('knnAnalysisDashboardImg');
            const vizImg2 = document.getElementById('knnOptimalKImg');
            
            if (data.visualizations) {
                vizImg1.src = 'data:image/png;base64,' + data.visualizations.disease_analysis;
                vizImg2.src = 'data:image/png;base64,' + data.visualizations.optimal_k_plot;
                knnVizContainer.style.display = 'block';
                this.showNotification('KNN dashboards updated!', 'info');
            } else {
                knnVizContainer.style.display = 'none';
            }
            
            // Hide the main ensemble results
            document.getElementById('results').innerHTML = '<div class="placeholder">Select symptoms and click "Predict Disease" to see results</div>';
            document.getElementById('visualizationContainer').style.display = 'none';
            
            // Also update main app status to "trained"
            this.isMLTrained = true;
            await this.loadModelInfo();

        } catch (error) {
            this.updateStatus(`❌ KNN Analysis error: ${error.message}`, 'error');
            this.showNotification(`KNN Analysis failed: ${error.message}`, 'error');
        } finally {
            document.getElementById('knnAnalysisBtn').disabled = false;
            document.getElementById('knnAnalysisBtn').innerHTML = '<i class="fas fa-search-plus"></i> Run KNN Analysis';
        }
    }


    formatModelName(model) {
        const names = {
            'random_forest': 'Random Forest',
            'naive_bayes': 'Naive Bayes', 
            'svm': 'SVM',
            'knn': 'K-Nearest Neighbors'
        };
        return names[model] || model;
    }

    formatSymptomName(symptom) {
        // Simple formatter, replace with a more robust one if needed
        return symptom.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    getModelIcon(model) {
        const icons = {
            'random_forest': 'fas fa-robot',
            'naive_bayes': 'fas fa-calculator',
            'svm': 'fas fa-vector-square',
            'knn': 'fas fa-users'
        };
        return icons[model] || 'fas fa-brain'; // Default icon
    }
    
    async debugBackend() {
        this.testBackendConnection(); // Just re-run the health check
    }

    updateStatus(message, type = 'ready') {
        const statusElement = document.getElementById('statusText');
        const dotElement = document.querySelector('.status-dot');
        
        statusElement.textContent = message;
        
        dotElement.className = 'status-dot'; // Reset classes
        if (type === 'ready') dotElement.classList.add('ready');
        else if (type === 'loading') dotElement.classList.add('loading');
        else dotElement.classList.add('error');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6'};
            color: white;
            border-radius: 8px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        `;
        notification.innerHTML = `
            <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'}"></i>
            ${message}
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DiseasePredictionApp();
});