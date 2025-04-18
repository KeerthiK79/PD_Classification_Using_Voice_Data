

# Early-Stage Parkinson’s Disease Prediction Using Voice Recording Data  

##  Overview  
This project develops a **machine learning (ML) model** to predict early-stage **Parkinson’s Disease (PD)** using voice recordings. By analyzing speech abnormalities, we aim to provide a **non-invasive, cost-effective diagnostic tool** for early detection.  

###  Key Features  
- **Class Imbalance Handling**: Applied **SMOTE** to balance PD vs. healthy voice samples.  
- **Feature Selection**: Used **Recursive Feature Elimination (RFE)** to identify optimal features.  
- **Model Comparison**: Evaluated **Random Forest, SVM, and KNN** with/without RFE.  
- **High Accuracy**: Achieved **94.87% accuracy** with Random Forest (outperforming literature benchmarks).  

---  

## Technologies Used  
- **Python** (NumPy, Pandas, Scikit-learn)  
- **ML Models**: Random Forest, SVM, KNN  
- **Techniques**: SMOTE, RFE, GridSearchCV  
- **Dataset**: [UCI Parkinson’s Voice Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)  

---  

##  Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Anannabiswas/PD_Classification_Using_Voice_Data.git
   cd PD_Classification_Using_Voice_Data
   ```  
2. Install dependencies:  
   ```bash  
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib  
   ```  

---  

##  Usage  
### Option 1: Run as Python Script
```bash
python pd_classification_voice_data.py
Outputs Generated:

Model evaluation reports (accuracy/precision/recall) printed to console

Confusion matrices for all models (saved as PNG)

ROC curves comparing all classifiers

Feature importance plots from RFE

Example terminal output:

Random Forest Results:
Accuracy: 0.9487 | Precision: 0.95 | Recall: 0.97
Confusion Matrix:
[[ 7  1]
 [ 0 30]]
Option 2: Interactive Jupyter Notebook
bash
jupyter notebook PD_Classification_Voice_Data.ipynb
Step-by-Step Execution:
1. Data Preparation

python
# Load and inspect data
df = pd.read_csv('data/parkinsons.data')
display(df.head())

# Split dataset (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
2. Class Balancing (SMOTE)

python
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
sns.countplot(x=y_balanced)  # Visualize balanced classes
3. Model Training

python
# Initialize and train models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_balanced, y_balanced)
4. Evaluation

python
# Generate evaluation metrics
for name, model in models.items():
    evaluate_model(model, X_test, y_test, name)  # Prints metrics + plots
Note: Ensure all dependencies are installed using:

bash
pip install -r requirements.txt
---  

## Results  
| Model          | Accuracy (with RFE) |  
|----------------|---------------------|  
| **Random Forest** | 94.87%              |  
| SVM            | 90.22%              |  
| KNN            | 92.30%              |  

---  

## Project Deliverables  
1. **Presentation Slides**: [AI In Healthcare-Final Project.pptx](Presentation_Slide/Final_Project_Grp15.pptx)  
2. **YouTube Presentation**: [Link to Video](https://youtube.com/your-video-link)  
3. **Dataset**: [UCI Repository](https://archive.ics.uci.edu/dataset/174/parkinsons) or [Zipped File](Data/parkinsons.zip)  
4. **Code**: [Code Files](Code/).  

---  

## Team Members  
- **Ananna Biswas** (PhD Candidate, MTU)  
  - Email: [anannab@mtu.edu](mailto:anannab@mtu.edu)  
  - Website: [https://anannabiswas.github.io/](https://anannabiswas.github.io/)  
- **Keerthi Kesavan** (MS Health Informatics, MTU)  
  - Email: [kkesavan@mtu.edu](mailto:kkesavan@mtu.edu)  

---  

##  Future Work  
- Implement **deep learning (CNNs/RNNs)** for raw audio feature extraction.  
- Test on larger datasets for generalizability.  

---  
 

**GitHub Links**:  
- [Ananna’s Repository](https://github.com/Anannabiswas/PD_Classification_Using_Voice_Data/tree/main)  
- [Keerthi’s Repository](https://github.com/keerthikesavan/parkinsons-voice-prediction)  

