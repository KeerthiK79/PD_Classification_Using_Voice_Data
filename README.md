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

---  
## How to Run  
### Run the Jupyter Notebook 
1. Clone the repository:  
   ```bash
   git clone https://github.com/Anannabiswas/PD_Classification_Using_Voice_Data.git
   ```
2. Open the notebook:  
   ```bash
   jupyter notebook PD_Classification_Using_Voice_Data.ipynb
   ```
3. Run the cells sequentially.  

### Run the Python Script 
```bash
python PD_Classification_Using_Voice_Data.py
```

## Dependencies  
Install required libraries:  
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib  
```


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

