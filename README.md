# Resume Classification App
##  Live Demo
 [Click here to try the app on Streamlit Cloud](https://resume-classification-app-hcpe3kghhssqrykpycsumx.streamlit.app/)

## Project Overview
This project is a **Streamlit-based web application** that classifies resumes into **four job roles**:
- React Developer
- SQL Developer
- PeopleSoft Consultant
- Workday Consultant
It uses **Natural Language Processing (NLP)** and a **Machine Learning pipeline** to predict the role and extract useful details like **email, experience, and skills** from resumes.

##  Features
-  Upload **PDF/DOCX** resumes
-  Extract text using **PyPDF2** and **python-docx**
-  Text preprocessing with **NLTK** (stopwords removal, stemming)
-  ML pipeline: **TF-IDF + LinearSVC**
-  Predict job role and display:
  - Email
  - Experience
  - Skills
-  Deployed on **Streamlit Cloud**

##  Tech Stack
- **Language:** Python 3.9+
- **Framework:** Streamlit
- **ML:** scikit-learn
- **NLP:** NLTK
- **Others:** joblib, python-docx, PyPDF2

##  Project Structure
├── resume_app.py # Streamlit app
├── Resume_Classification_Project.ipynb # Model training notebook
├── the_resume_classifier.pkl # Saved ML model
├── requirements.txt # Dependencies
├── resume_cleandataset_labeled.xlsx # Labeled dataset

## How to Run Locally
# Clone the repository
git clone https://github.com/farah-web/resume-classification-app.git
cd resume-classification-app

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run resume_app.py

