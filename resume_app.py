import streamlit as st
import joblib
import nltk
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os
# Download only if not present
nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Text preprocessing function   
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
def transform_text(text):
    # Handles both single text and lists
    if isinstance(text, (list, tuple)):
        return [transform_text(t) for t in text]
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = re.findall(r'\b\w+\b', text)
    filtered = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered)

# Load the saved model (Pipeline: Preprocessing → TF-IDF → LinearSVC)
model = joblib.load("the_resume_classifier.pkl")

# Role mapping
role_mapping = {
    0: "Peoplesoft Consultant",
    1: "React Developer",
    2: "SQL Developer",
    3: "Workday Consultant"
}

#  Extract name, experience, and skills function
def extract_details(text):
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "Email not found"

    # Extract experience (supports patterns like 4 years, 8+ years, 2.5 yrs)
    exp_matches = re.findall(r'(\d+(?:\.\d+)?\+?)\s*(?:years?|yrs?)', text, re.IGNORECASE)
    experience = f"{exp_matches[0]} years" if exp_matches else "Not found"

    # Extract skills from sections like SKILLS, TECHNICAL SKILLS, TOOLS
    skills = "Not found"
    # Capture text from "skills" to next heading (Experience, Education, Summary)
    skill_section = re.search(r'(skills|technical skills|skills & other)(.*?)(experience|education|summary|\Z)', 
                               text, re.IGNORECASE | re.DOTALL)
    if skill_section:
        raw_block = skill_section.group(2)
        # Split on colon, commas, newlines, tabs
        tokens = re.split(r'[:,\n\t]+', raw_block)
        # Keep tokens that look like technologies (exclude long phrases > 3 words)
        clean_tokens = [t.strip() for t in tokens if t.strip() and len(t.split()) <= 3]
        skills = ", ".join(sorted(set(clean_tokens)))

    return email, experience, skills


# Function to extract text from uploaded resume
def extract_text_from_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    
    if ext not in ['.pdf', '.docx']:
        st.error("Unsupported file format! Please upload PDF or DOCX files.")
        return None

    try:
        if ext == '.pdf':
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif ext == '.docx':
            doc = Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Streamlit UI
st.title("📄 Resume Classification App")
st.write("Upload your resume to predict its category.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    st.info(f"Processing file: {uploaded_file.name}")
    resume_text = extract_text_from_file(uploaded_file)

    if resume_text:
        # Display extracted text preview
        st.subheader("Extracted Resume Text (Preview)")
        # st.text_area("", resume_text[:1000], height=200)
        show_full = st.checkbox("Show full resume text")
        if show_full:
            st.text_area("Full Resume Text", resume_text, height=400)
        else:
            st.text_area("Resume Preview", resume_text[:1000], height=200)

        # Predict category (raw text → pipeline handles preprocessing)
        prediction = model.predict([resume_text])[0]
        role = role_mapping.get(prediction, "Unknown Role")

        st.success(f"**Predicted Role:** {role}")

        # Extract additional details for display
        email, experience, skills = extract_details(resume_text)
        st.write(f"**Email:** {email}")
        st.write(f"**Experience:** {experience}")
        st.write(f"**Skills:** {skills}")
