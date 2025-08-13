import streamlit as st
import joblib
import nltk
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import plotly.graph_objects as go
import io

nltk.download('stopwords', quiet=True)

# Text preprocessing
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def transform_text(text):
    if isinstance(text, list):
        return [transform_text(t) for t in text]
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = re.findall(r'\b\w+\b', text)
    filtered = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered)

# Load model
model = joblib.load("the_resume_classifier.pkl")

# Role mapping
role_mapping = {
    0: "Peoplesoft Consultant",
    1: "React Developer",
    2: "SQL Developer",
    3: "Workday Consultant"
}

# Pre-defined role skills (for gauge chart)
role_skills = {
    "Peoplesoft Consultant": ["peoplesoft", "oracle", "sql", "erp", "hrms"],
    "React Developer": ["react", "javascript", "redux", "css", "html", "typescript"],
    "SQL Developer": ["sql", "plsql", "database", "oracle", "mysql"],
    "Workday Consultant": ["workday", "hcm", "hr", "payroll", "integration"]
}

def extract_details(text):
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "Email not found"
    exp_matches = re.findall(r'(\d+(?:\.\d+)?\+?)\s*(?:years?|yrs?)', text, re.IGNORECASE)
    experience = f"{exp_matches[0]} years" if exp_matches else "Not found"
    skill_section = re.search(r'(skills|technical skills|core skills|key skills|technical summary|skills & other)(.*?)(experience|education|summary|\Z)', 
                               text, re.IGNORECASE | re.DOTALL)
    skills = "Not found"
    if skill_section:
        raw_block = skill_section.group(2)
        tokens = re.split(r'[:,\n\t]+', raw_block)
        clean_tokens = [t.strip() for t in tokens if t.strip() and len(t.split()) <= 3]
        skills = ", ".join(sorted(set(clean_tokens)))
    return email, experience, skills

def extract_text_from_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if ext not in ['.pdf', '.docx']:
        st.error("Unsupported file format! Please upload PDF or DOCX.")
        return None
    try:
        text = ""
        if ext == '.pdf':
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif ext == '.docx':
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def generate_excel(name, email, experience, skills, role):
    df = pd.DataFrame([{
        'Name': name,
        'Email': email,
        'Experience': experience,
        'Skills': skills,
        'Predicted Role': role
    }])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

# Sidebar
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("Resume Classifier")
st.sidebar.markdown(
    """
    <style>
    .sidebar-text {
        font-size: 8px;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True 
)
st.sidebar.markdown(
    '<p class="sidebar-text">Upload a resume to predict the most relevant job role.<br><br>' 
    'Supports: React, SQL, Peoplesoft, Workday</p>',
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3 = st.tabs(["📄 Upload Resume", "📊 Visualizations", "ℹ️ About"])

uploaded_file = None
resume_text = None
role = None
skills = None

# --- Tab 1: Upload Resume ---
with tab1:
    st.header("📄 Resume Classification App")
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    if uploaded_file is not None:
        st.info(f"Processing file: {uploaded_file.name}")
        resume_text = extract_text_from_file(uploaded_file)
        if resume_text:
            show_full = st.checkbox("Show full resume text")
            st.text_area("Resume Preview", resume_text if show_full else resume_text[:1000], height=300)

            # Prediction
            prediction = model.predict([resume_text])[0]
            role = role_mapping.get(prediction, "Unknown Role")
            st.success(f"**Predicted Role:** {role}")

            # Extract details
            email, experience, skills = extract_details(resume_text)
            st.write(f"**Email:** {email}")
            st.write(f"**Experience:** {experience}")
            st.write(f"**Skills:** {skills}")

            # Name extraction
            name = resume_text.split('\n')[0].strip() if resume_text else "Name not found"

            # Excel in sidebar
            excel_data = generate_excel(name, email, experience, skills, role)
            with st.sidebar:
                st.markdown("### 📥 Download Resume Info")
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name='resume_info.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key="download_excel"
                )

# --- Tab 2: Visualizations ---
with tab2:
    st.header("📊 Resume Visual Insights")
    if resume_text and skills and role:
        # --- Word Cloud ---
        st.subheader("Word Cloud")
        wc = WordCloud(width=600, height=300, background_color="white").generate(resume_text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # --- Top Keywords ---
        st.subheader("Top Keywords")
        words = transform_text(resume_text).split()
        word_freq = Counter(words).most_common(10)
        df_keywords = pd.DataFrame(word_freq, columns=["Keyword", "Frequency"])
        st.bar_chart(df_keywords.set_index("Keyword"))

        # --- Role Match Gauge ---
        st.subheader("Role Skills Match")
        expected = set(role_skills.get(role, []))
        present = set(transform_text(resume_text).split())
        match_percent = round((len(expected & present) / len(expected)) * 100, 2) if expected else 0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=match_percent,
            title={'text': "Skill Match %"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig_gauge)

        # --- Matched vs Missing Skills ---
        st.subheader("Matched vs Missing Skills")
        expected = set(role_skills.get(role, []))
        present = set(transform_text(resume_text).split())

        matched = expected & present
        missing = expected - present

        df_skill_match = pd.DataFrame({
            "Skill Status": ["Matched Skills", "Missing Skills"],
            "Count": [len(matched), len(missing)]
        })

        st.bar_chart(df_skill_match.set_index("Skill Status"))

        # Optional: show exact skills in text
        st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

    else:
        st.warning("Please upload a resume in the Upload Resume tab first.")

# --- Tab 3: About ---
with tab3:
    st.header("ℹ️ About This App")
    st.markdown("""
    This application leverages a machine learning model trained on a curated dataset of categorized resumes to automatically predict the most relevant job role for a candidate. 
    It supports the following roles:
    - React Developer
    - SQL Developer
    - Peoplesoft Consultant
    - Workday Consultant

    In addition to role prediction, the app:
    - Extracts essential details such as email address, total work experience, and listed skills.
    - Generates multiple visual insights including word clouds, top keywords, skill match analysis, and more, 
      helping both recruiters and candidates understand the resume’s content at a glance.

    The primary goal of this tool is to assist recruiters in quickly screening resumes and identifying potential 
    matches for job openings, while also giving job seekers feedback on how well their resumes align with a 
    specific role.
    """)

    # GitHub raw file link
    pdf_url = "https://github.com/farah-web/resume-classification-app/raw/main/Resume-Classification-Using-Machine-Learning.pdf"
    st.markdown(f"[📥 Download Project Report (PDF)]({pdf_url})", unsafe_allow_html=True)

