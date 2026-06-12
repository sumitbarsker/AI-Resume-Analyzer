import streamlit as st
import pdfplumber
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

# ---------------- API ----------------

genai.configure(
    api_key="
)

model_ai = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# ---------------- LOAD ML MODEL ----------------

role_model = pickle.load(
    open("model/resume_model.pkl", "rb")
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ResumeIQ AI",
    page_icon="📄",
    layout="centered"
)

# ---------------- CSS ----------------

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------- PDF TEXT EXTRACT ----------------

def extract_resume_text(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            content = page.extract_text()

            if content:
                text += content

    return text

# ---------------- ATS SCORE ----------------

def get_ats_score(resume, jd):

    data = [resume, jd]

    cv = CountVectorizer()

    matrix = cv.fit_transform(data)

    similarity = cosine_similarity(matrix)[0][1]

    return round(similarity * 100, 2)

# ---------------- UI ----------------

st.markdown(
    "<div class='title'>ResumeIQ AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Smart AI Resume Analyzer</div>",
    unsafe_allow_html=True
)

uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

# ---------------- PROCESS ----------------

if uploaded_resume and job_description:

    resume_text = extract_resume_text(
        uploaded_resume
    )

    # ATS SCORE

    ats_score = get_ats_score(
        resume_text,
        job_description
    )

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.subheader("ATS Match Score")

    st.markdown(
        f"<div class='score'>{ats_score}%</div>",
        unsafe_allow_html=True
    )

    st.progress(int(ats_score))

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ROLE PREDICTION

    predicted_role = role_model.predict(
        [resume_text]
    )[0]

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.subheader("Best Matching Role")

    st.success(predicted_role)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # AI FEEDBACK

    prompt = f"""
    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Analyze this resume and give:

    1. Missing skills
    2. Resume improvements
    3. Better bullet points
    4. Recruiter impression
    """

    with st.spinner("Analyzing Resume..."):

        response = model_ai.generate_content(
            prompt
        )

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.subheader("AI Resume Feedback")

    st.write(response.text)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
