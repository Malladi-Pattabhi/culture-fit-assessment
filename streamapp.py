import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "Definition\nCulture fit assessment refers to the evaluation of how well a candidate's stated values, preferences, "
        "and expectations align with the cultural aspects of a job description. This involves comparing the explicit "
        "statements made by the candidate in their application materials with the cultural attributes and expectations "
        "described in the job description.\n\n"
        "Identification and Extraction\n"
        "Analyze both the job description and the candidate's provided documents (resume, cover letter, application forms, etc.) "
        "to identify explicit statements about the following categories:\n"
        "Company Culture: Policy, goals, practices, core values, mission, vision, strategy.\n"
        "Team Dynamics: Working style, people, communication, collaboration.\n"
        "Work Environment: Physical space, remote work, office setup, flexibility.\n"
        "Personal Values: Integrity, work-life balance, learning, growth, innovation.\n"
        "Management Style: Leadership, decision-making, feedback, support.\n"
        "Diversity & Inclusion: Inclusion practices, non-discrimination, equity, representation.\n\n"
        "Output Format\n"
        "For each identified value or cultural expectation, provide the following:\n"
        "- Category (e.g., Company Culture, Team Dynamics, etc.)\n"
        "- Statement: A direct quote or close paraphrase of the candidate's stated preference or value.\n"
        "- Source: The specific document and section where this information was found.\n"
        "- Context: A brief description of the surrounding context in which this statement was made.\n"
        "- Relevance Score: (1-5, where 5 is highly emphasized or frequently mentioned, and 1 is briefly mentioned)\n"
        "Only include explicitly stated preferences and values. Do not infer or assume values that are not directly expressed."
    ),
)

# Load the BERT model for Cosine Similarity
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Functions for Cosine Similarity Calculations
def calculate_culture_fit_score(job_desc, resume_text, behavioral_answers):
    job_embedding = embedding_model.encode([job_desc])
    resume_embedding = embedding_model.encode([resume_text])
    behavioral_embeddings = [embedding_model.encode(answer) for answer in behavioral_answers]

    similarity_job_resume = cosine_similarity(job_embedding, resume_embedding)[0][0]
    similarity_behavioral = np.mean([cosine_similarity(job_embedding, [be]).flatten()[0] for be in behavioral_embeddings])

    culture_fit_score = (0.7 * similarity_job_resume) + (0.3 * similarity_behavioral)
    reasoning = f"Resume aligns with the job description by {similarity_job_resume:.2f}. " \
                f"Behavioral responses further fit with a score of {similarity_behavioral:.2f}."
    
    return culture_fit_score, reasoning

# Gemini API-based Culture Fit Assessment
def generate_gemini_assessment(job_description, resume_text, behavioral_answers):
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    f"Job Description: {job_description}\n\n"
                    f"Resume: {resume_text}\n\n"
                    f"Behavioral Answers: {' '.join(behavioral_answers)}"
                ],
            }
        ]
    )
    response = chat_session.send_message("Analyze the candidate's culture fit based on the provided information.")
    return response.text

# PDF Text Extraction
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Streamlit App Layout
st.title("Culture Fit Assessment Tool")

# Sidebar for file upload and model selection
st.sidebar.title("Input Data and Model Selection")
model_choice = st.sidebar.radio("Choose Model", ["Cosine Similarity (BERT Model)", "Gemini API Model"])

# Display text input fields
job_description = st.text_area("Enter the Job Description")
resume_text = st.text_area("Enter the Resume Text")

# Behavioral Questions Section
st.header("Behavioral Question Responses")
behavioral_answers = [
    st.text_input("1. Describe a time you worked in a team:"),
    st.text_input("2. How do you handle conflict?"),
    st.text_input("3. What motivates you to work hard?"),
    st.text_input("4. Describe a challenging project you worked on:"),
    st.text_input("5. How do you prioritize tasks when under pressure?")
]

# Calculate and display results
if st.button("Calculate Culture Fit"):
    if model_choice == "Cosine Similarity (BERT Model)":
        if job_description and resume_text:
            score, reasoning = calculate_culture_fit_score(job_description, resume_text, behavioral_answers[:2])
            st.subheader("Cosine Similarity (BERT) Culture Fit Score")
            st.write(f"Culture Fit Score: {score}")
            st.write(f"Reasoning: {reasoning}")
        else:
            st.error("Please fill in all fields to proceed.")
    elif model_choice == "Gemini API Model":
        if job_description and resume_text:
            response = generate_gemini_assessment(job_description, resume_text, behavioral_answers)
            st.subheader("Gemini API Model Culture Fit Assessment")
            st.write(response)
        else:
            st.error("Please fill in all fields to proceed.")
