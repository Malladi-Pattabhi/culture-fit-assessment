import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
import google.generativeai as genai

# Initialize models and API configuration only once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    return tokenizer, model

@st.cache_resource
def configure_gemini_api():
    genai.configure(api_key="YOUR_GEMINI_API_KEY")
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
        system_instruction="""
        Culture fit assessment involves evaluating candidate alignment with company culture based on explicit statements in the job description and candidate documents.
        """,
    )
    return model

embedding_model = load_embedding_model()
tokenizer, gpt2_model = load_gpt2_model()
gemini_model = configure_gemini_api()

# Functions for similarity calculations and Gemini API calls
def calculate_culture_fit_score(job_desc, resume_text, behavioral_answers):
    job_embedding = embedding_model.encode([job_desc])
    resume_embedding = embedding_model.encode([resume_text])
    behavioral_embeddings = [embedding_model.encode(answer) for answer in behavioral_answers]

    similarity_job_resume = cosine_similarity(job_embedding, resume_embedding)[0][0]
    similarity_behavioral = np.mean([cosine_similarity(job_embedding, [be]).flatten()[0] for be in behavioral_embeddings])

    culture_fit_score = (0.7 * similarity_job_resume) + (0.3 * similarity_behavioral)
    reasoning = f"Resume aligns with the job description by {similarity_job_resume:.2f}. " \
                f"Behavioral responses further fit with a score of {similarity_behavioral:.2f}, " \
                f"showing qualities like teamwork and adaptability."
    
    return culture_fit_score, reasoning

def get_gpt_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = gpt2_model(**inputs)
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()
    return embedding

def generate_gemini_assessment(job_description, resume_text, behavioral_answers):
    prompt = f"""
    Job Description: {job_description}

    Resume: {resume_text}

    Behavioral Questions and Answers:
    1. {behavioral_answers[0]}
    2. {behavioral_answers[1]}

    Evaluate the culture fit score and alignment reasoning.
    """

    chat_session = gemini_model.start_chat(history=[{"role": "user", "parts": [prompt]}])
    response = chat_session.send_message("Evaluate the culture fit score and alignment reasoning.")
    return response.text

# Streamlit App Layout
st.title("Culture Fit Assessment Tool")

# Sidebar for file upload and model selection
st.sidebar.title("Input Data and Model Selection")
uploaded_job_file = st.sidebar.file_uploader("Upload Job Descriptions File (.xlsx)", type="xlsx")
uploaded_resume_file = st.sidebar.file_uploader("Upload Resumes File (.xlsx)", type="xlsx")
model_choice = st.sidebar.radio("Choose Model", ["Cosine Similarity (BERT Model)", "Gemini Model"])

# Display text input fields if files are not uploaded
if uploaded_job_file and uploaded_resume_file:
    job_data = pd.read_excel(uploaded_job_file)
    candidate_data = pd.read_excel(uploaded_resume_file)
    st.write("Job Descriptions and Resumes loaded successfully!")
else:
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
        if uploaded_job_file and uploaded_resume_file:
            # Loop through each job and resume for batch processing
            culture_fit_scores = []
            for i, job_desc in enumerate(job_data['job_description']):
                for j, resume in enumerate(candidate_data['Resume']):
                    score, reasoning = calculate_culture_fit_score(job_desc, resume, behavioral_answers[:2])
                    culture_fit_scores.append({
                        'Job_ID': i,
                        'Resume_ID': j,
                        'Culture_Fit_Score': score,
                        'Reasoning': reasoning
                    })
            st.subheader("Cosine Similarity (BERT) Culture Fit Scores")
            st.write(pd.DataFrame(culture_fit_scores))
        elif job_description and resume_text:
            score, reasoning = calculate_culture_fit_score(job_description, resume_text, behavioral_answers[:2])
            st.subheader("Cosine Similarity (BERT) Culture Fit Score")
            st.write(f"Culture Fit Score: {score}")
            st.write(f"Reasoning: {reasoning}")
        else:
            st.error("Please fill in all fields or upload files to proceed.")

    elif model_choice == "Gemini Model":
        if uploaded_job_file and uploaded_resume_file:
            gemini_results = []
            for i, job_desc in enumerate(job_data['job_description']):
                for j, resume in enumerate(candidate_data['Resume']):
                    response = generate_gemini_assessment(job_desc, resume, behavioral_answers[:2])
                    gemini_results.append({
                        'Job_ID': i,
                        'Resume_ID': j,
                        'Culture_Fit_Assessment': response
                    })
            st.subheader("Gemini Model Culture Fit Assessments")
            st.write(pd.DataFrame(gemini_results))
        elif job_description and resume_text:
            response = generate_gemini_assessment(job_description, resume_text, behavioral_answers[:2])
            st.subheader("Gemini Model Culture Fit Assessment")
            st.write(response)
        else:
            st.error("Please fill in all fields or upload files to proceed.")