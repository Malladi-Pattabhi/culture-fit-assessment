import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import PyPDF2
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

st.markdown(
    """
    <style>
    .stApp {
        background-color: #B7CD7A;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the BERT model for Cosine Similarity
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load the GPT-2 model and tokenizer from Hugging Face
@st.cache_resource
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

embedding_model = load_embedding_model()
tokenizer, gpt2_model = load_gpt2_model()

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

# GPT-2 Text Generation for Culture Fit Assessment
def generate_gpt2_assessment(job_description, resume_text, behavioral_answers):
    system_instructions = (
        "Definition\n"
        "Culture fit assessment refers to the evaluation of how well a candidate's stated values, preferences, "
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
        
        "Scoring System\n"
        "Best Fit (4-5): Strong alignment\n"
        "Average Fit (2-3): Some alignment with areas needing improvement\n"
        "Poor Fit (1): Misalignment\n\n"
        
        "Additional Instructions\n"
        "Only include explicitly stated preferences and values. Do not infer or assume values that are not directly expressed."
    )

    # Truncate each input section to fit within GPT-2’s 1024-token limit
    truncated_job_description = job_description[:500]
    truncated_resume_text = resume_text[:500]
    truncated_behavioral_answers = "\n".join(behavioral_answers[:2])[:500]

    model_input = (
        f"{system_instructions}\n\n"
        f"Job Description:\n{truncated_job_description}\n\n"
        f"Resume:\n{truncated_resume_text}\n\n"
        f"Behavioral Answers:\n{truncated_behavioral_answers}\n\n"
        "Analyze the candidate's culture fit based on the provided information:"
    )

    # Tokenize with truncation to ensure it stays within bounds
    inputs = tokenizer(model_input, return_tensors="pt", truncation=True, max_length=1024)
    outputs = gpt2_model.generate(
        **inputs,
        max_new_tokens=150,  # Limit the output length
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


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
model_choice = st.sidebar.radio("Choose Model", ["Cosine Similarity (BERT Model)", "GPT-2 Model"])

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
    elif model_choice == "GPT-2 Model":
        if job_description and resume_text:
            response = generate_gpt2_assessment(job_description, resume_text, behavioral_answers[:2])
            st.subheader("GPT-2 Model Culture Fit Assessment")
            st.write(response)
        else:
            st.error("Please fill in all fields to proceed.")