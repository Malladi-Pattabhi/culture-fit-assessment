import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
import os
import google.generativeai as genai

file_pathj = 'job_descriptions.xlsx'
file_pathr = 'resumes.xlsx'


job_data = pd.read_excel(file_pathj)
candidate_data = pd.read_excel(file_pathr)

behavioral_questions = [
    "Describe a time you worked in a team.",
    "How do you handle conflict?",
    "What motivates you to work hard?",
    "Describe a challenging project you worked on.",
    "How do you prioritize tasks when under pressure?"
]

# Load the model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert job descriptions and resumes to embeddings
job_embeddings = embedding_model.encode(job_data['job_description'].tolist())
resume_embeddings = embedding_model.encode(candidate_data['Resume'].tolist())
behavioral_embeddings = embedding_model.encode(behavioral_questions)

# Calculate similarity scores
similarity_scores = cosine_similarity(job_embeddings, resume_embeddings)

culture_fit_scores = []
culture_pillars = {
    'Core Values': 5,
    'Behavior/Attitude': 4,
    'Mission': 4,
    'Vision': 4,
    'Practices': 3,
    'Goals': 3,
    'Work-Life Balance/Working Style': 3,
    'Diversity and Inclusion': 3,
    'Strategy': 2,
    'Policy': 2
}
for i, job_embedding in enumerate(job_embeddings):
    for j, resume_embedding in enumerate(resume_embeddings):
        culture_fit_score = cosine_similarity([job_embedding], [resume_embedding]).flatten()[0]

        question_match_scores = cosine_similarity(
            [resume_embedding], behavioral_embeddings
        ).flatten()

        total_score = culture_fit_score * culture_pillars['Core Values']
        for idx, question_score in enumerate(question_match_scores):
            if idx < 2:
                total_score += question_score * culture_pillars['Behavior/Attitude']
            else:
                total_score += question_score * culture_pillars['Mission']
        culture_fit_scores.append({
            'Job_ID': i,
            'Resume_ID': j,
            'Culture_Fit_Score': total_score / sum(culture_pillars.values())
        })

fit_score_df = pd.DataFrame(culture_fit_scores)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

def get_gpt_embedding(text):
    """
    Generate an embedding for a given text using GPT.
    """
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().detach().numpy()
    return embedding
new_job_description = "We are looking for a team-oriented individual who values collaboration and has strong critical thinking skills."
new_resume_text = "Experienced professional with a passion for teamwork and problem-solving. Known for strong leadership and adaptability in fast-paced environments."

new_job_embedding = get_gpt_embedding(new_job_description)
new_resume_embedding = get_gpt_embedding(new_resume_text)

fit_score = cosine_similarity([new_job_embedding], [new_resume_embedding]).flatten()[0]

reasoning_prompt = (
    f"Job Description: {new_job_description}\n"
    f"Candidate Resume: {new_resume_text}\n"
    "Provide a culture fit match score (0-100) and reasoning. Explain how well the candidate aligns with core values, "
    "team orientation, critical thinking, and collaboration skills."
)
reasoning_output = f"Based on the job description, the candidate matches well with the required team orientation and collaborative spirit. They demonstrate critical thinking and leadership, which aligns with the company’s values. Fit Score: {round(fit_score * 100, 2)}."

print("Culture Fit Score:", round(fit_score * 100, 2))
print("Reasoning:", reasoning_output)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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

job_desc_input = "Looking for a self-motivated, team-oriented individual with strong communication and leadership skills."
resume_text_input = "Experienced team player with strong communication and leadership abilities. Proven track record in collaborative environments."
behavioral_answers_input = ["I work well with teams by valuing each member's input.", "In conflicts, I aim for a balanced resolution that supports the team."]

# Calculate culture fit score and reasoning
score, reasoning = calculate_culture_fit_score(job_desc_input, resume_text_input, behavioral_answers_input)

print("Culture Fit Score:", score)
print("Match Reasoning:", reasoning)

# Configure the API with your Gemini API Key
genai.configure(api_key="AIzaSyAIzX4GLFp0_cSf_Jv0GbFpfxIEjK1dDRg")

# Define the configuration for the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Set up the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
    Culture fit assessment involves evaluating candidate alignment with company culture based on explicit statements in the job description and candidate documents.
    Analyze both the job description and the candidate's documents to identify values in categories like Company Culture, Team Dynamics, Work Environment, Personal Values, Management Style, Diversity & Inclusion.
    Output Format:
    - Category
    - Statement
    - Source
    - Context
    - Relevance Score (1-5)
    Scoring: Provide Best Fit, Average Fit, or Poor Fit based on alignment with the company's core values and expectations.
    """,
)

# Example input data
job_description = "Looking for a self-motivated, team-oriented individual with strong communication and leadership skills."
resume_text = """
Experienced team player with strong communication and leadership abilities. Proven track record in collaborative environments.
"""
behavioral_answers = ["I work well with teams by valuing each member's input.", "In conflicts, I aim for a balanced resolution that supports the team."]

# Formulate the prompt combining job description, resume, and behavioral questions
prompt = f"""
Job Description: {job_description}

Resume: {resume_text}

Behavioral Questions and Answers:
1. Describe a time you worked in a team: {behavioral_answers[0]}
2. How do you handle conflict?: {behavioral_answers[1]}

Please assess the culture fit based on alignment with company culture pillars and provide a score and reasoning.
"""

# Start chat session with the formatted prompt
chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])

# Send the message and retrieve the response
response = chat_session.send_message("Evaluate the culture fit score and alignment reasoning.")
print("Culture Fit Assessment:", response.text)