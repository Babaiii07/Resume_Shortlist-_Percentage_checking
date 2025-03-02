import streamlit as st
import easyocr
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
import PyPDF2

load_dotenv()

st.title('Resume Shortlist Applications')

file_input_resume = st.file_uploader('Enter your resume file', type=['png', 'jpg', 'jpeg', 'pdf'])
text_file_input_resume = st.file_uploader('Enter your job descriptions', type=['txt'])

def extract_text_easyocr(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img, detail=0)
    return "\n".join(result)

def text_loader(file):
    return file.read().decode('utf-8')

if file_input_resume and text_file_input_resume:
    if file_input_resume.type in ['image/png', 'image/jpeg', 'image/jpg']:
        image_bytes = file_input_resume.read()
        text = extract_text_easyocr(image_bytes)
    elif file_input_resume.type == 'application/pdf':
        pdf_reader = PyPDF2.PdfReader(file_input_resume)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    else:
        st.error("Unsupported file format for resume.")
        st.stop()
    
    text_job_descriptions = text_loader(text_file_input_resume)
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_resume_text = embeddings.embed_query(text)
    vector_job_description = embeddings.embed_query(text_job_descriptions)
    
    def calculate_similarity(vector1, vector2):
        return (cosine_similarity([vector1], [vector2])[0][0] * 100)
    
    if st.button('Check your Score'):
        similarity = calculate_similarity(vector_resume_text, vector_job_description)
        st.write(f'The similarity between your resume and job descriptions is {similarity:.2f}%')