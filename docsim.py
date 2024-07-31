import os
import spacy
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the German spaCy model
nlp = spacy.load("de_core_news_md")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    doc = nlp(text)
    # Keep only nouns, proper nouns, verbs, and numbers and dates 
    tokens = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'NUM','DATE']]
    return ' '.join(tokens)

def load_invoices(directory):
    invoices = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            path = os.path.join(directory, filename)
            text = extract_text_from_pdf(path)
            processed_text = preprocess_text(text)
            invoices[filename] = processed_text
    return invoices

def calculate_similarity(input_text, database_texts):
    input_vector = nlp(input_text).vector.reshape(1, -1)
    database_vectors = np.array([nlp(text).vector for text in database_texts.values()])
    
    cosine_similarities = cosine_similarity(input_vector, database_vectors)
    return cosine_similarities[0]

def find_most_similar_invoice(input_invoice, database):
    input_text = extract_text_from_pdf(input_invoice)
    processed_input_text = preprocess_text(input_text)
    similarities = calculate_similarity(processed_input_text, database)
    
    most_similar_index = np.argmax(similarities)
    most_similar_filename = list(database.keys())[most_similar_index]
    similarity_score = similarities[most_similar_index]
    
    return most_similar_filename, similarity_score

# Main execution
if __name__ == "__main__":
    train_directory = os.getcwd()+'/train'
    test_directory = os.getcwd()+'/test'
    
    # Load training invoices
    print("Loading and processing training invoices...")
    invoice_database = load_invoices(train_directory)
    
    # Process test invoices
    print("Processing test invoices...")
    for test_invoice in os.listdir(test_directory):
        if test_invoice.endswith('.pdf'):
            test_invoice_path = os.path.join(test_directory, test_invoice)
            most_similar, score = find_most_similar_invoice(test_invoice_path, invoice_database)
            print(f"Test Invoice: {test_invoice}")
            print(f"Most Similar: {most_similar}")
            print(f"Similarity Score: {score:.4f}")
            print()