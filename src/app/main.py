"""
Web application entry point and routes (Flask).
"""

import os
from flask import Flask, render_template, request, redirect, url_for

def create_app():
    
    # __name__ tells Flask where to look for templates and static files
    app = Flask(__name__)
    
    # --- Routes ---

    @app.route('/')
    def index():
        """Serves the homepage (index.html)."""
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handles the file upload form."""
        if 'document' not in request.files:
            # Handle error
            return redirect(url_for('index'))
        
        file = request.files['document']
        
        if file.filename == '':
            # Handle error
            return redirect(url_for('index'))

        if file:
            # --- ML LOGIC PLUG-IN ---
            # TODO: 1. Save the file to a secure location (e.g., data/raw/)
            # TODO: 2. Call a function from src.core.ingestion to process the file
            #       (e.g., ingestion.process_uploaded_file(file_path))
            print(f"File {file.filename} uploaded successfully.") # Placeholder
        
        # Send the user back to the homepage
        return redirect(url_for('index'))


    @app.route('/ask', methods=['POST'])
    def ask_question():
        """Handles the question form submission (Decision B2.2)."""
        
        # Get the question from the form
        question_text = request.form['question_text']
        
        # --- ML LOGIC PLUG-IN ---
        # TODO: 1. Call a function from src.core.rag to get the answer
        #       (e.g., result = rag.answer_question(question_text))
        
        # --- DUMMY DATA (for testing the frontend) ---
        # Replace this with the actual result from your RAG function
        dummy_explanation = f"This is a dummy explanation for your question: '{question_text}'. The system will generate a real answer here."
        dummy_chunks = [
            {"text": "This is the first dummy chunk of text from a document.", "source": "dummy_doc_1.pdf"},
            {"text": "This is a second, even more relevant dummy chunk.", "source": "dummy_doc_2.pdf"},
        ]
        # --- END DUMMY DATA ---

        # Render the answer.html page, passing in the data
        return render_template(
            'answer.html',
            question=question_text,
            explanation=dummy_explanation, # Use dummy_explanation
            chunks=dummy_chunks            # Use dummy_chunks
        )

    return app

# --- Entry Point ---
# This block runs when you execute 'python src/app/main.py'
if __name__ == '__main__':
    app = create_app()
    # debug=True automatically reloads the server when you save changes
    app.run(debug=True)