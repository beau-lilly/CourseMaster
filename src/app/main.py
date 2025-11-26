"""
Web application entry point and routes (Flask).
"""

import os
from flask import Flask, render_template, request, redirect, url_for

# Import RAG Logic
# Note: We import inside routes or here if circular deps aren't an issue.
# Here is fine as src.app depends on src.core, not vice-versa.
from src.core.rag import answer_question
from src.core.ingestion import process_uploaded_file
from src.core.types import PromptStyle
from src.core.database import DatabaseManager

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
            # 1. Save the file to a secure location
            upload_folder = os.path.join(os.getcwd(), 'data', 'raw')
            os.makedirs(upload_folder, exist_ok=True)
            
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            
            # 2. Call ingestion to process the file
            try:
                process_uploaded_file(file_path)
                print(f"File {file.filename} uploaded and ingested successfully.")
            except Exception as e:
                print(f"Error ingesting file: {e}")
                # Optional: Flash message to user

        
        # Send the user back to the homepage
        return redirect(url_for('index'))


    @app.route('/ask', methods=['POST'])
    def ask_question():
        """Handles the question form submission."""
        
        # Get the question from the form
        question_text = request.form['question_text']
        
        # Get the selected style (default to 'minimal')
        style_str = request.form.get('style', 'minimal').upper()
        try:
            selected_style = PromptStyle[style_str]
        except KeyError:
            selected_style = PromptStyle.MINIMAL
        
        # --- RAG LOGIC ---
        db_manager = DatabaseManager()
        result = answer_question(
            question_text=question_text,
            prompt_style=selected_style,
            db_manager=db_manager
        )
        
        # Get filenames
        doc_ids = [c.doc_id for c in result.used_chunks]
        filenames = db_manager.get_doc_filenames(list(set(doc_ids)))

        # Prepare chunks for display with rank and similarity score
        display_chunks = []
        scores = result.scores or []
        for rank, chunk in enumerate(result.used_chunks, start=1):
            score = scores[rank - 1] if rank - 1 < len(scores) else None
            display_chunks.append({
                "rank": rank,
                "text": chunk.chunk_text,
                "source": filenames.get(chunk.doc_id, chunk.doc_id),
                "chunk_index": chunk.chunk_index,
                "similarity": score
            })

        # Render the answer.html page, passing in the data
        return render_template(
            'answer.html',
            question=result.question,
            explanation=result.answer, 
            chunks=display_chunks
        )

    return app

# --- Entry Point ---
# This block runs when you execute 'python src/app/main.py'
if __name__ == '__main__':
    app = create_app()
    # debug=True automatically reloads the server when you save changes
    app.run(debug=True)
