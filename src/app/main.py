"""
Web application entry point and routes (Flask).
"""

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from src.core.rag import answer_question
from src.core.ingestion import process_uploaded_file
from src.core.types import PromptStyle
from src.core.database import DatabaseManager


UPLOAD_ROOT = os.path.join(os.getcwd(), "data", "raw")


def _save_upload(file) -> str:
    """Persist an uploaded file to disk and return its path."""
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_ROOT, filename)
    file.save(file_path)
    return file_path


def create_app():
    # __name__ tells Flask where to look for templates and static files
    app = Flask(__name__)
    db_manager = DatabaseManager()
    
    # --- Routes ---

    @app.route('/')
    def index():
        """Landing page: list courses or create one."""
        courses = db_manager.list_courses()
        return render_template('index.html', courses=courses)

    @app.route('/courses', methods=['POST'])
    def create_course():
        name = request.form.get('course_name', '').strip()
        if not name:
            return redirect(url_for('index'))
        course = db_manager.add_course(name)
        return redirect(url_for('view_course', course_id=course.course_id))

    @app.route('/courses/<course_id>')
    def view_course(course_id: str):
        course = db_manager.get_course(course_id)
        if not course:
            return redirect(url_for('index'))

        exams = db_manager.list_exams_for_course(course_id)
        documents = db_manager.get_documents_for_course(course_id)
        return render_template(
            'course.html',
            course=course,
            exams=exams,
            documents=documents,
        )

    @app.route('/courses/<course_id>/documents', methods=['POST'])
    def upload_course_document(course_id: str):
        course = db_manager.get_course(course_id)
        if not course:
            return redirect(url_for('index'))

        file = request.files.get('document')
        if not file or file.filename == '':
            return redirect(url_for('view_course', course_id=course_id))

        file_path = _save_upload(file)
        process_uploaded_file(file_path, course_id=course_id, exam_ids=None, db_manager=db_manager)
        return redirect(url_for('view_course', course_id=course_id))

    @app.route('/courses/<course_id>/exams', methods=['POST'])
    def create_exam(course_id: str):
        course = db_manager.get_course(course_id)
        if not course:
            return redirect(url_for('index'))
        exam_name = request.form.get('exam_name', '').strip()
        if not exam_name:
            return redirect(url_for('view_course', course_id=course_id))

        exam = db_manager.add_exam(course_id=course_id, name=exam_name)
        return redirect(url_for('view_exam', course_id=course_id, exam_id=exam.exam_id))

    @app.route('/courses/<course_id>/exams/<exam_id>')
    def view_exam(course_id: str, exam_id: str):
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        if not course or not exam:
            return redirect(url_for('index'))

        exam_docs = db_manager.get_documents_for_exam(exam_id)
        course_docs = db_manager.get_documents_for_course(course_id)
        attached_ids = {doc.doc_id for doc in exam_docs}
        attachable_docs = [doc for doc in course_docs if doc.doc_id not in attached_ids]
        problems = db_manager.list_problems_for_exam(exam_id)
        return render_template(
            'exam.html',
            course=course,
            exam=exam,
            exam_documents=exam_docs,
            attachable_docs=attachable_docs,
            problems=problems,
        )

    @app.route('/courses/<course_id>/exams/<exam_id>/documents', methods=['POST'])
    def upload_exam_document(course_id: str, exam_id: str):
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        if not course or not exam:
            return redirect(url_for('index'))

        file = request.files.get('document')
        if not file or file.filename == '':
            return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

        file_path = _save_upload(file)
        process_uploaded_file(file_path, course_id=course_id, exam_ids=[exam_id], db_manager=db_manager)
        return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

    @app.route('/courses/<course_id>/exams/<exam_id>/documents/attach', methods=['POST'])
    def attach_documents(course_id: str, exam_id: str):
        doc_ids = request.form.getlist('doc_ids')
        if doc_ids:
            db_manager.attach_documents_to_exam(exam_id, doc_ids)
        return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

    @app.route('/courses/<course_id>/exams/<exam_id>/ask', methods=['POST'])
    def ask_question(course_id: str, exam_id: str):
        """Handles the question form submission scoped to an exam."""
        
        # Get the question from the form
        question_text = request.form['question_text']
        
        # Get the selected style (default to 'minimal')
        style_str = request.form.get('style', 'minimal').upper()
        try:
            selected_style = PromptStyle[style_str]
        except KeyError:
            selected_style = PromptStyle.MINIMAL
        
        # --- RAG LOGIC ---
        result = answer_question(
            question_text=question_text,
            exam_id=exam_id,
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
            chunks=display_chunks,
            course_id=course_id,
            exam_id=exam_id,
        )

    return app

# --- Entry Point ---
# This block runs when you execute 'python src/app/main.py'
if __name__ == '__main__':
    app = create_app()
    # debug=True automatically reloads the server when you save changes
    app.run(debug=True)
