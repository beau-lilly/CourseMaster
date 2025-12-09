"""
Web application entry point and routes (Flask).
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from src.core.rag import answer_question
from src.core.ingestion import process_uploaded_file
from src.core.types import PromptStyle
from src.core.database import DatabaseManager
from src.core.retrieval import index_problem_context


UPLOAD_ROOT = os.path.join(os.getcwd(), "data", "raw")


def _save_upload(file) -> str:
    """Persist an uploaded file to disk and return its path."""
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    candidate = filename
    file_path = os.path.join(UPLOAD_ROOT, candidate)

    # Avoid clobbering files with the same name within a batch
    counter = 1
    while os.path.exists(file_path):
        candidate = f"{name}_{counter}{ext}"
        file_path = os.path.join(UPLOAD_ROOT, candidate)
        counter += 1

    file.save(file_path)
    return file_path


def create_app():
    # __name__ tells Flask where to look for templates and static files
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
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

        files = [
            f for f in request.files.getlist('documents')
            if f and f.filename
        ]
        if not files:
            return redirect(url_for('view_course', course_id=course_id))

        success_count = 0
        for file in files:
            file_path = _save_upload(file)
            _, message = process_uploaded_file(
                file_path,
                course_id=course_id,
                exam_ids=None,
                db_manager=db_manager
            )
            if message:
                flash(message, "warning")
            else:
                success_count += 1

        if success_count:
            flash(f"{success_count} document(s) uploaded successfully.", "success")
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
        assignments = db_manager.list_assignments_for_exam(exam_id)
        assignment_lookup = {a.assignment_id: a for a in assignments}

        display_mode = request.args.get("display", "chunks")
        ranking_strategy = request.args.get("ranking", "frequency").lower()
        ranking_is_frequency = ranking_strategy == "frequency"
        try:
            ranking_limit = max(1, int(request.args.get("limit", "5")))
        except ValueError:
            ranking_limit = 5

        ranking_options = db_manager.available_ranking_strategies()
        top_chunks = []
        top_documents = []

        if display_mode == "documents":
            display_mode = "documents"
            top_documents_raw = db_manager.get_top_documents_for_exam(
                exam_id=exam_id,
                ranking_strategy=ranking_strategy,
                limit=ranking_limit,
            )
            top_documents = [
                {
                    "rank": idx,
                    "doc_id": row["doc_id"],
                    "filename": row["filename"],
                    "score": row["score"],
                    "score_display": row["score"] if ranking_is_frequency else None,
                }
                for idx, row in enumerate(top_documents_raw, start=1)
            ]
        else:
            display_mode = "chunks"
            top_chunks_raw = db_manager.get_top_chunks_for_exam(
                exam_id=exam_id,
                ranking_strategy=ranking_strategy,
                limit=ranking_limit,
            )
            doc_lookup = db_manager.get_doc_filenames(
                list({row["doc_id"] for row in top_chunks_raw})
            )
            top_chunks = [
                {
                    "rank": idx,
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "doc_name": doc_lookup.get(row["doc_id"], row["doc_id"]),
                    "chunk_index": row["chunk_index"],
                    "chunk_text": row["chunk_text"],
                    "score": row["score"],
                    "score_display": row["score"] if ranking_is_frequency else None,
                }
                for idx, row in enumerate(top_chunks_raw, start=1)
            ]
        return render_template(
            'exam.html',
            course=course,
            exam=exam,
            exam_documents=exam_docs,
            attachable_docs=attachable_docs,
            problems=problems,
            assignments=assignments,
            assignment_lookup=assignment_lookup,
            display_mode=display_mode,
            ranking_strategy=ranking_strategy,
            ranking_is_frequency=ranking_is_frequency,
            ranking_limit=ranking_limit,
            ranking_options=ranking_options,
            top_chunks=top_chunks,
            top_documents=top_documents,
        )

    @app.route('/courses/<course_id>/exams/<exam_id>/documents', methods=['POST'])
    def upload_exam_document(course_id: str, exam_id: str):
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        if not course or not exam:
            return redirect(url_for('index'))

        files = [
            f for f in request.files.getlist('documents')
            if f and f.filename
        ]
        if not files:
            return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

        success_count = 0
        for file in files:
            file_path = _save_upload(file)
            _, message = process_uploaded_file(
                file_path,
                course_id=course_id,
                exam_ids=[exam_id],
                db_manager=db_manager
            )
            if message:
                flash(message, "warning")
            else:
                success_count += 1

        if success_count:
            flash(f"{success_count} document(s) uploaded successfully.", "success")
        return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

    @app.route('/courses/<course_id>/exams/<exam_id>/documents/attach', methods=['POST'])
    def attach_documents(course_id: str, exam_id: str):
        doc_ids = request.form.getlist('doc_ids')
        if doc_ids:
            db_manager.attach_documents_to_exam(exam_id, doc_ids)
        return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

    def _prepare_display_chunks(chunk_pairs):
        doc_ids = [c.doc_id for c, _ in chunk_pairs]
        filenames = db_manager.get_doc_filenames(list(set(doc_ids)))
        display_chunks = []
        for rank, (chunk, score) in enumerate(chunk_pairs, start=1):
            display_chunks.append({
                "rank": rank,
                "text": chunk.chunk_text,
                "source": filenames.get(chunk.doc_id, chunk.doc_id),
                "chunk_index": chunk.chunk_index,
                "similarity": score
            })
        return display_chunks

    @app.route('/courses/<course_id>/exams/<exam_id>/problems', methods=['POST'])
    def create_problem(course_id: str, exam_id: str):
        """Create a problem under an exam and precompute retrievals."""
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        if not course or not exam:
            return redirect(url_for('index'))

        problem_text = request.form.get('problem_text', '').strip()
        if not problem_text:
            return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

        new_assignment_name = request.form.get('new_assignment_name', '').strip()
        selected_assignment_id = request.form.get('assignment_id')
        assignment = None
        if new_assignment_name:
            assignment = db_manager.add_assignment(exam_id=exam_id, name=new_assignment_name)
        elif selected_assignment_id:
            assignment = db_manager.get_assignment(selected_assignment_id)

        if not assignment:
            return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

        number_raw = request.form.get('problem_number', '').strip()
        problem_number = int(number_raw) if number_raw else None

        try:
            problem = db_manager.add_problem(
                text=problem_text,
                exam_id=exam_id,
                assignment_id=assignment.assignment_id if assignment else None,
                problem_number=problem_number,
            )
        except ValueError:
            # Duplicate constraint violated; send back to exam page.
            return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

        # Precompute retrievals for this problem.
        index_problem_context(
            problem_text=problem_text,
            exam_id=exam_id,
            problem_id=problem.problem_id,
            db_manager=db_manager,
        )

        return redirect(
            url_for('view_problem', course_id=course_id, exam_id=exam_id, problem_id=problem.problem_id)
        )

    @app.route('/courses/<course_id>/exams/<exam_id>/problems/<problem_id>')
    def view_problem(course_id: str, exam_id: str, problem_id: str):
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        problem = db_manager.get_problem(problem_id)
        if not course or not exam or not problem or problem.exam_id != exam_id:
            return redirect(url_for('index'))

        assignment = db_manager.get_assignment(problem.assignment_id) if problem.assignment_id else None
        chunk_pairs = db_manager.get_chunks_for_problem(problem.problem_id)
        display_chunks = _prepare_display_chunks(chunk_pairs)
        questions = db_manager.list_questions_for_problem(problem.problem_id)

        return render_template(
            'problem.html',
            course=course,
            exam=exam,
            problem=problem,
            assignment=assignment,
            chunks=display_chunks,
            questions=questions,
        )

    @app.route('/courses/<course_id>/exams/<exam_id>/problems/<problem_id>/delete', methods=['POST'])
    def delete_problem(course_id: str, exam_id: str, problem_id: str):
        """Delete a problem and its related questions/retrievals."""
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        problem = db_manager.get_problem(problem_id)
        if not course or not exam or not problem or problem.exam_id != exam_id:
            return redirect(url_for('index'))

        db_manager.delete_problem(problem_id)
        flash("Problem deleted.", "success")
        return redirect(url_for('view_exam', course_id=course_id, exam_id=exam_id))

    @app.route('/courses/<course_id>/exams/<exam_id>/problems/<problem_id>/questions', methods=['POST'])
    def ask_question(course_id: str, exam_id: str, problem_id: str):
        """Handles the question form submission scoped to a problem."""
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        problem = db_manager.get_problem(problem_id)
        if not course or not exam or not problem:
            return redirect(url_for('index'))

        question_text = request.form.get('question_text', '').strip()
        if not question_text:
            return redirect(url_for('view_problem', course_id=course_id, exam_id=exam_id, problem_id=problem_id))

        style_str = request.form.get('style', 'minimal').upper()
        try:
            selected_style = PromptStyle[style_str]
        except KeyError:
            selected_style = PromptStyle.MINIMAL

        result = answer_question(
            question_text=question_text,
            problem_id=problem_id,
            prompt_style=selected_style,
            db_manager=db_manager
        )

        if result.question_id:
            return redirect(
                url_for(
                    'view_question',
                    course_id=course_id,
                    exam_id=exam_id,
                    problem_id=problem_id,
                    question_id=result.question_id,
                )
            )

        # Fallback: render inline if question could not be stored
        fallback_pairs = []
        for idx, chunk in enumerate(result.used_chunks):
            score = result.scores[idx] if result.scores and idx < len(result.scores) else None
            fallback_pairs.append((chunk, score))
        return render_template(
            'question.html',
            course=course,
            exam=exam,
            problem=problem,
            question_text=question_text,
            explanation=result.answer,
            chunks=_prepare_display_chunks(fallback_pairs),
            question=None,
            assignment=db_manager.get_assignment(problem.assignment_id) if problem.assignment_id else None,
        )

    @app.route('/courses/<course_id>/exams/<exam_id>/problems/<problem_id>/questions/<question_id>/delete', methods=['POST'])
    def delete_question(course_id: str, exam_id: str, problem_id: str, question_id: str):
        """Delete a single question for a problem."""
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        problem = db_manager.get_problem(problem_id)
        question = db_manager.get_question(question_id)
        if not course or not exam or not problem or not question:
            return redirect(url_for('index'))
        if problem.exam_id != exam_id or question.problem_id != problem_id:
            return redirect(url_for('index'))

        db_manager.delete_question(question_id)
        flash("Question deleted.", "success")
        return redirect(url_for('view_problem', course_id=course_id, exam_id=exam_id, problem_id=problem_id))

    @app.route('/courses/<course_id>/exams/<exam_id>/problems/<problem_id>/questions/<question_id>')
    def view_question(course_id: str, exam_id: str, problem_id: str, question_id: str):
        course = db_manager.get_course(course_id)
        exam = db_manager.get_exam(exam_id)
        problem = db_manager.get_problem(problem_id)
        question = db_manager.get_question(question_id)
        if not course or not exam or not problem or not question:
            return redirect(url_for('index'))
        if problem.exam_id != exam_id or question.problem_id != problem_id:
            return redirect(url_for('index'))

        assignment = db_manager.get_assignment(problem.assignment_id) if problem.assignment_id else None
        chunk_pairs = db_manager.get_chunks_for_problem(problem.problem_id)
        display_chunks = _prepare_display_chunks(chunk_pairs)

        return render_template(
            'question.html',
            course=course,
            exam=exam,
            problem=problem,
            question=question,
            question_text=question.question_text,
            explanation=question.answer_text,
            chunks=display_chunks,
            assignment=assignment,
        )

    return app

# --- Entry Point ---
# This block runs when you execute 'python src/app/main.py'
if __name__ == '__main__':
    app = create_app()
    # debug=True automatically reloads the server when you save changes
    app.run(debug=True)
