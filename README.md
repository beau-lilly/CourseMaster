# Project Title: CourseMaster

## What it Does

The purpose of the CourseMaster project is to help students study for exams more efficiently by centering their studying around relevant practice problems and how they relate to the course material. The application does this using a RAG system: the student uploads the course material for an exam scope, and enters the relevant practice problems from homeworks, practice exams, and textbook questions. The system splits the documents into chunks, and embeds both the chunks and problems. Upon entering a new problem, the system uses ChromaDB to conduct a vector search to identify the relevant chunks from the course material in the problem's exam scope, and enables the student to ask questions about the problem to an LLM, which has the relevant chunks and problem prepended to its context. The student may ask multiple questions related to a problem, each of which persists in an SQLite database. 


## Quick Start

1. Install dependencies:
pip install -r requirements.txt
   
2. Configure .env
Copy `.env.example` to `.env` and add your API keys in the placeholder(OpenRouter recommended)
cp .env.example .env

3. Run the Application:
python -m src.app.main

4. Access:
Open http://127.0.0.1:5000 in your browser


## Video Links

- [User Demo (Local File)](videos/user_demo.mp4)
    - https://drive.google.com/file/d/1F92zR_r7mM3U2Fb2Ef4yUyqF03LX-AQ6/view?usp=drive_link
- [Technical Walkthrough (Local File)](videos/technical_demo.mp4)
    - https://drive.google.com/file/d/19yvM_a5_RRiv5ngzUN8J7MJU4ov6q7-Q/view?usp=drive_link


## Evaluation

Evaluated the system's performance using 4 different prompt styles with the goal of assessing the trade offs between response latency, verbiosity, and ultimately, usefulness to the student. (Each prompt style is available in the website under a dropdown that displays when asking a question.)

### Experiment Setup
- **Task:** For each different problem, I asked the same question: *"Please explain how I should approach this problem."*

- **Styles Tested:**
    1. *Minimal:* Brief, direct answers based only on context.
    2. *Explanatory:* Expert tutor persona, cites specific chunks.
    3. *Tutoring:* Provides hints and leading questions rather than direct answers.
    4. *Similarity:* Analyzes the problem's relevance to the pulled chunks.

### Findings

Minimal
- Avg Latency: 3.32s
- Avg Length: 94 words

Explanatory
- Avg Latency: 9.88s
- Avg Length: 294 words

Tutoring
- Avg Latency: 2.79s
- Avg Length: 101 words

Similarity
- Avg Latency: 12.78s
- Avg Length: 496 words


Usefulness Analysis (seen in `notebooks/experiments.ipynb`)
- Miminal: used to check understanding quickly and get instantaneous feedback if one already has an answer and is confident in their understanding. 
- Explanatory: If a student is completely lost on a subject, or it is one of the first problems they have seen of this type, the explanatory response gives a good formula for how to solve these kinds of problems. 
- Tutoring: Once a student has seen a few of the same types of problems (covering the same content or using the same techniques), they can use the tutoring prompt if they are almost to the answer but are missing a small amount of vital information. The tutoring prompt will lead them back through the process of discovering the answer by explaining where in the notes the relevant content is, presumably leading the student to find the vital piece of missing information on their own. 
- Similarity: This proved to be the least helpful, but I think if I used semantic chunking rather than a fixed character limit, and used a threshold matching algorithm rather than top k (this combination would ensure highly relevant information to the Problem), this would result in this prompt being much more useful to the student. 


Full data in `notebooks/experiments.ipynb`


## Individual Contributions

- Solo project, details about AI use comprehensively laid out in `ATTRIBUTION.md`