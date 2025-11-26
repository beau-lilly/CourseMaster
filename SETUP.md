# Setup Instructions

## Prerequisites
- Python 3.x
- pip

## Installation
1. Clone the repository.
2. Create a virtual environment (recommended).
3. Install dependencies: `pip install -r requirements.txt`
4. Configure environment variables: Copy `.env.example` to `.env` and add your API keys.

## Running the Application

1. Navigate to the project root:
   ```bash
   cd course_master
   ```

2. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   # On Windows use: venv\Scripts\activate
   ```

3. Run the application module:
   ```bash
   python -m src.app.main
   ```

4. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000).