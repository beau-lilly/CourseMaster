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

## LLM Configuration (OpenRouter)

The app now prefers [OpenRouter](https://openrouter.ai/) for LLM calls.

1. Copy `.env.example` to `.env`.
2. Set `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL` and `OPENROUTER_BASE_URL` if you want a different model/endpoint).
3. (Optional) Add `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME` if you want to forward attribution headers that OpenRouter recommends.
4. If you want to fall back to direct OpenAI, set `OPENAI_API_KEY` instead.

Notes:
- The app now loads `.env` automatically (via `python-dotenv`), so you do not need to export variables manually.
- If no key is present, the app automatically uses a stubbed LLM so tests stay offline-safe.
