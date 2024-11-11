from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API = os.getenv('GROQ_API')
GOOGLE_API = os.getenv('GOOGLE_API')

if not GROQ_API or not GOOGLE_API:
    raise EnvironmentError("Missing GROQ or Google API key in .env file")
