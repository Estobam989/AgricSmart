
import os

# --- 1. Environment Variables ---
# For local testing, ensure these are set in your environment or a .env file
# For deployment with Streamlit, consider using st.secrets
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
MODEL_NAME = os.environ.get('MODEL_NAME', 'llama-3.3-70b-versatile')
TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.3))
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 4096))
