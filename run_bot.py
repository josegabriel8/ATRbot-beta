import os
import logging
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Delete any cached environment variables
if "GROQ_API_KEY" in os.environ:
    del os.environ["GROQ_API_KEY"]

# Force reload environment variables
load_dotenv(override=True)

# Directly set API key to ensure it's correct

logger.info("🔑 Environment Variables:")
logger.info(f"GROQ_API_KEY: {os.environ.get('GROQ_API_KEY', 'NOT_FOUND')[:20]}...")
logger.info(f"TELEGRAM_TOKEN: {os.environ.get('TELEGRAM_TOKEN', 'NOT_FOUND')[:20]}...")

# Test the Groq API connection
try:
    logger.info("🤖 Testing Groq API connection...")
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-70b-8192"
    )
    start_time = time.time()
    response = llm.invoke("Hola, ¿cómo estás?")
    end_time = time.time()
    logger.info(f"✅ Groq API connection successful! (Response time: {end_time-start_time:.2f}s)")
    logger.info(f"Response: {response.content[:100]}...")
    
    # Now import and run the chatbot
    logger.info("🚀 Starting the chatbot with the correct API key...")
    from src.bot.chatbot2 import run_chatbot
    run_chatbot()
except Exception as e:
    logger.error(f"❌ Error: {e}")
