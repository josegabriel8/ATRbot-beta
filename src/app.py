import os
import sys
sys.path.append("c:\\Users\\friva\\Documents\\ATRbot-beta")
import time
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from src.bot.chatbot import create_chatbot, generate_response

# Load environment variables
load_dotenv()

# Initialize the chatbot
qa_chain = create_chatbot()

# Dynamically load the TELEGRAM_TOKEN from the .env file
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN is not set in the environment variables.")

# Update the TELEGRAM_API_URL to include the /bot prefix
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
print(f"TELEGRAM_API_URL: {TELEGRAM_API_URL}")

def get_updates(offset=None):
    """Fetch new messages from Telegram."""
    url = f"{TELEGRAM_API_URL}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    try:
        response = requests.get(url, params=params)
        response_data = response.json()

        print(f"Fetching updates with offset: {offset}")
        print(f"Requesting URL: {url}")
        print(f"Response: {response_data}")
        print(f"Request headers: {response.request.headers}")
        print(f"Request body: {response.request.body}")

        if not response_data.get("ok"):
            print(f"Error from Telegram API: {response_data.get('description')}")

        return response_data
    except Exception as e:
        print(f"Exception occurred while fetching updates: {e}")
        return {"ok": False, "result": []}

def send_message(chat_id, text):
    """Send a message to a Telegram user."""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

def save_conversation(conversation):
    """Save the conversation to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), "..", "conversations", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=4)

    print(f"Conversation saved to {filepath}")

def main():
    """Run the Telegram bot locally using polling."""
    print("🤖 Bot is running locally. Press Ctrl+C to stop.")
    offset = None
    conversation = []  # Track the conversation
    user_input = ""  # Initialize user_input with a default value

    while True:
        updates = get_updates(offset)

        for update in updates.get("result", []):
            offset = update["update_id"] + 1
            message = update.get("message", {})
            chat_id = message.get("chat", {}).get("id")
            user_input = message.get("text", "")

            if chat_id and user_input:
                # Log user message
                conversation.append({"user": user_input})

                # Generate a response using the chatbot
                result, _ = generate_response(qa_chain, user_input)

                # Log bot response
                conversation.append({"bot": result})

                # Send the response back to the user
                send_message(chat_id, result)

        time.sleep(1)

        # Save the conversation when the user ends the chat
        if user_input.lower() == "salir":
            save_conversation(conversation)
            break

if __name__ == "__main__":
    main()
