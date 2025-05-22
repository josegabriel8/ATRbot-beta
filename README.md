# ATRbot: Telegram Chatbot with Retrieval-Augmented Generation (RAG)

## Overview

ATRbot is a Telegram chatbot designed to assist users by providing contextually relevant answers to their queries. It integrates a Retrieval-Augmented Generation (RAG) pipeline to retrieve information from Spanish-language PDFs and uses LLaMA 3 for conversational AI. The bot also logs conversations in JSON files for future reference.

---

## Features

### 1. Retrieval-Augmented Generation (RAG) Pipeline

- Processes Spanish-language PDFs to extract and chunk text.
- Generates embeddings using the `sentence-transformers/LaBSE` model for multilingual support.
- Stores embeddings in a FAISS index for efficient similarity search.
- Retrieves relevant information from the FAISS index to answer user queries.

### 2. Chatbot Integration

- Built using LLaMA 3 for empathetic and accurate conversational responses.
- Integrated with the RAG pipeline to provide contextually relevant answers.

### 3. Telegram Bot

- Allows users to interact with the chatbot via Telegram.
- Uses polling for local testing to fetch and respond to user messages.
- Supports sending and receiving messages in real-time.

### 4. Conversation Logging

- Automatically saves each chat session in a JSON file.
- Includes both user inputs and bot responses for future analysis.

---

## Project Structure

```plaintext
ATRbot-beta/
├── data/                     # Spanish-language PDFs for the RAG pipeline
├── faiss_index/              # FAISS index files
├── src/                      # Source code
│   ├── app.py                # Main script for running the Telegram bot
│   ├── bot/                  # Bot-related modules
│   │   ├── chatbot.py        # Chatbot logic using LLaMA 3
│   │   ├── rag_pipeline.py   # RAG pipeline implementation
│   ├── ingestion/            # Scripts for data ingestion
│   ├── utils/                # Utility scripts
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- Telegram Bot Token (obtainable via BotFather on Telegram)

### 2. Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd ATRbot-beta
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the `.env` file:

   ```env
   TELEGRAM_TOKEN=<your-telegram-bot-token>
   ```

### 3. Running the Bot

1. Start the bot locally:

   ```bash
   python src/app.py
   ```

2. Interact with the bot via Telegram.

---

## How It Works

### 1. User Interaction

- Users send messages to the bot via Telegram.
- The bot fetches messages using the Telegram API and processes them.

### 2. Response Generation

- The chatbot uses the RAG pipeline to retrieve relevant information from the FAISS index.
- LLaMA 3 generates a conversational response based on the retrieved context.

### 3. Logging

- Each chat session is logged in a JSON file, including user inputs and bot responses.
- The session ends when the user sends the message "salir."

---

## Next Steps

- Deploy the bot to a server for continuous operation.
- Replace polling with webhooks for better scalability.
- Enhance the chatbot's conversational capabilities with additional prompt engineering.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for the RAG pipeline.
- [Sentence Transformers](https://www.sbert.net/) for multilingual embeddings.
- [Telegram Bot API](https://core.telegram.org/bots/api) for seamless integration.

---

## Contact

For questions or feedback, please contact the project maintainer.
