import os
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_chatbot():
    """
    Configures the chatbot with LLaMA 3 model and RAG pipeline.
    """
    # Initialize the LLaMA 3 model through Groq
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-70b-8192"
    )

    # Load the retriever from the RAG pipeline
    from src.bot.rag_pipeline import retriever

    # Create the RAG-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def generate_response(qa_chain, user_input):
    """
    Generates a response using the QA chain with prompt engineering.
    """
    # Define the prompt template for accurate and empathetic responses
    prompt_template = (
        "Eres un asistente virtual empático y preciso. Responde a las preguntas basándote únicamente en la información de tu base de datos. "
        "Si no tienes información suficiente, responde: 'Lo siento, pero eso se sale de mi base de conocimiento y no tengo cómo responder.' "
        "Si detectas que el usuario expresa dolor, miedo o ansiedad, responde de manera empática y tranquilizadora."
    )

    # Combine the prompt with the user input
    prompt = f"{prompt_template}\n\nPregunta del usuario: {user_input}"

    # Generate the response
    response = qa_chain.invoke(prompt)

    # Extract the result and source documents
    result = response['result']
    source_documents = response.get('source_documents', [])

    return result, source_documents

if __name__ == "__main__":
    # Create the chatbot
    qa_chain = create_chatbot()

    print("🤖 Chatbot listo para responder tus preguntas. Escribe 'salir' para terminar.")

    while True:
        # Get user input
        user_input = input("Tú: ")

        if user_input.lower() == "salir":
            print("🤖 Adiós. ¡Que tengas un buen día!")
            break

        # Generate a response
        result, sources = generate_response(qa_chain, user_input)

        # Display the response
        print(f"🤖: {result}")

        # Optionally display source documents
        for doc in sources:
            print(f"Fuente: {doc}")

