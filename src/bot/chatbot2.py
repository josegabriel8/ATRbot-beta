import os
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from openai import OpenAIError
from dotenv import load_dotenv
from src.bot.rag_pipeline import retriever
from telegram import Update
from telegram.ext import Application, MessageHandler, CallbackContext, filters

# Load environment variables
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PromptTemplate para empoderar RAG + empatía
template = """
Eres un asistente virtual empático y confiable diseñado para ayudar a pacientes que se han sometido o se someterán a una cirugía de artroplastia de rodilla.
Tu objetivo es responder de forma clara, respetuosa y tranquilizadora, basándote en la información contenida en los documentos que tienes como referencia.
Si la pregunta se relaciona con el proceso quirúrgico, la recuperación, el dolor, la fisioterapia o las emociones asociadas al procedimiento, intenta brindar orientación general y apoyo,
incluso si no cuentas con información específica exacta. Sé honesto si no tienes una respuesta precisa, pero ofrece siempre una alternativa útil, como consultar al equipo médico.
Si detectas ansiedad, miedo o dolor en el usuario, valida sus emociones y responde con calidez, sin sonar automatizado.
Nunca inventes datos médicos ni ofrezcas diagnósticos personalizados.
Si no puedes responder con certeza, puedes decir: 'No tengo información específica sobre eso, pero te recomiendo hablar con tu equipo médico para recibir orientación precisa.'

Usa la siguiente información de contexto para responder la pregunta:
{context}

Pregunta: {question}
Respuesta útil:"""

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Métricas de evaluación en tiempo real
class ChatbotEvaluator:
    def __init__(self):
        self.total = 0
        self.abstentions = 0

    def record(self, response: str):
        self.total += 1
        if "No tengo información específica" in response:
            self.abstentions += 1

    def stats(self):
        if self.total == 0:
            return {"tasa_abstencion": 0.0}
        return {"tasa_abstencion": self.abstentions / self.total}

evaluator = ChatbotEvaluator()

def create_chatbot():
    """
    Configura el chatbot con LLaMA 3 vía Groq, RAG pipeline y memoria conversacional.
    """
    # Ensure we're using the correct API key (not from cached environment)
    api_key = os.environ.get("GROQ_API_KEY")
    
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-70b-8192"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

    # Crear el chain con configuración personalizada
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )
    return convo_chain

def generate_response(chain, user_input: str):
    """
    Genera una respuesta usando el chain conversacional con manejo de errores.
    """
    try:
        output = chain.invoke({"question": user_input})
        result = output.get("answer", "")
        source_documents = output.get("source_documents", [])
    except Exception as e:
        if "Invalid API Key" in str(e) or "401" in str(e):
            logger.error("Error: Groq API Key inválida o expirada")
            result = (
                "🔧 **Estado del Bot: Modo de Mantenimiento**\n\n"
                "Actualmente estamos experimentando problemas técnicos con nuestro servicio de IA. "
                "El sistema de recuperación de documentos está funcionando correctamente, pero la generación de respuestas está temporalmente deshabilitada.\n\n"
                "**Mientras tanto, puedes:**\n"
                "• Contactar a tu equipo médico directamente\n"
                "• Revisar la documentación proporcionada por tu clínica\n"
                "• Intentar nuevamente en unos minutos\n\n"
                "Lamentamos las molestias. Estamos trabajando para resolver este problema."
            )
        else:
            logger.error(f"Unexpected error: {e}", exc_info=e)
            result = (
                "Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo."
            )
        source_documents = []
    
    evaluator.record(result)
    return result, source_documents

async def handle_message(update: Update, context: CallbackContext):
    """Handler asíncrono para mensajes de Telegram"""
    user_input = update.message.text
    result, sources = generate_response(context.bot_data["chain"], user_input)
    await update.message.reply_text(result)
    # (Opcional) enviar fuentes:
    # for doc in sources:
    #     await update.message.reply_text(f"Fuente: {doc.metadata.get('source')}")

def main():
    """Función principal para iniciar el bot"""
    # Crea el chatbot y guárdalo en bot_data para usar en handlers
    chain = create_chatbot()

    # Inicializa el bot de Telegram (v20+ syntax)
    application = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()
    application.bot_data["chain"] = chain

    # Handler de mensajes de texto
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))    # Arranca en polling (local)
    logger.info("🤖 Chatbot listo. Iniciando polling...")
    application.run_polling()
    
    # Al finalizar, mostrar métricas
    stats = evaluator.stats()
    logger.info(f"Tasa de abstención: {stats['tasa_abstencion']*100:.1f}%")

def run_chatbot():
    """Función wrapper para ejecutar el bot desde app2.py"""
    main()

if __name__ == "__main__":
    main()
# This code is a Telegram bot that uses a conversational retrieval chain to answer questions about knee arthroplasty surgery.
