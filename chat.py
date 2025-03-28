import os

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context.
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know".

    <context>
    {context}
    </context>
    """


class ChatChain:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.chat_model = self.get_gemini_chat_model()
        self.chat_chain = self.initialize_chat_chain()

    def get_gemini_chat_model(self):
        """
        Initialize chat model with Google Gemini.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def initialize_chat_chain(self):
        """
        Initialize LangChain chain with chat model and templates.
        """
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return create_stuff_documents_chain(self.chat_model, question_answering_prompt)

    def query_rag(self, query):
        """
        Retrieve relevant documents from vector database.
        """
        relevant_docs = self.vector_store.similarity_search(query, k=20)
        reply = self.chat_chain.invoke(
            {
                "context": relevant_docs,
                "messages": [HumanMessage(content=query)],
            }
        )
        return reply

    def gradio_predict(self, message, history):
        """
        Gradio-friendly chat response function.
        """
        history_langchain_format = []
        for msg in history:
            if msg["role"] == "user":
                history_langchain_format.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_langchain_format.append(AIMessage(content=msg["content"]))

        history_langchain_format.append(HumanMessage(content=message))
        response = self.query_rag(message)
        return response
