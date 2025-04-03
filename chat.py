import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

SYSTEM_TEMPLATE = """
    You are a Chatbot for the Edgewood County Park and Natural Perserve.

    Answer the user's questions based on the below context.  The context consists of pages from the Edgewood County Park website.

    Format the entire response in Markdown.
    Include in the response any relevant images the context - images can be determined by a .jpg extension in html links.


    If the context doesn't contain any relevant information to the question, don't make something up 
    and just say "I don't know".

    <context>
    {context}
    </context>
    """


class ChatChainWithHistory:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.rag_chain = self.initialize_rag_chain()

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

    def initialize_question_answer_chain(self, llm):
        """
        Initialize LangChain chain with chat model and templates.
        """
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_stuff_documents_chain(llm, question_answering_prompt)

    def initialize_history_aware_retriever(self, llm, retriever):
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    def initialize_rag_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        llm = self.get_gemini_chat_model()
        question_answer_chain = self.initialize_question_answer_chain(llm)
        history_aware_retriever = self.initialize_history_aware_retriever(
            llm, retriever
        )
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

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
        response = self.rag_chain.invoke(
            {"input": message, "chat_history": history_langchain_format}
        )
        return response["answer"]
