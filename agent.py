import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["LANGCHAIN_TRACING_V2"] = "true"

load_dotenv()

# You may not need to use tools for every query - the user may just want to chat!

SYSTEM_TEMPLATE = """
    You are a Chatbot for the Edgewood County Park and Natural Perserve.
 
    Please respond in an informative manner and format results in Markdown.
    Display within the response any relevant images retrieved from the tools. 
    Images can be determined from the search results by a .jpg extension in html links.
    """


class Agent:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.agent_executor = self.construct_agent()

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

        from langchain.tools.retriever import create_retriever_tool

    def initialize_retriever_tool(self):

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        llm = self.get_gemini_chat_model()

        tool = create_retriever_tool(
            retriever,
            "search_edgewood_website",
            """Searches and returns excerpts from the Friends of Edgewood website.  The Friends of Edgewood
                is a non-profit dedicated to restoring the Edgewood County Park and Nature Preserve in San Mateo County.
                The Friends of Edgewood website includes a comprehensive catalog of native plants, birds and wildlife that is found in the park.
                The website also includes descriptions of hiking trails, volunteer opportunities, and events."
            """,
        )
        tools = [tool]
        return tools

    def construct_agent(self):

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm = self.get_gemini_chat_model()
        tools = self.initialize_retriever_tool()
        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools)

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
        response = self.agent_executor.invoke(
            {"input": message, "chat_history": history_langchain_format}
        )
        return response["output"]
