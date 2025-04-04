import gradio as gr

from agent import Agent
from store import create_vector_store


vector_store = create_vector_store(test=False)
agent = Agent(vector_store)

title = "Edgewood Park Chatbot"
desc = """<p style="text-align: center;">I am a chatbot for <a href="https://friendsofedgewood.org/">Friends of Edgewood</a>.
        Please ask me about native plants, wildlife, hiking trails, and volunteer opportunities.</p>
        """
iface = gr.ChatInterface(
    fn=agent.gradio_predict,
    type="messages",
    autofocus=False,
    title=title,
    description=desc,
)
iface.launch()
