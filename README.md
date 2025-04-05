# Edgewood Park Chatbot

The Edgewood Park Chatbot is an Agentic workflow powered by a RAG system built from the [Friends of Edgewood](https://friendsofedgewood.org/) website, and is deployed on [HuggingFace Spaces](https://huggingface.co/spaces/random-tesseract/edgewood), free for all to try.

The *Friends of Edgewood* is all-volunteer, donor-funded, non-profit that supports the Edgewood County Park and Natural Preserve in San Mateo County, California.  Their website includes a field guide of plant species and wildlife found in the park, descriptions of
hiking trails, and volunteer opportunities.

The chatbot was built using LangChain, makes use of Google Gemini models and embeddings, and has a Gradio frontend.  Traces are are logged using LangSmith.  Firecrawl was used to crawl and process web data, which is stored in a Qdrant vector database.  

The chatbot uses an Agentic workflow to decide when to use RAG tools and makes use of chat history. Responses are formatted with images and links.

<p align="center">
<img src="images/agentic-peeling-bark.png">
</p>