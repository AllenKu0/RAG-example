import gradio as gr
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("http://localhost:6333")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def chat(text, chatbox):
    chatbox.append([text, ""])
    search_results = client.search(
        collection_name="Health_Tips",
        query_vector=model.encode([text])[0],
        limit=1
    )
    chatbox[-1][1] = search_results[0].payload["document"]
    return "", chatbox

with gr.Blocks() as app:
    tb = gr.Textbox(lines=1, placeholder="Enter your text here")
    submit = gr.Button(value="Submit")
    chatbox = gr.Chatbot(label="Chatbot")

    submit.click(chat, inputs=[tb, chatbox], outputs=[tb, chatbox])

if __name__ == "__main__":
    app.launch()