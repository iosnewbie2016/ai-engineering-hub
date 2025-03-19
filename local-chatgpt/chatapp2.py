import pytesseract
from PIL import Image
import pdfplumber
import faiss  # ✅ Changed from ChromaDB to FAISS
import chainlit as cl
import ollama
import time
import asyncio
from io import BytesIO
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt')

# Load the embedding model (same as before)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = "llama3.2"

# ✅ Initialize FAISS (Replacing ChromaDB)
dimension = 384  # Embedding size for MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)  # L2 Distance based FAISS index
metadata_store = {}  # Dictionary to store text metadata (since FAISS does not store metadata)

@cl.on_chat_start
async def start_chat():
    """Initializes the chat session and sends the welcome message."""
    cl.user_session.set(
        "interaction",
        [{"role": "system", "content": "You are a helpful assistant."}]
    )
    
    start_message = f"Hello, I'm your 100% local alternative to ChatGPT running on {MODEL_NAME}. How can I help you today?"
    msg = cl.Message(content="")
    
    for token in start_message:
        await msg.stream_token(token)
    
    await msg.send()
    
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF using pdfplumber, including OCR for images.
    """
    print(f"Reading from pdf file: {pdf_path}")
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text from page
            page_text = page.extract_text() if page.extract_text() else ""
            text += page_text
            print(f"Extracted text from page {page_num + 1}")
            
            # Handle images on pages (OCR processing)
            print(f"Number of images on page {page_num + 1}: {len(page.images)}")
            for img in page.images:
                print(img)
                # Extract image from the stream
                image_data = img['stream'].get_data()  # Get the raw image data
                image = Image.open(BytesIO(image_data))  # Open image from byte stream
                
                # Use OCR to extract text from image
                image_text = pytesseract.image_to_string(image)
                text += "\n" + image_text  # Append OCR result to extracted text
                print(f"Extracted text from image on page {page_num + 1}")
    
    print("Text extraction complete")
    return text

def semantic_chunking(text, max_tokens=300):
    """
    Splits text into chunks based on sentence boundaries, ensuring each chunk is meaningful.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())  # Count words
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add the last chunk

    return chunks

def add_pdf_to_db(pdf_path):
    """Extracts text from a PDF and stores embeddings into FAISS."""
    global index, metadata_store  # Use global variables

    text = extract_text_from_pdf(pdf_path)
    chunks = semantic_chunking(text, max_tokens=300)
    embeddings = embedding_model.encode(chunks)

    # ✅ Add to FAISS (Replacing ChromaDB)
    index.add(np.array(embeddings, dtype=np.float32))

    # ✅ Store metadata separately
    for i, chunk in enumerate(chunks):
        metadata_store[i] = {"text": chunk, "source": pdf_path}

def query_faiss(query, relevance_threshold=0.75):
    """Queries FAISS for similar embeddings and returns relevant text."""
    faiss_query_start_time = time.time()
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    
    # ✅ Perform similarity search in FAISS
    D, I = index.search(query_embedding, 3)  # Get top 3 results

    retrieved_chunks = []
    for i, score in zip(I[0], D[0]):
        if i != -1 and score >= relevance_threshold:
            retrieved_chunks.append(metadata_store[i]["text"])

    if not retrieved_chunks:
        print("No highly relevant documents found.")
        return []
    
    faiss_end_time = time.time()
    print(f"Time taken for FAISS query: {faiss_end_time - faiss_query_start_time} seconds")

    return retrieved_chunks  # Returns a list of relevant text chunks

async def stream_ollama_response(messages):
    """Runs Ollama chat in a separate thread to avoid async generator issues."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama.chat(model=MODEL_NAME, messages=messages, stream=True))

@cl.step(type="tool")
async def process_file_upload(file_path):
    """Processes the uploaded PDF and stores embeddings in FAISS."""
    print("Starting file upload process...")
    add_pdf_to_db(file_path)
    await cl.Message(content="File processed and stored in FAISS. You can now chat with the assistant.").send()

@cl.step(type="tool")
async def tool(input_message, image=None):
    """Handles user input and retrieves relevant context from FAISS."""
    start_time = time.time()
    
    interaction = cl.user_session.get("interaction")
    user_message = input_message

    # ✅ Query FAISS instead of ChromaDB
    relevant_text = query_faiss(input_message)

    if relevant_text:
        context = "\n\n".join(relevant_text)
        user_message = (
            f"You are an AI assistant with access to relevant information from documents.\n\n"
            f"**Context:**\n{context}\n\n"
            f"Based on the above context, answer the following question accurately.\n\n"
            f"**Question:** {input_message}\n"
            f"**Answer:**"
        )
    else:
        user_message = (
            f"You are an AI assistant, but no relevant information was found in the documents.\n"
            f"Do not guess. If you do not know the answer, say so.\n\n"
            f"**Question:** {input_message}\n"
            f"**Answer:**"
        )

    interaction.append({"role": "user", "content": user_message, "images": image} if image else {"role": "user", "content": user_message})

    msg = cl.Message(content="")
    
    try:
        response = await stream_ollama_response(interaction)
        for chunk in response:
            await msg.stream_token(chunk["message"]["content"])

        interaction.append({"role": "assistant", "content": msg.content})
        await msg.send()
    
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

    end_time = time.time()
    print(f"Time taken for Ollama processing: {end_time - start_time} seconds")

@cl.on_message
async def main(message: cl.Message):
    """Processes user messages and interacts with FAISS."""
    overall_start_time = time.time()
    
    pdf_files = [file for file in message.elements if file.mime == "application/pdf"]

    if pdf_files:
        await process_file_upload(pdf_files[0].path)

    try:
        await asyncio.wait_for(tool(message.content), timeout=300)
    except asyncio.TimeoutError:
        await cl.Message(content="The model took too long to respond. Please try again later.").send()
        return

    overall_end_time = time.time()
    print(f"Overall time taken for the entire process: {overall_end_time - overall_start_time} seconds")

def view_faiss_contents():
    """
    Retrieves and prints contents from the FAISS index.
    """
    global faiss_index, stored_texts  # Ensure access to the stored text and metadata dictionary
    
    if not stored_texts:
        print("FAISS index is empty.")
        return

    print("Stored text chunks in FAISS:")
    for idx, text in stored_texts.items():
        print(f"ID {idx}: {text}")

# Run the Chainlit application
# cl.run()

