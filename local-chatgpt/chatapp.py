import pytesseract
from PIL import Image
import pdfplumber
import chromadb
import chainlit as cl
import ollama
import pypdf
import time
import asyncio
from io import BytesIO
import nltk

# Ensure NLTK sentence tokenizer is available
nltk.download('punkt')

# Initialize ChromaDB (Persistent DB)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("pdf_rag")

# Load the embedding model (using a lightweight model for text embedding)
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = "llama3.2" # "mistral-7b"

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
    """
    Extracts text from a PDF and stores embeddings into ChromaDB.
    """
    text = extract_text_from_pdf(pdf_path)
    # chunks = [text[i : i + 500] for i in range(0, len(text), 500)]  # Chunk the text into parts
    chunks = semantic_chunking(text, max_tokens=300)  # Improved chunking
    embeddings = embedding_model.encode(chunks).tolist()

    # Add to ChromaDB
    # for i, chunk in enumerate(chunks):
    #     collection.add(ids=[str(i)], embeddings=[embeddings[i]], metadatas=[{"text": chunk}])
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)], 
            embeddings=[embeddings[i]], 
            documents=[chunk],  # ✅ Store actual text here
            metadatas=[{"source": pdf_path}]  # Optional metadata
        )

def query_chromadb(query, relevance_threshold=0.75):
    """
    Queries ChromaDB for similar embeddings, filters results, and returns relevant text.
    """
    chroma_query_start_time = time.time()
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query]).tolist()
    
    # Perform similarity search
    results = collection.query(query_embeddings=query_embedding, n_results=3)  # Get top 3 results
    
    # Debug: print the structure of the results
    print("ChromaDB query results:", results)
    chroma_query_end_time = time.time()
    print(f"Time taken for ChromaDB query: {chroma_query_end_time - chroma_query_start_time} seconds")
    
    # Extract and return the relevant chunks if available
    # if "metadatas" in results and results["metadatas"]:
    #     return [metadata.get("text", "") for metadata in results["metadatas"][0]]
    # else:
    #     print("No 'metadatas' field found in ChromaDB query result.")
    #     return []

    # if "documents" in results and results["documents"]:
    #     return [doc for doc in results["documents"][0] if doc is not None]  # Remove any None values
    # else:
    #     print("No 'documents' field found in ChromaDB query result.")
    #     return []

    # Extract relevant text chunks and scores
    retrieved_chunks = []
    scores = results.get("distances", [[]])[0]  # Get similarity scores
    if "documents" in results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            if doc is not None and scores[i] >= relevance_threshold:  # Only keep high-relevance chunks
                retrieved_chunks.append(doc)

    # If no high-confidence results, return an empty list
    if not retrieved_chunks:
        print("No highly relevant documents found.")
        return []

    return retrieved_chunks  # Returns a list of relevant text chunks


async def stream_ollama_response(messages):
    """Runs Ollama chat in a separate thread to avoid async generator issues."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama.chat(model=MODEL_NAME, messages=messages, stream=True))

@cl.step(type="tool")
async def process_file_upload(file_path):
    """
    This step handles the file upload, processes the PDF, and saves embeddings to ChromaDB.
    """
    print("Starting file upload process...")
    add_pdf_to_db(file_path)
    await cl.Message(content="File processed and stored in ChromaDB. You can now chat with the assistant.").send()

@cl.step(type="tool")
async def tool(input_message, image=None):
    """
    Handles user input, interacting with the language model and conversation history.
    Processes PDF text and image files if provided.
    """
    start_time = time.time()
    
    interaction = cl.user_session.get("interaction")
    user_message = input_message
        
    # Query ChromaDB for relevant text chunks based on the user's query
    relevant_text = query_chromadb(input_message)
    
    # Add the relevant text from ChromaDB to the user message
    # if relevant_text:
    #     user_message += "\n\nHere are some relevant pieces of text from the document:\n" + "\n".join(relevant_text)

    # Construct a structured prompt to keep responses grounded
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
        # Call Ollama and stream response
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
    """
    Processes user messages with images and PDF uploads, then calls the 'tool' function to interact with the language model.
    """
    overall_start_time = time.time()
    
    images = [file for file in message.elements if "image" in file.mime]
    pdf_files = [file for file in message.elements if file.mime == "application/pdf"]
    
    # If there is a PDF, we will upload and process it separately
    if pdf_files:
        # We can separate file upload and chat in a modular way
        await process_file_upload(pdf_files[0].path)
    
    try:
        if images:
            await asyncio.wait_for(tool(message.content, [i.path for i in images]), timeout=300)
        else:
            await asyncio.wait_for(tool(message.content), timeout=300)
    except asyncio.TimeoutError:
        await cl.Message(content="The model took too long to respond. Please try again later.").send()
        return

    overall_end_time = time.time()
    print(f"Overall time taken for the entire process: {overall_end_time - overall_start_time} seconds")

# Example function to view contents in ChromaDB
def view_chromadb_contents():
    """
    Retrieves and prints contents from ChromaDB.
    """
    results = collection.get()
    for item in results['metadatas']:
        print(item["text"])

# Run the Chainlit application
# cl.run()

