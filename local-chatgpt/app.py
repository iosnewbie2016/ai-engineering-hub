import chainlit as cl
import ollama
import pypdf
import time
import asyncio

MODEL_NAME = "llama3.2"

@cl.on_chat_start
async def start_chat():
    """
    Initializes the chat session when a user starts a new conversation.
    Sets up the system prompt and sends a welcome message.
    """
    # Initialize the interaction history with a system prompt.
    cl.user_session.set(
        "interaction",
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
    )

    # Create an empty message to stream tokens into.
    msg = cl.Message(content="")

    # Define the initial welcome message.
    start_message = f"Hello, I'm your 100% local alternative to ChatGPT running on {MODEL_NAME}. How can I help you today?"

    # Stream the welcome message token by token.
    for token in start_message:
        await msg.stream_token(token)

    # Send the complete welcome message to the user.
    await msg.send()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    print("Reading from pdf file...", pdf_path)
    # Opens the PDF file in read-binary mode.
    with open(pdf_path, "rb") as f:
        # Creates a PDF reader object.
        reader = pypdf.PdfReader(f)
        # Extracts text from each page and joins them with newlines.
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    print("Text Extraction from PDF is complete")
    return text

async def stream_ollama_response(messages):
    """Runs Ollama chat in a separate thread to avoid async generator issues."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama.chat(model=MODEL_NAME, messages=messages, stream=True))


@cl.step(type="tool")
async def tool(input_message, image=None, pdf_text=None):
    """
    Handles the core logic of processing user input, interacting with the language model, 
    and managing the conversation history.

    Args:
        input_message (str): The user's text message.
        image (list, optional): A list of image paths. Defaults to None.
        pdf_text (str, optional): Extracted text from a PDF. Defaults to None.

    Returns:
        dict: The response from the language model.
    """

    start_time = time.time()
    # Retrieve the current interaction history from the user session.
    interaction = cl.user_session.get("interaction")

    # Prepare the user message, adding PDF text if available.
    user_message = input_message
    print("User Input message:", user_message)
    if pdf_text:
        user_message += f"\n\nHere is the extracted text from the uploaded document:\n{pdf_text}"

    # Append the user's message to the interaction history.
    interaction.append({"role": "user", "content": user_message, "images": image} if image else {"role": "user", "content": user_message})
    # if image:
    #     interaction.append({"role": "user", "content": user_message, "images": image})
    # else:
    #     interaction.append({"role": "user", "content": user_message})
    
    # Send the interaction to the Ollama language model.
    # Commented out llama3.2-vision, as the doc state it is not compatible with ollama.chat at the moment
    # response = ollama.chat(model="llama3.2-vision", messages=interaction)
    # response = ollama.chat(model=MODEL_NAME, messages=interaction)
    
    # # Append the model's response to the interaction history.
    # interaction.append({"role": "assistant", "content": response.message.content})

# Create an empty message for streaming response
    msg = cl.Message(content="")

    try:
        # Call Ollama and stream response
        response = await stream_ollama_response(interaction)
        for chunk in response:
            await msg.stream_token(chunk["message"]["content"])  # Stream token

        # Save final response to interaction history
        interaction.append({"role": "assistant", "content": msg.content})
        await msg.send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

    end_time=time.time()
    print("Time taken for Ollama processing:", end_time-start_time)
    
    # return response

@cl.on_message 
async def main(message: cl.Message):
    """
    Handles incoming messages from the user.
    Processes images and PDF attachments, then calls the 'tool' function to interact with the language model.

    Args:
        message (cl.Message): The incoming message from the user.
    """

    overall_start_time = time.time()
    # Filter images and pdf files from the elements attached to the message
    images = [file for file in message.elements if "image" in file.mime]
    pdf_files = [file for file in message.elements if file.mime == "application/pdf"]

    # pdf_text = None
    # # If pdf file are present, process it
    # if pdf_files:
    #     pdf_text = extract_text_from_pdf(pdf_files[0].path)

    # If pdf file are present, process it
    pdf_text = extract_text_from_pdf(pdf_files[0].path) if pdf_files else None

    try:
        # Call the tool function, with the image files or not
        if images:
            # tool_res = await tool(message.content, [i.path for i in images], pdf_text)
            await asyncio.wait_for(tool(message.content, [i.path for i in images], pdf_text), timeout=300)
        else:
            # tool_res = await tool(message.content, pdf_text=pdf_text)
            await asyncio.wait_for( tool(message.content, pdf_text=pdf_text), timeout=300)
    except asyncio.TimeoutError:
        await cl.Message(content="The model took too long to respond. Please try again later.").send()
        return

    # # Prepare a message to send back to the user
    # msg = cl.Message(content="")

    # # Stream the model's response token by token
    # for token in tool_res.message.content:
    #     await msg.stream_token(token)

    # # Send the full response to the user
    # await msg.send()

    overall_end_time = time.time()
    print("Overall time taken for the entire process:", overall_end_time-overall_start_time)
