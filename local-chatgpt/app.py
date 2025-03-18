import chainlit as cl
import ollama
import pypdf

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
    start_message = "Hello, I'm your 100% local alternative to ChatGPT running on Llama3.2-Vision. How can I help you today?"

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
    return text

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

    # Retrieve the current interaction history from the user session.
    interaction = cl.user_session.get("interaction")

    # Prepare the user message, adding PDF text if available.
    user_message = input_message
    if pdf_text:
        user_message += f"\n\nHere is the extracted text from the uploaded document:\n{pdf_text}"

    # Append the user's message to the interaction history.
    if image:
        interaction.append({"role": "user", "content": user_message, "images": image})
    else:
        interaction.append({"role": "user", "content": user_message})
    
    # Send the interaction to the Ollama language model.
    # Commented out llama3.2-vision, as the doc state it is not compatible with ollama.chat at the moment
    # response = ollama.chat(model="llama3.2-vision", messages=interaction)
    response = ollama.chat(model="llama3.2", messages=interaction)
    
    # Append the model's response to the interaction history.
    interaction.append({"role": "assistant", "content": response.message.content})
    
    return response

@cl.on_message 
async def main(message: cl.Message):
    """
    Handles incoming messages from the user.
    Processes images and PDF attachments, then calls the 'tool' function to interact with the language model.

    Args:
        message (cl.Message): The incoming message from the user.
    """

    # Filter images and pdf files from the elements attached to the message
    images = [file for file in message.elements if "image" in file.mime]
    pdf_files = [file for file in message.elements if file.mime == "application/pdf"]

    pdf_text = None
    # If pdf file are present, process it
    if pdf_files:
        pdf_text = extract_text_from_pdf(pdf_files[0].path)

    # Call the tool function, with the image files or not
    if images:
        tool_res = await tool(message.content, [i.path for i in images], pdf_text)
    else:
        tool_res = await tool(message.content, pdf_text=pdf_text)

    # Prepare a message to send back to the user
    msg = cl.Message(content="")

    # Stream the model's response token by token
    for token in tool_res.message.content:
        await msg.stream_token(token)

    # Send the full response to the user
    await msg.send()
