import os
import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# Define the absolute path to your FAISS index
DB_FAISS_PATH = os.path.join(os.getcwd(), "vectorstore", "faiss_db")

# Define the custom prompt template with a clear separator
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Helpful answer:
###"""

# Function to set up a custom prompt
def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# Load Language Model using HuggingFacePipeline
def load_llm():
    # Load the Hugging Face pipeline for text generation
    text_gen_pipeline = pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-2.7B",
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available, otherwise CPU
        max_new_tokens=256,  # Set maximum number of tokens for the generated output
        temperature=0.9,  # Adjust creativity level
        pad_token_id=50256  # Avoid errors with token padding
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)

# Function to set up the QA bot
def qa_bot():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # Load FAISS database
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}!")

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Load language model
    llm = load_llm()

    # Set up the prompt
    qa_prompt = set_custom_prompt()

    # Set up the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True  # Ensure sources are included
    )
    return qa

# Chainlit handlers
@cl.on_chat_start
async def start():
    try:
        # Initialize the QA bot and store it in the user session
        chain = qa_bot()
        cl.user_session.set("chain", chain)
        await cl.Message(content="Hi, Welcome to CareConnect. What is your query?").send()
    except Exception as e:
        # Send error message if QA bot fails to load
        await cl.Message(content=f"Error loading the QA bot: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    conversation_context = message.content

    try:
        # Debug input
        print("Received query:", conversation_context)

        # Generate response using the QA chain
        result = chain({"query": conversation_context})
        print("Result from chain:", result)

        # Extract the answer
        answer = result.get("result", "Sorry, I couldn't find an answer.")
        if "###" in answer:
            answer = answer.split("###")[1].strip()  # Extract only the answer part

        # Handle sources
        sources = result.get("source_documents", [])
        formatted_sources = ""
        if sources:
            formatted_sources = "\n\n**Sources:**\n"
            for doc in sources:
                source_name = doc.metadata.get("source", "Unknown Source")
                page_number = doc.metadata.get("page", "N/A")
                formatted_sources += f"- {source_name} (Page {page_number})\n"
        else:
            formatted_sources = "\nNo sources found."

        # Combine answer and sources
        final_output = f"{answer}{formatted_sources}"

        # Debug final output
        print("Final output:", final_output)

        # Send response to the user
        await cl.Message(content=final_output).send()
    except Exception as e:
        # Send error message if something goes wrong during query handling
        print("Error occurred:", str(e))
        await cl.Message(content=f"An error occurred: {str(e)}").send()