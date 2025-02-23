import subprocess
import shutil
import os
import configparser
import sys
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import gradio as gr
import time
from langchain.memory import ConversationBufferMemory


def clone_git_repo(repo_url, target_dir):
    """
    Clone a Git repository to the specified directory.
    
    :param repo_url: The URL of the Git repository to clone.
    :param target_dir: The target directory where the repository will be cloned.
    """
    try:
        # Clean the target directory if it exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            print(f"Cleaned existing directory: {target_dir}")
        subprocess.run(["git", "clone", repo_url, target_dir], check=True)
        print(f"Repository cloned successfully to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")

def initialize_model(model_name=None):
    # Initialize the ChatOpenAI model with optional API key, base URL, and model name
    return ChatOpenAI(model_name=model_name)

def process_repo_to_vector_store(repo_dir, model_name):
    """
    Process the files in the cloned repository and convert them into vectors stored in Chroma.
    
    :param repo_dir: The directory of the cloned repository.
    :param openai_api_key: The API key for OpenAI embeddings.
    :return: Chroma vector store containing the document vectors.
    """
    # Load documents from the repository directory
    loader = DirectoryLoader(repo_dir, recursive=True)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and store them in Chroma
    embeddings = OllamaEmbeddings(model=model_name)
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return vector_store

def local_to_vector_store(model_name):
    if os.path.exists("./chroma_db") == False:
        sys.exit("No Chroma DB found, please run the script first")
    embeddings = OllamaEmbeddings(model=model_name)
    vector_store = Chroma(embedding_function = embeddings, persist_directory="./chroma_db")
    return vector_store


def init_environ():
    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config['api']['api_key']
    base_url = config['api']['base_url']
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['DASHSCOPE_API_KEY'] = api_key
    os.environ['OPENAI_BASE_URL'] = base_url


# while True:
#     input_text = input("Enter your question (or 'exit' to quit): ")
#     if input_text.lower() == 'exit':
#         print("Exiting the conversation.")
#         break
    
#     retrieved_docs = vector_store.similarity_search(input_text)
#     context = "\n".join([doc.page_content for doc in retrieved_docs])
#     # Invoke the chat chain with the input text and context
#     for chunk in chat_chain.stream({"input_text": input_text, "context": context}):
#         print(chunk.content, end="", flush=True)
#     print()  # 换行


def echo(message, history):
    retrieved_docs = vector_store.similarity_search(message)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    result = "result: "
    for chunk in chat_chain.stream({"input_text": message, "context": context}):
       result += chunk.content
       yield result

global vector_store
global chat_chain
global history

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    model_name = config['api']['model_name']
    init_environ()
    # Clone the Git repository before initializing the model
    git_repo_url = "https://github.com/yikousu/sms.git"  # Replace with your Git repository URL
    target_directory = "./cloned_repo"  # Replace with your target directory
    reload = input("Reload Repository? (y/n)")
    vector_store = None
    if reload.lower() == "y":
        clone_git_repo(git_repo_url, target_directory)
        # Process the repository files and convert them into vectors stored in Chroma
        vector_store = process_repo_to_vector_store(target_directory, model_name)
    else:
        vector_store = local_to_vector_store(model_name)

    # Initialize the model with optional API key, base URL, and model name
    model = initialize_model(model_name=model_name)
    # Define the prompt template with context from the vector store
    prompt_template = PromptTemplate(
        input_variables=["input_text", "context"],
        template="You are a helpful assistant. Use the following context to respond to the question: {context}\n\nQuestion: {input_text}"
    )
    # Create the RunnableSequence
    chat_chain = prompt_template | model


    # Retrieve relevant context from the vector store
    demo = gr.ChatInterface(fn=echo, type="messages", save_history=True, examples=["你能做什么？", "介绍一下这个项目有哪些controller", "介绍一下com.manger.sms.controller.User.AdminController 中 userLogin() 方法的时序图，并用Mermaid画出来"], title="Echo Bot")
    demo.launch()