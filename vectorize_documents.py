from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# Load the embedding model
embeddings = HuggingFaceEmbeddings()

# Specify the file path for the CSV data
data_file_path = r"C:\LLM_summarizer\data.csv"  # Replace with your actual CSV file name containing the shlokas

# Ensure the file exists
if not os.path.isfile(data_file_path):
    raise FileNotFoundError(f"The file '{data_file_path}' does not exist. Please check the path.")

# Load the CSV file
loader = CSVLoader(file_path=data_file_path)
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

# Create a vector database
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"  # Directory to save the vector database
)

print("Shlokas Vectorized")
