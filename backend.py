
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os


def load_documents(directory_path):
    documents = []

    try:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        for file in os.listdir(directory_path):
            print("Name: ", file)
            file_path = os.path.join(directory_path, file)
            print("File: ",file_path)
            if file.endswith((".pdf", ".docx", ".doc", ".txt", ".csv", ".xlsx", ".xls", ".pptx")):
                loaders = {
                    ".pdf": PyPDFLoader,
                    ".docx": Docx2txtLoader,
                    ".doc": Docx2txtLoader,
                    ".txt": TextLoader,
                    ".csv": CSVLoader,
                    ".xlsx": UnstructuredExcelLoader,
                    ".xls": UnstructuredExcelLoader,
                    ".pptx": UnstructuredPowerPointLoader
                }
                ext = os.path.splitext(file)[-1].lower()
                print(ext)
                loader = loaders.get(ext)
                print(loader)
                if loader:
                    documents.extend(loader(file_path).load())
    except Exception as e:
        print(f"Error loading documents: {e}")
        documents = []
    return documents

def split_documents(documents, chunk_size=2000, chunk_overlap=20):
    try:
        if not documents:
            raise ValueError("No documents provided for splitting.")
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        return text_splitter.split_documents(documents)
    
    except ValueError as ve:
        print(f"Error in split_documents: {ve}")
        return [] 
                    
def user_model(vectordb, llm):
    try:
        memory = ConversationBufferMemory(memory_key="chat_history",
                                          return_messages=True,
                                          output_key='answer')

        model = ConversationalRetrievalChain.from_llm(llm,
                                                      retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
                                                      return_source_documents=True,
                                                      memory=memory,
                                                      verbose=False)

        return model
    
    except ValueError as ve:
        print(f"Error in user_model: {ve}")
        return None