import streamlit as st # to create the web application's user interface.
from PyPDF2 import PdfReader #to read text content from PDF files.
from langchain.text_splitter import RecursiveCharacterTextSplitter #splits large documents into smaller pieces or "chunks."
import os #to get the API key from environment variables.
from langchain_google_genai import GoogleGenerativeAIEmbeddings #for creating embeddings (numerical representations) of text using a Google Generative AI model.
import google.generativeai as genai #core library for interacting with Google's Generative AI models.
from langchain.vectorstores import FAISS #an efficient library for similarity search, which will be used to store and search the text embeddings.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain # Imports a pre-built chain from LangChain that is designed for question-answering tasks.
from langchain.prompts import PromptTemplate #for creating custom prompts for the language model.
from dotenv import load_dotenv #to load environment variables from a .env file.
import time #used here to pause the application for a moment to make the progress bar updates more visible.

load_dotenv() #This function looks for a file named .env and loads any key-value pairs inside it as environment variables.
os.getenv("GOOGLE_API_KEY") # Retrieves the value of the GOOGLE_API_KEY environment variable. This is a common and secure way to handle API keys.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Configures the Google Generative AI library with the retrieved API key, authenticating your application to use the models.



def get_pdf_text(pdf_docs):
    """This function takes a list of PDF file objects (pdf_docs).
It initializes an empty string text.
It loops through each pdf file in the list. For each file, it creates a PdfReader object and then iterates through every page.
The extract_text() method reads the text from each page, and this text is added to the main text string.
Finally, the function returns a single, long string containing all the text from all the uploaded PDFs."""
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text


def get_text_chunks(text):
    """This function takes the large text string.
It initializes a RecursiveCharacterTextSplitter.
chunk_size=10000: This specifies that each text chunk should be a maximum of 10,000 characters long. 
chunk_overlap=1000: This ensures that consecutive chunks share 1,000 characters. 
The split_text() method divides the input text into a list of these smaller chunks, which the function then returns."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """This function takes the list of text chunks.
GoogleGenerativeAIEmbeddings creates an embeddings model. Embeddings are numerical representations of text that capture its semantic meaning.
FAISS.from_texts uses the embeddings model to convert each text_chunk into a vector (a list of numbers) and stores them in a vector store.
vector_store.save_local("faiss_index") creates a vector store which is saved locally on your machine. This allows the application to load the index later without having to re-process the PDFs every time."""
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():
    """This function sets up the LLM (gemini-pro) for the question-answering part.
prompt_template defines the system prompt that guides the LLM's behavior. 
It tells the model to answer the question based only on the provided context and to state "answer is not available in the context" if it cannot find the information.
ChatGoogleGenerativeAI Initializes the Gemini Pro chat model from Google. 
temperature=0.3 makes the model's responses more focused and less creative.
PromptTemplate Creates a template for the prompt, specifying that it will receive two inputs: context and question.
load_qa_chain is a pre-built pipeline that combines the model and the prompt. 
The chain_type="stuff" means it will take all the retrieved context documents and "stuff" them into a single prompt for the model to process."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain




def user_input(user_question):
    """This function is called every time a user submits a question.
FAISS.load_local loads the previously saved vector store from the local directory.
new_db.similarity_search(user_question) converts the user's question into an embedding, 
and the vector store is searched for the most semantically similar text chunks from the original PDFs. 
The most relevant chunks are returned in a list called docs.
chain = get_conversational_chain initializes the LLM chain.
response = chain(...): The retrieved docs and the user_question are passed to the chain. 
The model generates an answer based on the provided context.
return response["output_text"] This function returns the final generated answer as a string"""
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]

def main():
    st.set_page_config("Chat PDF") #Sets the title of the browser tab.
    st.header("Chat with PDF using Gemini💁") #Displays a main header on the web page.

    # Initialize chat history
    #st.session_state: This is a dictionary-like object that persists data across Streamlit reruns. 
    #This line checks if a list called messages exists in the session state. 
    #If not, it creates it, initializing it as an empty list. This list will store the chat history.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    #This loop runs every time the app reruns. 
    #It iterates through the messages in the session state and displays them in the chat window, 
    #ensuring that the conversation history is preserved and visible.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_question = st.chat_input("Ask a Question from the PDF Files") #This creates the text input box at the bottom of the page, styled for chat. When a user submits text, the value is stored in user_question.

    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question}) #The user's message is added to the messages list.
        # Display user message in chat message container
        with st.chat_message("user"): #This creates a chat bubble styled for the user. 
            st.markdown(user_question) #displays the user's message inside it.

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = user_input(user_question)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar: # defines elements that will be placed in a sidebar on the left side of the app.
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True) # Creates a button to upload one or more PDF files.
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Get PDF text
                    status_text.text("Extracting text from PDFs...")
                    raw_text = get_pdf_text(pdf_docs)
                    progress_bar.progress(33)
                    time.sleep(1)

                    # Step 2: Get text chunks
                    status_text.text("Splitting text into chunks...")
                    text_chunks = get_text_chunks(raw_text)
                    progress_bar.progress(66)
                    time.sleep(1)

                    # Step 3: Create vector store
                    status_text.text("Creating vector store...")
                    get_vector_store(text_chunks)
                    progress_bar.progress(100)
                    time.sleep(1)

                    status_text.text("Done!")
                    st.success("Vector store created successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__": #standard Python syntax that ensures the main() function is called only when the script is executed directly (and not when it's imported as a module).
    main()