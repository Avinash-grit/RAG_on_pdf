import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from ctransformers import AutoModelForCausalLM,AutoConfig

config = AutoConfig.from_pretrained("llama-2-7b-chat.ggmlv3.q8_0.bin")
# Explicitly set the max_seq_len
config.max_seq_len = 4096
config.max_answer_len= 1024


DB_FAISS_PATH = 'vectorstore/db_faiss'
prompt_file_path = 'aim_desc_.txt'
# Define prompts as a global variable
prompts = []

# Load the prompts from the text file and assign them to the global variable
with open(prompt_file_path, 'r') as f:
    prompts = [line.strip() for line in f.readlines()]


# Load the locally downloaded model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=2048,
        temperature=0.2,
        max_length=2048,
        top_p = .95,
        top_k = 40
        
    )
    return llm

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}


def main():
    st.title("Chat with your PDF file.")
    st.markdown("<h3 style='text-align: center; color: white;'>Built for Ford </a></h3>", unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload your Data", type="pdf")

    if uploaded_file:
        # Read the PDF file
        pdfReader = PdfReader(uploaded_file)
        documents = []
        for page in pdfReader.pages[:68]:  # Only take first 2 chapter
            page_text = page.extract_text()
            documents.append(Document(page_text))

        # Split the text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(documents)

        # Create the embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(text_chunks, embeddings)
        db.save_local(DB_FAISS_PATH)

        # Load the model and set up the retrieval chain
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        


        # Define a function to handle conversational chat
        def conversational_chat(query):
            max_length = 2048
            chat_history = [] # We don't need to keep track of chat history in this case
            answer = ""
            global prompts
            for prompt in prompts:
                prompt_query = f'{prompt}{query}'
                #prompt_query = "Please give me information on " + query
                chunks = [prompt_query[i:i+max_length] for i in range(0, len(prompt_query), max_length)]
                for chunk in chunks:
                    result = chain({"question": chunk, "chat_history": chat_history})
                    answer += result["answer"]
            return answer


        
        #def conversational_chat(query):
            #result = chain({"question": query, "chat_history": []})
            #return result["answer"]


        # Set up the container for user input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your pdf data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.write(output)


if __name__ == "__main__":
    main()
