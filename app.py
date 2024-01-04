from flask import Flask, render_template, request, jsonify ,session,redirect,url_for
import os
from PyPDF2 import PdfReader
# from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain,RetrievalQA
# from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv


app = Flask(__name__)


persist_directory='db'
upload_folder = 'pdf'
load_dotenv()


app.secret_key = 'your_secret_key'

# Dummy admin credentials for demonstration purposes
admin_username = 'admin'
admin_password = 'password123'



@app.route("/") 
def index():
    pdf_files = []
    # Replace 'path_to_your_folder' with the path to your PDF folder
    folder_path = 'pdf'

    # Fetching PDF file names from the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_files.append({'filename': filename})

    return render_template('chat.html', pdfs=pdf_files)


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)



def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_conversation_chain():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb=Chroma(persist_directory=persist_directory ,embedding_function=embedding_function)

    system_template = system_template = """
    Remember that you are a law bot you only answer questions realated to law and Answer questions using only the specified context provided. Points to adhere to:
    
        - Respond solely based on the given information.
        - Avoid speculation or assumptions; if uncertain, state 'I don't know.'
        - Do not incorporate outside knowledge or additional details beyond the context provided.
        - answer i dont know if the answer cant be found from the context and if the question is not related to law 
        - Ensure that responses strictly align with the provided context and refrain from extrapolating.
    ----------------
{summaries}"""
    # - Do not reference or disclose the provided context in your responses.

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1, max_tokens=1300)  # Modify model_name if you have access to GPT-4
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={'k':6}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return chain 


def get_Chat_response(user_input):
    # conversation_chain=get_conversation_chain()
    # response=conversation_chain({"question": user_input})
    # response_data = {'answer': response['answer'],
    #                  'sources':response['sources']}
    response_data = {'answer': 'answerssss',
                     'sources':'sources'}
    return jsonify({'sys_out': response_data})
    
    
def get_text_chunks():
    def load_docs():
        folder_path = 'pdf/*.pdf'
        pdf_files = glob.glob(folder_path)
        total_pages=[]
        for file in pdf_files:
            # print(file)
            loader = PyPDFLoader(file)
            pages = loader.load_and_split()
            total_pages.extend(pages)
        return total_pages
    documents = load_docs()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vectorstore():
    texts=get_text_chunks()
    # embeddings = OpenAIEmbeddings()
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectordb= FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectordb=Chroma.from_documents(texts,embedding=embedding_function,persist_directory=persist_directory )
    vectordb.persist()
    vectordb=None
    
# Function to save uploaded PDF to a specific folder
def save_pdf_to_folder(pdf_file):
    upload_folder = 'pdf'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, pdf_file.filename)
    pdf_file.save(file_path)
    return 


@app.route('/trigger_vectorization', methods=['POST'])
def trigger_vectorization():
    get_vectorstore()
    import time
    time.sleep(20)  # Simulating a process taking 5 seconds
    return render_template('chat.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == admin_username and password == admin_password:
            session['username'] = username
            return redirect(url_for('admin'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'username' in session:
        if session['username'] == admin_username:

            if request.method == 'POST':
                if 'pdf_file' in request.files:
                    pdf_file = request.files['pdf_file']
                    if pdf_file.filename != '':
                        save_pdf_to_folder(pdf_file)
            feedback_data=fetch_feedback_data()
            pdf_files = []
            # Replace 'path_to_your_folder' with the path to your PDF folder
            folder_path = 'pdf'

            # Fetching PDF file names from the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.pdf'):
                    pdf_files.append({'filename': filename})
            data={'feedback_data':feedback_data,
                  'pdfs':pdf_files}

            return render_template('admin.html',data=data)

    return redirect(url_for('login'))




import sqlite3

def initialize_database():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_text TEXT,
            user_message TEXT,
            bot_response TEXT,
            message_time TEXT,
            feedback_type TEXT
        )
    ''')
    conn.commit()
    conn.close()
@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    initialize_database()
    if request.method == 'POST':
        feedback_data = request.json  # Get the JSON data sent in the POST request
        feedback_text = feedback_data.get('feedback')
        user_message = feedback_data.get('userMessage')
        bot_response = feedback_data.get('botResponse')
        message_time = feedback_data.get('messageTime')
        feedback_type = feedback_data.get('feedbackType')  # Get the feedback type

        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (feedback_text, user_message, bot_response, message_time, feedback_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (feedback_text, user_message, bot_response, message_time, feedback_type))
        conn.commit()
        conn.close()

        return {"message": "Feedback added to database successfully"}, 200

def fetch_feedback_data():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM feedback')
    feedback_data = cursor.fetchall()
    conn.close()
    return feedback_data  # Ensure this returns a list of dictionaries or tuples

# @app.route('/admin', methods=['GET', 'POST'])
# def admin():
#     if request.method == 'POST':
#         if 'pdf_file' in request.files:
#             pdf_file = request.files['pdf_file']
#             if pdf_file.filename != '':
#                 save_pdf_to_folder(pdf_file)

#     feedback_data = fetch_feedback_data()

#     return render_template('admin.html', feedback_data=feedback_data)


if __name__ == '__main__':
    app.run()
