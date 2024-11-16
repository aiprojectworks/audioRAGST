import glob
import json
import os
from openai import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, ID3NoHeaderError
from pydub import AudioSegment
from fpdf import FPDF
from fpdf.enums import Align
import base64
# from htmlTemplates import css, bot_template, user_template


css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://github.com/akshxyjagtap/Ask-form-PDF-using-langchain/blob/main/data/bot.png?raw=true" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://raw.githubusercontent.com/akshxyjagtap/Ask-form-PDF-using-langchain/main/data/user.webp">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""

def speech_to_text(audio_file):
    client = OpenAI(api_key=st.secrets["API_KEY"])
    dialog =""

    # Transcribe the audio
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_file, "rb"),
        prompt="Elena Pryor, Samir, Sahil, Mihir, IPP, IPPFA"

    )
    dialog = transcription.text
    # OPTIONAL: Uncomment the line below to print the transcription
    # print("Transcript: ", dialog + "  \n\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": """Insert speaker labels for a telemarketer and a customer. Return in a JSON format together with the original language code. Always translate the transcript fully to English."""},
        {"role": "user", "content": f"The audio transcript is: {dialog}"}
        ],
        temperature=0
    )

    output = response.choices[0].message.content
    print(output)
    dialog = output.replace("json", "").replace("```", "")
    formatted_transcript = ""
    dialog = json.loads(dialog)
    language_code = dialog["language_code"]
    # print(language_code)
    for entry in dialog['transcript']:
        formatted_transcript += f"{entry['speaker']}: {entry['text']}  \n\n"
    # print(formatted_transcript)

    # Joining the formatted transcript into a single string
    dialog = formatted_transcript

    return dialog, language_code

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # print(page.extract_text())
            text += page.extract_text()
        # print(text)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# embeddings = OpenAIEmbeddings()
# text_embeddings = embeddings.embed_documents(texts)
# text_embedding_pairs = zip(texts, text_embeddings)
# text_embedding_pairs_list = list(text_embedding_pairs)
# faiss = FAISS.from_embeddings(text_embedding_pairs_list, embeddings)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    print(text_chunks)
    text_embeddings = embeddings.embed_documents(text_chunks)
    text_embedding_pairs = zip(text_chunks, text_embeddings)
    text_embedding_pairs_list = list(text_embedding_pairs)
    faiss = FAISS.from_embeddings(text_embedding_pairs_list, embeddings)
    return faiss


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


@st.fragment
def download_history_button():
    # Prepare the conversation history as a string
    history_text = ""
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 0:
            history_text += f"OpenAI: {message.content}\n\n"
        else:
            history_text += f"User: {message.content}\n\n"

    # Button to download conversation history as a text file
    st.download_button(
        label="Download Conversation History",
        data=history_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )


def save_audio_file(audio_bytes, name):
    try:
        if name.lower().endswith(".wav") or name.lower().endswith(".mp3"):
            file_name = "./" + f"audio_{name}"
            with open(file_name, "wb") as f:
                f.write(audio_bytes)
            print(f"File saved successfully: {file_name}")
            return file_name  # Ensure you return the file name
    except Exception as e:
        print(f"Failed to save file: {e}")
        return None  # Explicitly return None on failure

def delete_mp3_files(directory):
    # Construct the search pattern for MP3 files
    mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
    
    for mp3_file in mp3_files:
        try:
            os.remove(mp3_file)
            # print(f"Deleted: {mp3_file}")
        except FileNotFoundError:
            print(f"{mp3_file} does not exist.")
        except Exception as e:
            print(f"Error deleting file {mp3_file}: {e}")

def convert_audio_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    wav_file = audio_file.name.split(".")[0] + ".wav"
    audio.export(wav_file, format="wav")
    return wav_file

def is_valid_mp3(file_path):
    # Check if file exists
    if not os.path.isfile(file_path):
        print("File does not exist.")
        return False

    try:
        # Check the file using mutagen
        audio = MP3(file_path, ID3=ID3)
        
        # Check for basic properties
        if audio.info.length <= 0:  # Length should be greater than 0
            print("File is invalid: Length is zero or negative.")
            return False
        
        # You can check additional metadata if needed
        print("File is valid MP3 with duration:", audio.info.length)
        
        # Optional: Check if the file can be loaded with pydub
        AudioSegment.from_file(file_path)  # This will raise an exception if the file is not valid
        
        return True
    except (ID3NoHeaderError, Exception) as e:
        print(f"Invalid MP3 file: {e}")
        # create_log_entry(f"Error: Invalid MP3 file: {e}")
        return False

def rag_process(pdf_docs):
    # get pdf text
    raw_text = get_pdf_text(pdf_docs)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vectorstore)

@st.fragment
def document_interaction():
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    user_question = st.text_input("Ask a question about your document:", value=st.session_state.user_question)
    if user_question:
        handle_userinput(user_question)
        # Reset the user_question field
        st.session_state.user_question = ""

@st.fragment
def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chat_history = response["chat_history"]

        # Create an expander for the chat history
        with st.expander("Chat History", expanded=True):
            # Display chat history in a scrollable manner
            for i, message in enumerate(reversed(st.session_state.chat_history)):
                if i % 2 == 0:
                    st.write(
                        bot_template.replace("{{MSG}}", message.content),
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(
                        user_template.replace("{{MSG}}", message.content),
                        unsafe_allow_html=True
                    )

        # Button to download conversation history
        download_history_button()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# @st.fragment
# def submit():
#     user_question = st.session_state.widget
#     if user_question:
#         handle_userinput(user_question)
#         # Reset the user_question field
#         st.session_state.user_question = ""
#     st.session_state.widget = ''

# @st.fragment
# def document_interaction():
#     if 'user_question' not in st.session_state:
#         st.session_state.user_question = ""
    
#     user_question = st.text_input("Ask a question about your documents:", on_change=submit, key='widget', )

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():
    audio_files = []
    display_is_true = False
    upload_method = False

    # Initialize session state to track files and change detection
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'file_change_detected' not in st.session_state:
        st.session_state.file_change_detected = False
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []



    st.set_page_config(page_title="Chat with a PDF",
                       page_icon=":books:")
    st.sidebar.title("OpenAI Configuration")
    os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", value=os.environ["OPENAI_API_KEY"])
    st.sidebar.markdown('### Need an OpenAI API Key?')
    st.sidebar.markdown(
        "To use this application, you'll need an OpenAI API key. If you don't have one, you can obtain it [here](https://platform.openai.com/api-keys) "
    )
    # ... (rest of the code remains unchanged)

    with st.sidebar:
        # Create a radio button to switch between text and PDF
        choice = st.radio("Choose an option:", ("Text", "PDF"))

        if choice == "Text":
            st.subheader("Upload Your Text Document")
            pdf_docs = st.file_uploader("Upload your Text here and click on 'Process'", accept_multiple_files=False, type="txt")
            pdf_doc_for_ai = [pdf_docs]
            if st.button("Process Text"):
                if pdf_doc_for_ai != [None]:
                    with st.spinner("Processing"):
                        pdf_doc_for_ai = pdf_conversion(pdf_docs)
                        rag_process([pdf_doc_for_ai])
                        pdf_docs = pdf_doc_for_ai
                        upload_method = True
                        display_is_true = True
                    st.write("Text File Successfully Converted to PDF!")
                else:
                    st.error("No Text File Found!")

        elif choice == "PDF":
            st.subheader("Upload Your PDF Document")
            pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=False, type="pdf")
            pdf_doc_for_ai = [pdf_docs]
            if st.button("Process PDF"):
                if pdf_doc_for_ai != [None]:
                    with st.spinner("Processing"):
                        rag_process(pdf_doc_for_ai)
                    st.write(f"Using PDF File: {pdf_doc_for_ai[-1].name}")
                else:
                    st.error("No PDF Found!")

    # Add attribution in the app
    st.markdown(
        """
        *RAG*
        """,
        unsafe_allow_html=True
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with a PDF :books:")
    uploaded_file = st.file_uploader(
    label="Choose an audio file", 
    label_visibility="collapsed", 
    type=["wav", "mp3"], 
    accept_multiple_files=False  # Allow only a single file
    )

    if uploaded_file is not None:
        # Track current file
        current_file_name = uploaded_file.name

        # Check if the file has been uploaded before
        if current_file_name not in st.session_state.uploaded_files:
            # create_log_entry(f"Action: File Uploaded - {current_file_name}")
            
            try:
                audio_content = uploaded_file.read()
                saved_path = save_audio_file(audio_content, current_file_name)
                if is_valid_mp3(saved_path):
                    st.session_state.audio_files = [saved_path]  # Store only the latest file
                    st.session_state.uploaded_files = [current_file_name]  # Update uploaded files
                    st.session_state.file_change_detected = True
                else:
                    st.error(f"{saved_path[2:]} is an Invalid MP3 or WAV File")
                    # create_log_entry(f"Error: {saved_path[2:]} is an Invalid MP3 or WAV File")
            except Exception as e:
                st.error(f"Error loading audio file: {e}")
                # create_log_entry(f"Error loading audio file: {e}")

    # Clear the audio_files if no file is uploaded
    if uploaded_file is None:
        st.session_state.audio_files = []

    audio_files = list(st.session_state.audio_files)
    # print(st.session_state.audio_files)
    # print(type(audio_files))


    submit = st.button("Submit", use_container_width=True)

    if submit and audio_files == []:
        # create_log_entry("Service Request: Fail (No Files Uploaded)")
        st.error("No Files Uploaded, Please Try Again!")

    elif display_is_true and upload_method:
        if isinstance(pdf_docs, str):
            displayPDF(pdf_docs)
            os.remove(pdf_docs)
        else:
            displayPDF(pdf_docs.name)
        document_interaction()


    elif submit:
        for audio_file in audio_files:
            if not os.path.isfile(audio_file):
                st.error(f"{audio_file[2:]} Not Found, Please Try Again!")
            else:
                with st.spinner("Transcribing In Progress..."):
                    # Transcribe the audio to text and detect language
                    text, language_code = speech_to_text(audio_file)
                
                with st.expander(f"{audio_file[2:]} ({language_code})"):
                    st.write(text)
                    # print(audio_file)
                    pdf_filename = pdf_conversion(text, audio_file)
    
                    # Call `rag_process` to handle the generated PDF
                    rag_process([pdf_filename])
                displayPDF(pdf_filename)
                os.remove(pdf_filename)
                    
        directory = "./"
        delete_mp3_files(directory)
        document_interaction()


def pdf_conversion(text, audio_file=""):
    if audio_file == "":
        audio_file = "./" + text.name.replace(".txt", "")
        text = text.read().decode("utf-8")
    
    print(text)

    print(audio_file)

    # Create and configure PDF
    pdf = FPDF()
    pdf.add_page()

    # Register and set a Unicode font
    pdf.add_font("DejaVu", "", "./dejavu-sans/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=16)

    # Add a title with centered alignment
    pdf.multi_cell(0, 10, txt=f"Title: {audio_file[2:]}", align='C')

    # Add space after the title
    pdf.ln(10)
    pdf.set_font("DejaVu", size=12)

    # Set margins
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)

    # Split the text into paragraphs or lines
    lines = text.split('\n')

    for line in lines:
        # Add each line to the PDF
        pdf.multi_cell(0, 10, txt=line, align='L')

    # Save the PDF with the name derived from the audio file
    pdf_filename = f"{os.path.splitext(os.path.basename(audio_file))[0]}.pdf"
    pdf.output(pdf_filename)

    return pdf_filename

if __name__ == '__main__':
    main()
