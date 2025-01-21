import os
import pprint

import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from st_keyup import st_keyup

DB_FAISS_PATH = 'vectorestore/faiss'


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        st.error('Document format is not supported!')
        return None

    data = loader.load()
    return data


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# Helper function to save uploaded files
def save_uploaded_files(uploaded_files, upload_dir="./uploaded_files"):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    uploaded_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        uploaded_file_paths.append(file_path)
    return uploaded_file_paths


def read_excel(uploaded_file):
    cols = ['Weight (C)', 'Category', 'Weight (I)', 'Indicator_ID', 'Indicator', 'Description', 'Prompting',
            'Prompting/Keyword', 'Weight (SI)']
    framework = pd.read_excel(uploaded_file, sheet_name='Prompting+scoring', usecols=cols).fillna(method='ffill')
    return framework


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db


def open_ai_ask_and_get_answer(vector_store, q, k=3, temperature=1, system_prompt=""):
    print('k ', k)
    print('temperature: ', temperature)
    print('system_prompt: ', system_prompt)
    from langchain.chat_models import ChatOpenAI
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain

    _system_prompt = (
            system_prompt +
            " Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _system_prompt),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=temperature)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    answer = chain.invoke({"input": q})
    return answer


load_dotenv(find_dotenv(), override=True)

choice = 'Vector RAG'

if choice == 'Vector RAG':
    # State to track app progress
    if "step" not in st.session_state:
        st.session_state["step"] = 1

    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = []

    if "questions" not in st.session_state:
        st.session_state['questions'] = []

    if st.session_state["step"] == 1:
        st.title("Upload Files")
        uploaded_files = st.file_uploader(
            "", type=["pdf", "docx", "txt"], accept_multiple_files=True
        )
        with st.spinner("Uploading files..."):
            uploaded_file_paths = save_uploaded_files(uploaded_files)
        # Add button in the right column
        col1, col2 = st.columns([9, 3])  # Adjust column width ratio as needed
        with col2:
            if st.button("Proceed to Next Step",
                         disabled=len(uploaded_files) == 0 and len(uploaded_file_paths) == len(uploaded_files)):
                st.session_state["uploaded_files"] = uploaded_files
                st.session_state["step"] = 2
                st.experimental_rerun()
    elif st.session_state["step"] == 2:
        st.title("Step 2: Upload Excel File")

        # File uploader for Excel
        excel_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

        if excel_file:
            try:
                # Read and process the Excel file
                st.write("Reading Excel file...")
                excel_data = read_excel(excel_file)

                if not excel_data.empty:
                    st.success("Excel file successfully uploaded and processed!")

                    # Display a preview of the questions
                    st.write("Preview of Questions:")
                    questions = excel_data['Prompting'].tolist()
                    print("*********** questions here", questions)
                    print("*********** questions len here", len(questions))
                    st.session_state["questions"] = questions.copy()
                    for idx, question in enumerate(questions, 1):
                        st.write(f"{idx}. {question}")
                    pprint.pprint('len(questions) here')
                    pprint.pprint(len(st.session_state['questions']))

            except Exception as e:
                st.error(f"Error processing Excel file: {e}")

        col1, col2 = st.columns([9, 3])
        with col2:
            if st.button("Proceed to Next Step", disabled=not bool(excel_file)):
                st.session_state["step"] = 3
                st.experimental_rerun()


    elif st.session_state["step"] == 3:

        st.title("Step 3: Configure and Process Files")

        model_option = st.selectbox("Select a model:", ["GPT-3.5", "Gemini", "PaperQA"])
        st.session_state['model'] = model_option
        if model_option == "GPT-3.5":
            api_key = st_keyup("OpenAI API Key: ", key='311', debounce=500)
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key

            system_prompt = st.text_area("Enter the system prompt:",
                                         help="A system prompt is a pre-written text prompt used to guide users through a conversation with an AI system. It sets the context, tone, and boundaries for the AI's responses, and is an essential part of building conversational AI systems123. The system prompt acts as a guiding framework, shaping the behavior and style of the AI throughout the interaction",
                                         height=110,
                                         value="""Use the given context to answer the question.If you don't know the answer, say you don't know.Use three sentence maximum and keep the answer concise.\n always start your answer with the name of company and then the rest.""")
            chunk_size = st.number_input("Chunk size:", min_value=100, max_value=2200, value=512,
                                         on_change=clear_history)
            chunk_overlap = st.number_input("Chunk overlap:", min_value=0, max_value=100, value=20,
                                            on_change=clear_history)
            temperature = st.slider("Temperature:", value=1.0, min_value=0.0, max_value=1.0, step=0.01)
            k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        if model_option == 'Gemini':
            api_key = st_keyup("GOOGLE_API_KEY : ", key='212121', debounce=500)

            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key

            system_prompt = st.text_area("Enter the system prompt:",
                                         help="A system prompt is a pre-written text prompt used to guide users through a conversation with an AI system. It sets the context, tone, and boundaries for the AI's responses, and is an essential part of building conversational AI systems123. The system prompt acts as a guiding framework, shaping the behavior and style of the AI throughout the interaction",
                                         height=110,
                                         value="""Use the given context to answer the question.If you don't know the answer, say you don't know.Use three sentence maximum and keep the answer concise.\n always start your answer with the name of company and then the rest.""")
            chunk_size = st.number_input("Chunk size:", min_value=100, max_value=2200, value=512,
                                         on_change=clear_history)
            chunk_overlap = st.number_input("Chunk overlap:", min_value=0, max_value=100, value=20,
                                            on_change=clear_history)
            temperature = st.slider("Temperature:", value=1.0, min_value=0.0, max_value=1.0, step=0.01)
            k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # Process button
        col1, col2 = st.columns([10.3, 1.7])

        with col1:
            process_button = st.button("Process Files", disabled=not api_key)
        with col2:
            next_button = st.button("Next Step",
                                    disabled="completed" not in st.session_state or not st.session_state["completed"])

        if process_button:
            st.session_state['temperature'] = temperature
            st.session_state['k'] = k
            st.session_state['system_prompt'] = system_prompt

            if "uploaded_files" in st.session_state:
                for uploaded_file in st.session_state["uploaded_files"]:
                    with st.spinner(f'Reading, chunking, and embedding file {uploaded_file.name}...'):
                        file_path = os.path.join("./uploaded_files", uploaded_file.name)

                        data = load_document(file_path)

                        if data:
                            chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                            tokens, embedding_cost = calculate_embedding_cost(chunks)

                            st.session_state["processed_files"].append({
                                "name": uploaded_file.name,
                                "chunks": chunks,
                                "tokens": tokens,
                                "embedding_cost": embedding_cost

                            })

                st.success("Files processed successfully!")

                st.session_state["completed"] = True
                st.experimental_rerun()

        if "processed_files" in st.session_state:

            st.write("Processed Files:")

            for processed_file in st.session_state["processed_files"]:
                st.write(
                    f"File: {processed_file['name']}, Tokens: {processed_file['tokens']}, Cost: ${processed_file['embedding_cost']:.4f}")

        if next_button:
            st.session_state["step"] = 4
            st.experimental_rerun()

    if st.session_state["step"] == 4:
        st.title("Step 4: Answer Questions and Export Results")
        print("Processed files:")
        pprint.pprint(st.session_state["processed_files"])
        print("Len(Questions)", len(st.session_state["questions"]))
        print("Questions")
        pprint.pprint(st.session_state["questions"])
        # Check if questions from Excel and processed files are available

        if "processed_files" not in st.session_state or len(st.session_state["processed_files"]) == 0:
            st.error("No processed files found. Please go back and process files first.")
        elif "questions" not in st.session_state or len(st.session_state["questions"]) == 0:
            st.error("No questions found from the Excel file. Please upload a valid Excel file.")
        else:
            questions = st.session_state["questions"]
            processed_files = st.session_state["processed_files"]
            vector_store_map = {}

            # Process files to create vector stores if not already done
            for processed_file in processed_files:
                if processed_file["name"] not in vector_store_map:
                    chunks = processed_file["chunks"]
                    vector_store = create_embeddings(chunks)
                    vector_store_map[processed_file["name"]] = vector_store

            # Answer questions

            st.write("Questions extracted from the Excel file:")

            # for idx, question in enumerate(questions[:10], 1):  # Show a preview of the first 10 questions
            #     st.write(f"{idx}. {question}")
            if st.button("Answer Questions"):
                results = []
                with st.spinner("Answering questions..."):
                    for question in questions:
                        answers = []
                        for file_name, vector_store in vector_store_map.items():
                            try:
                                answer = open_ai_ask_and_get_answer(
                                    vector_store,
                                    question,
                                    k=st.session_state.k,
                                    temperature=st.session_state.temperature,
                                    system_prompt=st.session_state.system_prompt

                                )
                                answers.append((file_name, answer))

                            except Exception as e:
                                st.error(f"Error answering question: {question}. Error: {e}")
                                answers.append((file_name, "Error generating answer"))
                        results.append((question, answers))
                st.session_state["results"] = results
                st.success("Questions answered successfully!")

            # Export results to Excel

            if "results" in st.session_state:
                results = st.session_state["results"]
                if st.button("Export Results"):
                    rows = []
                    for question, answers in results:
                        for file_name, answer in answers:
                            rows.append({"Question": question, "File": file_name, "Answer": answer})
                    results_df = pd.DataFrame(rows)
                    results_file = os.path.join("results", "questions_answers.xlsx")

                    if not os.path.exists("results"):
                        os.makedirs("results")
                    results_df.to_excel(results_file, index=False)
                    st.success(f"Results exported successfully to {results_file}!")
