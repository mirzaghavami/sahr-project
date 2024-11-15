import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


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


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def open_ai_ask_and_get_answer(vector_store, q, k=3, temperature=1, system_prompt=""):
    from langchain.chat_models import ChatOpenAI
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain

    _system_prompt = (
            system_prompt +
            " Context: {context}"
    )

    print('Ssss', _system_prompt)

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


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


def read_excel(uploaded_file):
    cols = ['Indicator_ID', 'Indicator', 'Description', 'Prompting', 'Prompting/Keyword', 'Keywords']
    framework = pd.read_excel(uploaded_file, sheet_name='Prompting+scoring', usecols=cols).fillna(method='ffill')
    return framework


def ask_questions_from_excel(vector_store, questions, k=3, temperature=1, system_prompt=""):
    print("SSSSSSSSS,", system_prompt)
    q_and_a = []
    for question in questions:
        answer = open_ai_ask_and_get_answer(vector_store, question, k, temperature=temperature,
                                            system_prompt=system_prompt)
        q_and_a.append((question, answer))
    return q_and_a


def export_to_excel(q_and_a, name):
    df = pd.DataFrame(q_and_a, columns=['Question', 'Answer'])

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_path = os.path.join(results_dir, f'{name}_exported_results.xlsx')
    df.to_excel(file_path, index=False)

    return file_path


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)

    st.image('banner.png')
    st.subheader('LLM Question-Answering Application')

    with st.sidebar:
        # Model selection combo box
        model_option = st.selectbox("Select a model:", ["GPT-3.5", "Llama 3.2", "PaperQA"])

        # Conditional rendering based on selected model
        if model_option == "GPT-3.5":
            api_key = st.text_input("OpenAI API Key: ", type='password')
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key

            system_prompt = st.text_area("Enter the system prompt:", value=""
                                                                           "Use the given context to answer the question. "
                                                                           "If you don't know the answer, say you don't know. "
                                                                           "Use three sentence maximum and keep the answer concise. "
                                                                           "")
            uploaded_files = st.file_uploader('Upload files:', type=['pdf', 'docx', 'txt'], accept_multiple_files=True,
                                              on_change=clear_history)

            excel_file = st.file_uploader("Upload Excel (optional):", type=["xlsx"], on_change=clear_history)
            chunk_size = st.number_input("Chunk size:", min_value=100, max_value=2200, value=512,
                                         on_change=clear_history)
            chunk_overlap = st.number_input("Chunk overlap:", min_value=0, max_value=100, value=20,
                                            on_change=clear_history)
            temperature = st.slider("Temperature:", value=1.0, min_value=0.0, max_value=1.0, step=0.01)
            k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
            add_data = st.button('Add Data', on_click=clear_history)

            if uploaded_files and add_data:
                for uploaded_file in uploaded_files:
                    with st.spinner(f'Reading, chunking, and embedding file {uploaded_file.name}...'):
                        bytes_data = uploaded_file.read()
                        file_path = os.path.join('./uploaded_files', uploaded_file.name)
                        with open(file_path, 'wb') as f:
                            f.write(bytes_data)

                        data = load_document(file_path)
                        if data:
                            chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            tokens, embedding_cost = calculate_embedding_cost(chunks)
                            st.write(f"Embedding cost for {uploaded_file.name}: ${embedding_cost:.4f}")

                            vector_store = create_embeddings(chunks)
                            st.session_state[f'vs_{uploaded_file.name}'] = vector_store
                            st.success(f'File {uploaded_file.name} uploaded, chunked, and embedded successfully.')
        if model_option == 'Llama 3.2':
            st.success("Sag Model")

    if excel_file:
        sheet = read_excel(excel_file)
        if not sheet.empty:
            questions = sheet['Description'].tolist()
            st.write("Questions from Excel:")
            for idx, question in enumerate(questions, 1):
                st.write(f"{idx}. {question}")

            if st.button("Ask Questions from Excel", disabled=not bool(excel_file)):
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    vector_store = st.session_state.get(f'vs_{file_name}')
                    if vector_store:
                        with st.spinner(f'Asking questions for {file_name}...'):
                            q_and_a = ask_questions_from_excel(vector_store, questions, k=k, temperature=temperature,
                                                               system_prompt=system_prompt)
                            path_of_export = export_to_excel(q_and_a, file_name)
                            st.success(f'Results for {file_name} successfully exported to: {path_of_export}')
    q = st.text_input("Ask a question about the content of your file:")

    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = open_ai_ask_and_get_answer(vector_store, q, k, temperature=temperature,
                                                system_prompt=system_prompt)
            st.text_area('LLM Answer:', value=answer)

            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
