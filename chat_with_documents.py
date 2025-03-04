"""
Chat with Documents Application

A Streamlit application that allows users to upload documents, process them,
and ask questions using various LLM models.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import streamlit as st
import tiktoken
from dotenv import load_dotenv, find_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from st_keyup import st_keyup

# Constants
DEFAULT_SYSTEM_PROMPT = """Use the given context to answer the question.
If you don't know the answer, say you don't know.
Use three sentence maximum and keep the answer concise.
Always start your answer with the name of company and then the rest."""

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_TEMPERATURE = 1.0
DEFAULT_K = 3
VECTOR_STORE_PATH = Path('vectorestore/faiss')
UPLOAD_DIR = Path('./uploaded_files')
RESULTS_DIR = Path('./results')


# Enums for better type safety
class ModelType(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GEMINI = "gemini-1.5-flash-8b"


class Step(int, Enum):
    UPLOAD_FILES = 1
    UPLOAD_EXCEL = 2
    CONFIGURE = 3
    ANSWER_QUESTIONS = 4


# Data classes for better structure
@dataclass
class ProcessedFile:
    name: str
    chunks: List[Document]
    tokens: Optional[int] = None
    embedding_cost: Optional[float] = None


@dataclass
class AppConfig:
    model: ModelType
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    temperature: float = DEFAULT_TEMPERATURE
    k: int = DEFAULT_K


@dataclass
class QuestionAnswer:
    question: str
    file_name: str
    answer: Any



# Add a new UI helper class after the LLMService class
class UIHelper:
    @staticmethod
    def create_loading_container():
        """Create a container for loading messages and spinners."""
        return st.empty()
    
    @staticmethod
    def show_loading(container, message="Processing..."):
        """Show a loading spinner with a message."""
        with container:
            with st.spinner(message):
                st.info(message)
    
    @staticmethod
    def clear_loading(container):
        """Clear the loading container."""
        container.empty()
    
    @staticmethod
    def disable_buttons_during_processing(key_prefix="processing"):
        """Set a session state flag to disable buttons during processing."""
        processing_key = f"{key_prefix}_in_progress"
        if processing_key not in st.session_state:
            st.session_state[processing_key] = False
        return processing_key
    
    @staticmethod
    def start_processing(processing_key):
        """Mark processing as started."""
        st.session_state[processing_key] = True
    
    @staticmethod
    def end_processing(processing_key):
        """Mark processing as completed."""
        st.session_state[processing_key] = False
    
    @staticmethod
    def is_processing(processing_key):
        """Check if processing is in progress."""
        return st.session_state.get(processing_key, False)
    
    @staticmethod
    def create_progress_bar(total_steps, key="progress"):
        """Create a progress bar for tracking multi-step processes."""
        if f"{key}_value" not in st.session_state:
            st.session_state[f"{key}_value"] = 0
        if f"{key}_total" not in st.session_state:
            st.session_state[f"{key}_total"] = total_steps
        
        progress = st.progress(0)
        return progress, f"{key}_value", f"{key}_total"
    
    @staticmethod
    def update_progress(progress_bar, value_key, total_key, step=1, message=None):
        """Update the progress bar by incrementing the current value."""
        st.session_state[value_key] += step
        current = st.session_state[value_key]
        total = st.session_state[total_key]
        progress_bar.progress(min(current / total, 1.0))
        if message:
            st.write(message)
    
    @staticmethod
    def reset_progress(value_key):
        """Reset the progress counter."""
        st.session_state[value_key] = 0


# Abstract classes for embedding and LLM strategies
class EmbeddingStrategy(ABC):
    @abstractmethod
    def create_embeddings(self, chunks: List[Document]) -> FAISS:
        pass


class LLMStrategy(ABC):
    @abstractmethod
    def ask_and_get_answer(self, vector_store: FAISS, question: str, config: AppConfig) -> Any:
        pass


# Concrete embedding strategies
class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    def create_embeddings(self, chunks: List[Document]) -> FAISS:
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(VECTOR_STORE_PATH)
        return db


class GoogleAIEmbeddingStrategy(EmbeddingStrategy):
    def create_embeddings(self, chunks: List[Document]) -> FAISS:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(VECTOR_STORE_PATH)
            return db
        except Exception as e:
            st.error(f"Error creating Google AI embeddings: {e}")
            return None


# Concrete LLM strategies
class OpenAILLMStrategy(LLMStrategy):
    def ask_and_get_answer(self, vector_store: FAISS, question: str, config: AppConfig) -> Any:
        from langchain.chat_models import ChatOpenAI

        _system_prompt = f"{config.system_prompt} Context: {{context}}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", _system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatOpenAI(model=config.model, temperature=config.temperature)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': config.k})
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        return chain.invoke({"input": question})


class GeminiLLMStrategy(LLMStrategy):
    def ask_and_get_answer(self, vector_store: FAISS, question: str, config: AppConfig) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _system_prompt = f"{config.system_prompt} Context: {{context}}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", _system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", temperature=config.temperature)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': config.k})
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        try:
            return chain.invoke({"input": question})
        except Exception as e:
            st.error(f"Error with Gemini: {e}")
            return None


# Factory for creating strategies
class StrategyFactory:
    @staticmethod
    def get_embedding_strategy(model_type: ModelType) -> EmbeddingStrategy:
        if model_type == ModelType.GEMINI:
            return GoogleAIEmbeddingStrategy()
        return OpenAIEmbeddingStrategy()

    @staticmethod
    def get_llm_strategy(model_type: ModelType) -> LLMStrategy:
        if model_type == ModelType.GEMINI:
            return GeminiLLMStrategy()
        return OpenAILLMStrategy()


# Service classes for different functionalities
class DocumentService:
    @staticmethod
    def load_document(file_path: str) -> Optional[List[Document]]:
        """Load a document from a file path."""
        path = Path(file_path)
        extension = path.suffix.lower()

        try:
            if extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif extension == '.txt':
                loader = TextLoader(file_path)
            else:
                st.error(f'Document format {extension} is not supported!')
                return None

            return loader.load()
        except Exception as e:
            st.error(f"Error loading document {path.name}: {e}")
            return None

    @staticmethod
    def chunk_data(data: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(data)

    @staticmethod
    def calculate_embedding_cost(texts: List[Document]) -> Tuple[int, float]:
        """Calculate the cost of embedding the given texts."""
        enc = tiktoken.encoding_for_model('text-embedding-3-large')
        total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
        cost_per_1k_tokens = 0.00013  # Updated cost for text-embedding-3-large
        return total_tokens, total_tokens / 1000 * cost_per_1k_tokens


class FileService:
    @staticmethod
    def save_uploaded_files(uploaded_files, upload_dir: Path = UPLOAD_DIR) -> List[str]:
        """Save uploaded files to disk and return their paths."""
        upload_dir.mkdir(exist_ok=True, parents=True)

        uploaded_file_paths = []
        for uploaded_file in uploaded_files:
            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            uploaded_file_paths.append(str(file_path))
        return uploaded_file_paths

    @staticmethod
    def read_excel(uploaded_file) -> pd.DataFrame:
        """Read and process an Excel file."""
        cols = [
            'Weight (C)', 'Category', 'Weight (I)', 'Indicator_ID', 
            'Indicator', 'Description', 'Prompting', 'Prompting/Keyword', 'Weight (SI)'
        ]
        try:
            framework = pd.read_excel(uploaded_file, sheet_name='Prompting+scoring', usecols=cols).fillna(method='ffill')
            return framework
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return pd.DataFrame()

    @staticmethod
    def export_results(results: List[QuestionAnswer], filename: str = "questions_answers.xlsx") -> str:
        """Export results to an Excel file."""
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        file_path = RESULTS_DIR / filename
        
        df = pd.DataFrame([
            {"Question": qa.question, "File": qa.file_name, "Answer": qa.answer}
            for qa in results
        ])
        
        df.to_excel(file_path, index=False)
        return str(file_path)


class LLMService:
    @staticmethod
    def retry_gemini_call(
        vector_store: FAISS, 
        question: str, 
        config: AppConfig, 
        max_retries: int = 10, 
        base_delay: int = 10
    ) -> Any:
        """Retry Gemini API calls with exponential backoff."""
        strategy = GeminiLLMStrategy()
        retries = 0
        
        while retries < max_retries:
            answer = strategy.ask_and_get_answer(vector_store, question, config)
            if answer is not None:
                return answer
                
            retries += 1
            wait_time = base_delay * (2 ** (retries - 1))  # Exponential backoff
            st.write(f"Retrying Gemini call ({retries}/{max_retries}) in {wait_time} seconds...")
            time.sleep(wait_time)
            
        return "Error generating answer after multiple attempts"




# Main application class
class ChatWithDocumentsApp:
    def __init__(self):
        self._initialize_session_state()
        load_dotenv(find_dotenv(), override=True)
        # Initialize processing state keys
        self.upload_processing_key = UIHelper.disable_buttons_during_processing("upload")
        self.excel_processing_key = UIHelper.disable_buttons_during_processing("excel")
        self.config_processing_key = UIHelper.disable_buttons_during_processing("config")
        self.qa_processing_key = UIHelper.disable_buttons_during_processing("qa")
        self.export_processing_key = UIHelper.disable_buttons_during_processing("export")

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "step" not in st.session_state:
            st.session_state["step"] = Step.UPLOAD_FILES
        
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = []
            
        if "questions" not in st.session_state:
            st.session_state['questions'] = []
            
        if "completed" not in st.session_state:
            st.session_state["completed"] = False
            
        if "results" not in st.session_state:
            st.session_state["results"] = []

    def clear_history(self):
        """Clear conversation history."""
        if 'history' in st.session_state:
            del st.session_state['history']

    def run(self):
        """Run the application."""
        # Add a nice header with styling
        st.markdown("""
        <style>
        .app-header {
            background: linear-gradient(135deg, #00416A 0%, #E4E5E6 50%, #00416A 100%);
            color: #00416A;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        .step-container {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .button-container {
            display: flex;
            justify-content: flex-end;
            margin-top: 1rem;
        }
        </style>
        <div class="app-header">
            <h1>Chat with Documents</h1>
            <p>Upload documents, process them, and ask questions using AI models</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the current step as a progress indicator
        current_step = st.session_state["step"]
        steps = ["Upload Files", "Upload Excel", "Configure", "Answer Questions"]
        
        cols = st.columns(len(steps))
        for i, step_name in enumerate(steps):
            with cols[i]:
                if i + 1 < current_step:
                    st.markdown(f"✅ **{step_name}**")
                elif i + 1 == current_step:
                    st.markdown(f"🔄 **{step_name}**")
                else:
                    st.markdown(f"⏳ {step_name}")
        
        st.markdown("---")
        
        # Run the appropriate step
        if current_step == Step.UPLOAD_FILES:
            self._handle_upload_files_step()
        elif current_step == Step.UPLOAD_EXCEL:
            self._handle_upload_excel_step()
        elif current_step == Step.CONFIGURE:
            self._handle_configure_step()
        elif current_step == Step.ANSWER_QUESTIONS:
            self._handle_answer_questions_step()

    def _handle_upload_files_step(self):
        """Handle the file upload step."""
        with st.container():
            # st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.subheader("Upload Documents")
            st.markdown("Upload PDF, DOCX, or TXT files that you want to analyze.")
            
            uploaded_files = st.file_uploader(
                "Select files to upload", 
                type=["pdf", "docx", "txt"], 
                accept_multiple_files=True,
                help="You can upload multiple files at once"
            )
            
            # Create a loading container for file upload status
            upload_status_container = UIHelper.create_loading_container()
            
            if uploaded_files:
                file_info = ", ".join([f"{file.name} ({round(file.size/1024, 1)} KB)" for file in uploaded_files])
                st.success(f"Files ready for upload: {file_info}")
            
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            proceed_button = st.button(
                "Proceed to Next Step ➡️", 
                disabled=len(uploaded_files) == 0 or UIHelper.is_processing(self.upload_processing_key),
                type="primary"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if proceed_button:
                UIHelper.start_processing(self.upload_processing_key)
                UIHelper.show_loading(upload_status_container, "Saving uploaded files...")
                
                try:
                    uploaded_file_paths = FileService.save_uploaded_files(uploaded_files)
                    st.session_state["uploaded_files"] = uploaded_files
                    
                    # Show success message with a brief delay for better UX
                    UIHelper.clear_loading(upload_status_container)
                    with upload_status_container:
                        st.success(f"Successfully saved {len(uploaded_file_paths)} files!")
                    time.sleep(1)
                    
                    st.session_state["step"] = Step.UPLOAD_EXCEL
                    UIHelper.end_processing(self.upload_processing_key)
                    st.rerun()
                except Exception as e:
                    UIHelper.clear_loading(upload_status_container)
                    with upload_status_container:
                        st.error(f"Error saving files: {e}")
                    UIHelper.end_processing(self.upload_processing_key)
            
            st.markdown('</div>', unsafe_allow_html=True)

    def _handle_upload_excel_step(self):
        """Handle the Excel upload step."""
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.subheader("Upload Excel File")
            st.markdown("Upload an Excel file containing questions to ask about the documents.")
            
            # # Add a back button
            # col1, col2 = st.columns([3, 11])
            # with col1:
            #     if st.button("⬅️ Back", key="back_to_upload"):
            #         st.session_state["step"] = Step.UPLOAD_FILES
            #         st.rerun()
            
            excel_file = st.file_uploader(
                "Upload your Excel file", 
                type=["xlsx"],
                help="The Excel file should contain a 'Prompting+scoring' sheet with questions in the 'Prompting' column"
            )
            
            # Create a loading container for Excel processing
            excel_status_container = UIHelper.create_loading_container()
            
            if excel_file:
                st.info(f"Excel file: {excel_file.name} ({round(excel_file.size/1024, 1)} KB)")
                
                try:
                    UIHelper.show_loading(excel_status_container, "Reading Excel file...")
                    excel_data = FileService.read_excel(excel_file)
                    
                    if not excel_data.empty:
                        UIHelper.clear_loading(excel_status_container)
                        with excel_status_container:
                            st.success("Excel file successfully processed!")
                        
                        questions = excel_data['Prompting'].tolist()
                        st.session_state["questions"] = questions.copy()
                        
                        st.markdown("### Preview of Questions")
                        
                        # Show questions in an expandable section if there are many
                        if len(questions) > 5:
                            with st.expander(f"View all {len(questions)} questions"):
                                for idx, question in enumerate(questions, 1):
                                    st.write(f"{idx}. {question}")
                        else:
                            for idx, question in enumerate(questions, 1):
                                st.write(f"{idx}. {question}")
                    else:
                        UIHelper.clear_loading(excel_status_container)
                        with excel_status_container:
                            st.warning("The Excel file doesn't contain any questions in the expected format.")
                except Exception as e:
                    UIHelper.clear_loading(excel_status_container)
                    with excel_status_container:
                        st.error(f"Error processing Excel file: {e}")
            
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            proceed_button = st.button(
                "Proceed to Next Step ➡️", 
                disabled=not bool(excel_file) or UIHelper.is_processing(self.excel_processing_key),
                type="primary"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if proceed_button:
                UIHelper.start_processing(self.excel_processing_key)
                try:
                    st.session_state["step"] = Step.CONFIGURE
                    UIHelper.end_processing(self.excel_processing_key)
                    st.rerun()
                except Exception as e:
                    with excel_status_container:
                        st.error(f"Error proceeding to next step: {e}")
                    UIHelper.end_processing(self.excel_processing_key)
            
            st.markdown('</div>', unsafe_allow_html=True)

    def _handle_configure_step(self):
        """Handle the configuration step."""
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.subheader("Configure Processing")
            st.markdown("Configure the AI model and processing parameters.")
            
            # Add a back button
            # col1, col2 = st.columns([1, 11])
            # with col1:
            #     if st.button("⬅️ Back", key="back_to_excel"):
            #         st.session_state["step"] = Step.UPLOAD_EXCEL
            #         st.rerun()
            
            # Create tabs for different configuration sections
            model_tab, chunking_tab, advanced_tab = st.tabs(["Model Selection", "Document Chunking", "Advanced Settings"])
            
            with model_tab:
                model_options = [model.value for model in ModelType]
                model_option = st.selectbox(
                    "Select AI Model:", 
                    model_options,
                    help="Choose the AI model to use for answering questions"
                )
                st.session_state['model'] = model_option
                
                # Model-specific API key input
                if model_option in [ModelType.GPT_4O.value, ModelType.GPT_35_TURBO.value]:
                    api_key = st_keyup(
                        "OpenAI API Key:", 
                        key='openai_key', 
                        debounce=500
                    )
                    if api_key:
                        os.environ['OPENAI_API_KEY'] = api_key
                        st.success("OpenAI API key set successfully!")
                elif model_option == ModelType.GEMINI.value:
                    api_key = st_keyup(
                        "Google AI API Key:", 
                        key='google_key', 
                        debounce=500
                    )
                    if api_key:
                        os.environ['GOOGLE_API_KEY'] = api_key
                        st.success("Google AI API key set successfully!")
            
            with chunking_tab:
                chunk_size = st.number_input(
                    "Chunk size:", 
                    min_value=100, 
                    max_value=2200, 
                    value=DEFAULT_CHUNK_SIZE,
                    help="Size of text chunks in characters. Smaller chunks may improve accuracy but increase processing time.",
                    on_change=self.clear_history
                )
                
                chunk_overlap = st.number_input(
                    "Chunk overlap:", 
                    min_value=0, 
                    max_value=100, 
                    value=DEFAULT_CHUNK_OVERLAP,
                    help="Percentage of overlap between chunks to maintain context across chunk boundaries.",
                    on_change=self.clear_history
                )
            
            with advanced_tab:
                system_prompt = st.text_area(
                    "System Prompt:",
                    help="Instructions for the AI model on how to answer questions.",
                    height=110,
                    value=DEFAULT_SYSTEM_PROMPT
                )
                
                temperature = st.slider(
                    "Temperature:", 
                    value=DEFAULT_TEMPERATURE, 
                    min_value=0.0, 
                    max_value=1.0, 
                    step=0.01,
                    help="Controls randomness in responses. Lower values make responses more deterministic."
                )
                
                k = st.number_input(
                    'Number of chunks (k):', 
                    min_value=1, 
                    max_value=20, 
                    value=DEFAULT_K, 
                    help="Number of most relevant chunks to retrieve for each question.",
                    on_change=self.clear_history
                )
            
            # Create a loading container for processing status
            processing_status_container = UIHelper.create_loading_container()
            
            # Process and Next buttons
            col1, col2 = st.columns([5, 5])
            with col1:
                process_button = st.button(
                    "Process Files", 
                    disabled=not api_key or UIHelper.is_processing(self.config_processing_key),
                    type="primary"
                )
            with col2:
                next_button = st.button(
                    "Next Step ➡️",
                    disabled=not st.session_state.get("completed", False) or UIHelper.is_processing(self.config_processing_key)
                )
            
            # Display processed files in an expander
            if "processed_files" in st.session_state and st.session_state["processed_files"]:
                with st.expander("View Processed Files", expanded=True):
                    for processed_file in st.session_state["processed_files"]:
                        if processed_file.tokens is not None:
                            st.write(
                                f"✅ **{processed_file.name}** - "
                                f"Tokens: {processed_file.tokens:,}, "
                                f"Cost: ${processed_file.embedding_cost:.4f}"
                            )
                        else:
                            st.write(f"✅ **{processed_file.name}**")
            
            if process_button:
                UIHelper.start_processing(self.config_processing_key)
                UIHelper.show_loading(processing_status_container, "Initializing processing...")
                
                try:
                    # Save configuration to session state
                    config = AppConfig(
                        model=model_option,
                        system_prompt=system_prompt,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        temperature=temperature,
                        k=k
                    )
                    st.session_state['config'] = config
                    
                    # Process files with progress tracking
                    if "uploaded_files" in st.session_state:
                        total_files = len(st.session_state["uploaded_files"])
                        UIHelper.clear_loading(processing_status_container)
                        
                        with processing_status_container:
                            progress_bar, value_key, total_key = UIHelper.create_progress_bar(total_files, "file_processing")
                            st.write("Processing files...")
                            
                            # Clear previous processed files
                            st.session_state["processed_files"] = []
                            
                            for i, uploaded_file in enumerate(st.session_state["uploaded_files"]):
                                file_status = st.empty()
                                file_status.info(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")
                                
                                file_path = os.path.join(str(UPLOAD_DIR), uploaded_file.name)
                                data = DocumentService.load_document(file_path)
                                
                                if data:
                                    file_status.info(f"Chunking file {i+1}/{total_files}: {uploaded_file.name}")
                                    chunks = DocumentService.chunk_data(
                                        data, 
                                        chunk_size=config.chunk_size, 
                                        chunk_overlap=config.chunk_overlap
                                    )
                                    
                                    if config.model in [ModelType.GPT_4O.value, ModelType.GPT_35_TURBO.value]:
                                        file_status.info(f"Calculating embedding cost for {uploaded_file.name}")
                                        tokens, embedding_cost = DocumentService.calculate_embedding_cost(chunks)
                                        processed_file = ProcessedFile(
                                            name=uploaded_file.name,
                                            chunks=chunks,
                                            tokens=tokens,
                                            embedding_cost=embedding_cost
                                        )
                                    else:
                                        processed_file = ProcessedFile(
                                            name=uploaded_file.name,
                                            chunks=chunks
                                        )
                                    
                                    st.session_state["processed_files"].append(processed_file)
                                    UIHelper.update_progress(
                                        progress_bar, 
                                        value_key, 
                                        total_key, 
                                        message=f"Completed {i+1}/{total_files} files"
                                    )
                                else:
                                    file_status.error(f"Failed to process {uploaded_file.name}")
                            
                            st.success(f"Successfully processed {len(st.session_state['processed_files'])}/{total_files} files!")
                    
                    st.session_state["completed"] = True
                    UIHelper.end_processing(self.config_processing_key)
                    st.rerun()
                except Exception as e:
                    UIHelper.clear_loading(processing_status_container)
                    with processing_status_container:
                        st.error(f"Error during processing: {e}")
                    UIHelper.end_processing(self.config_processing_key)
            
            if next_button:
                st.session_state["step"] = Step.ANSWER_QUESTIONS
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

    def _handle_answer_questions_step(self):
        """Handle the question answering step."""
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.subheader("Answer Questions and Export Results")
            
            # Add a back button
            col1, col2 = st.columns([1, 11])
            with col1:
                if st.button("⬅️ Back", key="back_to_config"):
                    st.session_state["step"] = Step.CONFIGURE
                    st.rerun()
            
            # Check if questions and processed files are available
            if not st.session_state.get("processed_files"):
                st.error("No processed files found. Please go back and process files first.")
                return
            elif not st.session_state.get("questions"):
                st.error("No questions found from the Excel file. Please upload a valid Excel file.")
                return
            
            questions = st.session_state["questions"]
            processed_files = st.session_state["processed_files"]
            config = st.session_state.get('config', AppConfig(model=st.session_state['model']))
            
            # Display summary information
            st.markdown("### Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions", len(questions))
            with col2:
                st.metric("Documents", len(processed_files))
            with col3:
                st.metric("Model", config.model)
            
            # Create a loading container for answering status
            answering_status_container = UIHelper.create_loading_container()
            
            # Display questions in an expander
            with st.expander("View Questions", expanded=False):
                for i, question in enumerate(questions):
                    st.write(f"{i+1}. {question}")
            
            # Answer and Export buttons
            col1, col2 = st.columns(2)
            with col1:
                answer_button = st.button(
                    "Answer Questions", 
                    disabled=UIHelper.is_processing(self.qa_processing_key),
                    type="primary"
                )
            with col2:
                export_button = st.button(
                    "Export Results",
                    disabled=not st.session_state.get("results") or UIHelper.is_processing(self.export_processing_key)
                )
            
            if answer_button:
                UIHelper.start_processing(self.qa_processing_key)
                UIHelper.show_loading(answering_status_container, "Preparing to answer questions...")
                
                try:
                    # Create vector stores with progress tracking
                    with answering_status_container:
                        st.write("Creating vector stores for each document...")
                        vector_store_progress, vs_value_key, vs_total_key = UIHelper.create_progress_bar(
                            len(processed_files), 
                            "vector_store"
                        )
                        
                        vector_store_map = {}
                        embedding_strategy = StrategyFactory.get_embedding_strategy(config.model)
                        
                        for i, processed_file in enumerate(processed_files):
                            vs_status = st.empty()
                            vs_status.info(f"Creating vector store for {processed_file.name} ({i+1}/{len(processed_files)})")
                            
                            vector_store = embedding_strategy.create_embeddings(processed_file.chunks)
                            if vector_store:
                                vector_store_map[processed_file.name] = vector_store
                                UIHelper.update_progress(
                                    vector_store_progress, 
                                    vs_value_key, 
                                    vs_total_key
                                )
                            else:
                                vs_status.error(f"Failed to create vector store for {processed_file.name}")
                        
                        if not vector_store_map:
                            st.error("Failed to create any vector stores. Cannot proceed.")
                            UIHelper.end_processing(self.qa_processing_key)
                            return
                        
                        st.success(f"Created vector stores for {len(vector_store_map)} documents!")
                        
                        # Answer questions with progress tracking
                        st.write("Answering questions...")
                        qa_progress, qa_value_key, qa_total_key = UIHelper.create_progress_bar(
                            len(questions) * len(vector_store_map), 
                            "qa_progress"
                        )
                        
                        results = []
                        llm_strategy = StrategyFactory.get_llm_strategy(config.model)
                        
                        for i, question in enumerate(questions):
                            qa_status = st.empty()
                            qa_status.info(f"Processing question {i+1}/{len(questions)}: {question}")
                            
                            for file_name, vector_store in vector_store_map.items():
                                file_qa_status = st.empty()
                                file_qa_status.info(f"Answering with document: {file_name}")
                                
                                try:
                                    if config.model == ModelType.GEMINI.value:
                                        answer = LLMService.retry_gemini_call(vector_store, question, config)
                                    else:
                                        answer = llm_strategy.ask_and_get_answer(vector_store, question, config)
                                    
                                    results.append(QuestionAnswer(
                                        question=question,
                                        file_name=file_name,
                                        answer=answer
                                    ))
                                    
                                    UIHelper.update_progress(qa_progress, qa_value_key, qa_total_key)
                                    file_qa_status.empty()
                                except Exception as e:
                                    file_qa_status.error(f"Error with {file_name}: {e}")
                                    results.append(QuestionAnswer(
                                        question=question,
                                        file_name=file_name,
                                        answer=f"Error generating answer"
                                    ))
                                    UIHelper.update_progress(qa_progress, qa_value_key, qa_total_key)
                            
                            qa_status.success(f"Completed question {i+1}/{len(questions)}")
                        
                        st.session_state["results"] = results
                        st.success("All questions answered successfully!")
                    
                    UIHelper.end_processing(self.qa_processing_key)
                    st.rerun()
                except Exception as e:
                    with answering_status_container:
                        st.error(f"Error during question answering: {e}")
                    UIHelper.end_processing(self.qa_processing_key)
            
            # Display results if available
            if "results" in st.session_state and st.session_state["results"]:
                results = st.session_state["results"]
                
                with st.expander("View Results", expanded=True):
                    # Group results by question
                    results_by_question = {}
                    for result in results:
                        if result.question not in results_by_question:
                            results_by_question[result.question] = []
                        results_by_question[result.question].append((result.file_name, result.answer))
                    
                    # Display results for each question
                    for i, (question, answers) in enumerate(results_by_question.items()):
                        st.markdown(f"**Q{i+1}: {question}**")
                        for file_name, answer in answers:
                            with st.container():
                                st.markdown(f"*Document: {file_name}*")
                                st.markdown(f"{answer}")
                            st.markdown("---")
            
            # Export results
            if export_button and st.session_state.get("results"):
                UIHelper.start_processing(self.export_processing_key)
                export_status = UIHelper.create_loading_container()
                
                try:
                    with export_status:
                        with st.spinner("Exporting results to Excel..."):
                            results_file = FileService.export_results(st.session_state["results"])
                            st.success(f"Results exported successfully to {results_file}!")
                    
                    results_file_path = RESULTS_DIR / "questions_answers.xlsx"
                    if results_file_path.exists():
                        with open(results_file_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Results",
                                data=file,
                                file_name="questions_answers.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_results"
                            )
                    
                    UIHelper.end_processing(self.export_processing_key)
                except Exception as e:
                    with export_status:
                        st.error(f"Error exporting results: {e}")
                    UIHelper.end_processing(self.export_processing_key)
            
            st.markdown('</div>', unsafe_allow_html=True)


# Main entry point
if __name__ == "__main__":
    app = ChatWithDocumentsApp()
    app.run()
