"""
Q/A with Documents Application

A Streamlit application that allows users to upload documents, process them,
and ask questions using various LLM models.
"""

import os
import time
from datetime import datetime
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
DEFAULT_SYSTEM_PROMPT = """You are an expert automated analyst designed to systematically extract and interpret sustainability and human rights information from corporate sustainability reports. Your primary function is to analyze provided texts (sustainability reports) and respond accurately and concisely to a set of predefined questions, focusing explicitly on assessing aspects of sustainability, human rights maturity, policy commitments, GRI Standards adherence, and relevant international guidelines mentioned.

When analyzing reports, adhere strictly to these rules:

1. Direction and Persona:
- Operate as a meticulous, data-driven analyst trained in sustainability reporting, international human rights standards, corporate responsibility frameworks (e.g., GRI Standards, OECD Guidelines, UN Global Compact, ISO protocols), and assessment methodologies.
- Only answer based on information explicitly present in the sustainability report provided. Do NOT infer or assume responses if explicit statements or evidence are not available.
- When answering, clearly state "Yes" or "No", briefly cite relevant sentences (verbatim excerpts are preferred) from the report as evidence, and indicate the report page or section if available. If explicitly requested information is not present, respond clearly with: "Information not explicitly disclosed."

2. Response Format:
Use the following structured response format for each question separately:

Question: {{question}}
Answer: Yes/No/Information not explicitly disclosed
Evidence: "{{exact sentence or passage from report text}}"
Location: {{page number or section of report document; write "not specified" if unavailable}}

3. Robustness for Varied Questions:
- You will be provided questions that concern specific human rights practices, due diligence processes, stakeholder involvement, adherence to GRI criteria, UN standards, OECD Guidelines, and other standards. Questions may include but are not limited to explicit mentions of policies, processes, certifications, trainings, governance, reporting standards adherence, and international initiatives (e.g., UNGC, OECD, ISO standards, SA8000, Bloomberg GEI, Valore D, Ethical Trading Initiative, BSCI Amfori, FLA, Principles for Responsible Investment, Human Rights Indicators for Business, Universal Declaration of Human Rights, etc.)

- Be particularly attentive to precise wording, standards codes (GRI 414.1, GRI 414.2, 202.1, 405.1, 405.2, etc.), and listed alternative keywords (e.g., discrimination, child labor, gender equality, forced labor, etc.) and always carefully check for explicitly matching terms or codes in the provided report text.

4. Handling Ambiguities:
- Answer "Yes" only if there is explicit clarity in the sustainability report that the question's criteria are explicitly met (e.g., explicit mention of adherence, certification, practices, or standards).
- Short, unclear, indirect, or vague references without explicit compliance statements must result in "Information not explicitly disclosed."

Examples (illustrative):

EXAMPLE 1:
Question: Does the company have SA 8000 certification?
Answer: Yes
Evidence: "Our factories have received SA 8000 certification, demonstrating a commitment to socially acceptable practices in our facilities."
Location: Page 23, Section "Certifications & Commitments."

EXAMPLE 2:
Question: Does the company consider material under GRI the Topic 405.2: Ratio of basic salary and remuneration of women to men?
Answer: Information not explicitly disclosed
Evidence: "We analyzed gender ratios at various job levels."
Location: Page 44, Section "Diversity and Inclusion."

EXAMPLE 3:
Question: Does the company have a human rights due diligence (internal and external) process that involves potentially impacted stakeholders?
Answer: Yes
Evidence: "We maintain an ongoing due diligence process, including consultation sessions with potentially affected stakeholders and community representatives, both internally and externally."
Location: Page 17, Section "Human Rights Management Approach."

Your goal is to reliably capture explicit evidence to objectively support numeric scoring in corporate sustainability evaluations related to sustainability and human rights maturity indexes.

Now, evaluate the provided sustainability report and answer the following:

Question: {{question}}
Answer:
Evidence:
Location:
"""

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_TEMPERATURE = 0.3
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
    AB_TESTING = 5


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


@dataclass
class ABTestData:
    variant: str  # "A" or "B"
    prompt: str   # The question that was asked
    file_name: str  # The file name that was used
    response: str  # The response generated
    feedback: Optional[int] = None  # 1 for thumbs up, 0 for thumbs down


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
            st.markdown(
                f"""
                <div class="custom-spinner">
                    <div class="spinner"></div>
                    <div class="message">{message}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
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

        _system_prompt = f"{config.system_prompt} context: {{context}}"
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

        _system_prompt = f"{config.system_prompt} context: {{context}}"
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
    def read_ab_testing_excel(uploaded_file) -> pd.DataFrame:
        """Read and process an Excel file."""
        cols = [
            'Question', 'File', 'Answer'
        ]
        try:
            framework = pd.read_excel(uploaded_file, sheet_name='Sheet1', usecols=cols).fillna(method='ffill')
            return framework
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return pd.DataFrame()

    @staticmethod
    def export_results(results: List[QuestionAnswer], filename: str = "questions_answers.xlsx", metadata: dict = None) -> str:
        """Export the results to an Excel file."""
        results_dir = RESULTS_DIR
        results_dir.mkdir(exist_ok=True)
        
        # Convert results to a DataFrame
        data = [{"Question": qa.question, "File": qa.file_name, "Answer": qa.answer} for qa in results]
        df = pd.DataFrame(data)
        
        # Save to Excel with metadata in a separate sheet if available
        file_path = results_dir / filename
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            
            # Export metadata to Sheet2 if available
            if metadata:
                metadata_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])
                metadata_df.to_excel(writer, sheet_name='Sheet2', index=False)
        
        return str(file_path)
        
    @staticmethod
    def export_ab_test_results(results: List[ABTestData], filename: str = "ab_test_results.xlsx") -> str:
        """Export the A/B testing results to an Excel file."""
        results_dir = RESULTS_DIR
        results_dir.mkdir(exist_ok=True)
        
        # Convert results to a DataFrame
        df = pd.DataFrame([{
            "Variant": data.variant,
            "Question": data.prompt,
            "Answer": data.response,
            "Feedback": data.feedback
        } for data in results])
        
        # Create a summary DataFrame
        if not df.empty and 'Feedback' in df.columns and df['Feedback'].notna().any():
            summary_df = (
                df.groupby("Variant")
                .agg(
                    count=("Feedback", "count"),
                    score=("Feedback", "mean")
                )
                .reset_index()
            )
            
            # Save both DataFrames to different sheets in the same Excel file
            file_path = results_dir / filename
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Detailed Results', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        else:
            # If no feedback data, just save the main DataFrame
            file_path = results_dir / filename
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
            print('gemini log', answer)
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
        """Initialize the session state variables."""
        if 'step' not in st.session_state:
            st.session_state.step = Step.UPLOAD_FILES
        
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        
        if 'questions' not in st.session_state:
            st.session_state.questions = []
        
        if 'config' not in st.session_state:
            st.session_state.config = AppConfig(
                model=ModelType.GPT_35_TURBO,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                temperature=DEFAULT_TEMPERATURE,
                k=DEFAULT_K
            )
        
        if 'answers' not in st.session_state:
            st.session_state.answers = []
        
        # Initialize A/B testing related state
        if 'ab_test_data' not in st.session_state:
            st.session_state.ab_test_data = []
        
        if 'ab_test_excel_files' not in st.session_state:
            st.session_state.ab_test_excel_files = {'A': None, 'B': None}
        
        if 'ab_test_responses' not in st.session_state:
            st.session_state.ab_test_responses = []
        
        if 'ab_test_current_index' not in st.session_state:
            st.session_state.ab_test_current_index = 0
        
        if 'ab_test_completed' not in st.session_state:
            st.session_state.ab_test_completed = False

    

    def run(self):
        """Run the application."""
        # Add a nice header with styling
        st.markdown("""
        <style>
        .app-header {
            background: linear-gradient(135deg, #2E8B57 0%, #98FB98 50%, #2E8B57 100%);
            color: #006400;
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
            <h3>Sustainability Report Analyzer: Automated Assessment of Corporate Environmental & Social Performance</h1>
            <p>Leverage AI to systematically analyze sustainability reports, extract key metrics, and generate standardized evaluations of corporate sustainability and human rights practices</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the current step as a progress indicator
        current_step = st.session_state["step"]
        steps = ["1. Upload Files", "2. Upload Excel", "3. Configure", "4. Answer Questions"]
        
        cols = st.columns(len(steps))
        for i, step_name in enumerate(steps):
            with cols[i]:
                if i + 1 < current_step:
                    st.markdown(f"✅ **{step_name}**")
                elif i + 1 == current_step:
                    st.markdown(f"🔄 **{step_name}**")
                else:
                    st.markdown(f"⏳ {step_name}")
        # Display navigation
        nav_selection = st.sidebar.radio(
            "Navigation",
            ["Home", "Q/A with Documents", "A/B Testing"],
            index=0,
        )
        
        if nav_selection == "Home":
            st.session_state.step = Step.UPLOAD_FILES
            self._show_home_page()
        elif nav_selection == "Q/A with Documents":
            # Show the appropriate step
            if st.session_state.step == Step.UPLOAD_FILES:
                self._handle_upload_files_step()
            elif st.session_state.step == Step.UPLOAD_EXCEL:
                self._handle_upload_excel_step()
            elif st.session_state.step == Step.CONFIGURE:
                self._handle_configure_step()
            elif st.session_state.step == Step.ANSWER_QUESTIONS:
                self._handle_answer_questions_step()
        elif nav_selection == "A/B Testing":
            st.session_state.step = Step.AB_TESTING
            self._handle_ab_testing_step()

    def _show_home_page(self):
        """Show the home page."""
        st.markdown("""
        # Welcome to Q/A with Documents
        
        This application allows you to Q/A with your documents using OpenAI language models.
        
        ## Features:
        - Upload PDF, DOCX, and TXT files
        - Break documents into smaller chunks for efficient processing
        - Configure system prompt, chunk size, and other parameters
        - Ask questions about your documents
        - Compare different system prompts with A/B testing
        
        Get started by navigating to "Q/A with Documents" or "A/B Testing" in the sidebar.
        """)

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
                disabled=len(uploaded_files) == 0 or st.session_state.get("proceed_button", False),
                type="primary",
                key="proceed_button"
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
            # st.markdown('<div class="step-container">', unsafe_allow_html=True)
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
                disabled=not bool(excel_file) or st.session_state.get("proceed_button2", False),
                type="primary",
                key="proceed_button2"
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
            # st.markdown('<div class="step-container">', unsafe_allow_html=True)
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
                    help="Size of text chunks in characters. Smaller chunks may improve accuracy but increase processing time."
                )
                
                chunk_overlap = st.number_input(
                    "Chunk overlap:", 
                    min_value=0, 
                    max_value=100, 
                    value=DEFAULT_CHUNK_OVERLAP,
                    help="Percentage of overlap between chunks to maintain context across chunk boundaries."
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
                    help="Number of most relevant chunks to retrieve for each question."
                )
            
            # Create a loading container for processing status
            processing_status_container = UIHelper.create_loading_container()
            
            # Process and Next buttons
            col1, col2 = st.columns([9.9, 2.1])
            with col1:
                process_button = st.button(
                    "Process Files", 
                    disabled=not api_key or st.session_state.get("process_config_button", False),
                    type="primary",
                    key="process_config_button"
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
            # st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.subheader("Answer Questions and Export Results")
            
            # Add a back button
            # col1, col2 = st.columns([1, 11])
            # with col1:
            #     if st.button("⬅️ Back", key="back_to_config"):
            #         st.session_state["step"] = Step.CONFIGURE
            #         st.rerun()
            
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
                st.metric("Chunk Size", config.chunk_size)
            with col2:
                st.metric("Documents", len(processed_files))
                st.metric("Chunk Overlap", config.chunk_overlap)
            with col3:
                st.metric("Model", config.model)
                st.metric("Temperature", config.temperature)
            
            # Create a loading container for answering status
            answering_status_container = UIHelper.create_loading_container()
            
            # Display questions in an expander
            with st.expander("View Questions", expanded=False):
                for i, question in enumerate(questions):
                    st.write(f"{i+1}. {question}")
            
            # Answer and Export buttons
            col1, col2 = st.columns([9.5, 2.5])
            with col1:
                answer_button = st.button(
                    "Answer Questions", 
                    disabled=st.session_state.get("answer_button", False),
                    type="primary",
                    key="answer_button"
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
                        
                        for file_name, vector_store in vector_store_map.items():
                            file_status = st.empty()
                            file_status.info(f"Processing document: {file_name}")
                            
                            for i, question in enumerate(questions):
                                qa_status = st.empty()
                                qa_status.info(f"Processing question {i+1}/{len(questions)} for {file_name}: {question}")
                                
                                try:
                                    if config.model == ModelType.GEMINI.value:
                                        answer = LLMService.retry_gemini_call(vector_store, question, config)
                                    else:
                                        answer = llm_strategy.ask_and_get_answer(vector_store, question, config)
                                    
                                    results.append(QuestionAnswer(
                                        question=question,
                                        file_name=file_name,
                                        answer=answer['answer']
                                    ))
                                    
                                    UIHelper.update_progress(qa_progress, qa_value_key, qa_total_key)
                                    qa_status.success(f"Completed question {i+1}/{len(questions)} for {file_name}")
                                except Exception as e:
                                    qa_status.error(f"Error with question for {file_name}: {e}")
                                    results.append(QuestionAnswer(
                                        question=question,
                                        file_name=file_name,
                                        answer=f"Error generating answer {e}"
                                    ))
                                    UIHelper.update_progress(qa_progress, qa_value_key, qa_total_key)
                            
                            file_status.success(f"Completed all questions for document: {file_name}")
                        
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
                                st.markdown(f"<pre>{answer}</pre>", unsafe_allow_html=True)
                            st.markdown("---")
            
            # Export results
            if export_button and st.session_state.get("results"):
                UIHelper.start_processing(self.export_processing_key)
                export_status = UIHelper.create_loading_container()
                config = st.session_state.get('config', AppConfig(model=st.session_state['model']))
                try:
                    with export_status:
                        with st.spinner("Exporting results to Excel..."):
                            # Get metadata for export
                            metadata = {
                                "Model": config.model,
                                "System Prompt": config.system_prompt,
                                "Chunk Size": config.chunk_size,
                                "Chunk Overlap": config.chunk_overlap,
                                "Temperature": config.temperature,
                                "K":config.k,
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            results_file = FileService.export_results(st.session_state["results"], metadata=metadata)
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

    def _handle_ab_testing_step(self):
        """Handle the A/B testing step."""
        st.title("A/B Testing for System Prompts")
        
        # Reset button
        if st.sidebar.button("Reset A/B Testing"):
            # Clear A/B testing state
            st.session_state.ab_test_data = []
            st.session_state.ab_test_excel_files = {'A': None, 'B': None}
            st.session_state.ab_test_responses = []
            st.session_state.ab_test_current_index = 0
            st.session_state.ab_test_completed = False
            st.rerun()
        
        # Information about expected file format
        st.info("""
        ## How to Use A/B Testing
        
        1. **Prepare two Excel files** with answers generated using different system prompts
           - Each file should have at least 'Question' and 'Answer' columns
           - You can use the files exported from the "Q/A with Documents" feature
           
        2. **Upload both files** below and click "Process Files"
        
        3. **Evaluate the responses** by giving thumbs up or down to each answer
           - The responses are shuffled so you won't know which system prompt generated them
           
        4. **Compare the results** to see which system prompt performed better
        """)
        
        # Step 1: Upload Excel files and set system prompts
        st.header("Step 1: Upload Excel Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Variant A")
            uploaded_file_a = st.file_uploader("Upload Excel File A (with Questions & Answers)", type=["xlsx", "xls"], key="file_uploader_a")
            
            if uploaded_file_a is not None:
                st.session_state.ab_test_excel_files['A'] = uploaded_file_a
                st.success(f"File uploaded: {uploaded_file_a.name}")
        
        with col2:
            st.subheader("Variant B")
            uploaded_file_b = st.file_uploader("Upload Excel File B (with Questions & Answers)", type=["xlsx", "xls"], key="file_uploader_b")
            
            if uploaded_file_b is not None:
                st.session_state.ab_test_excel_files['B'] = uploaded_file_b
                st.success(f"File uploaded: {uploaded_file_b.name}")
        
        # Step 2: Process files
        st.header("Step 2: Process Files")
        
        if (st.session_state.ab_test_excel_files['A'] is not None and 
            st.session_state.ab_test_excel_files['B'] is not None):
            
            if st.button("Process Files", key="process_files_button"):
                with st.spinner("Processing files..."):
                    # Process the Excel files
                    self._process_ab_test_files()
                    
                    if len(st.session_state.ab_test_responses) > 0:
                        st.success(f"Successfully processed {len(st.session_state.ab_test_responses)} responses!")
                        st.session_state.ab_test_current_index = 0
                        st.rerun()
        else:
            st.warning("Please upload both Excel files first.")
        
        # Step 3: Provide feedback
        if len(st.session_state.ab_test_responses) > 0:
            st.header("Step 3: Provide Feedback")
            
            current_index = st.session_state.ab_test_current_index
            
            if current_index < len(st.session_state.ab_test_responses):
                response_data = st.session_state.ab_test_responses[current_index]
                st.subheader("Question/Prompt")
                st.info(f"File: {response_data.file_name}")
                st.info(response_data.prompt)
                
                st.subheader("Response")
                st.write(response_data.response)
                
                st.write(f"Response {current_index + 1} of {len(st.session_state.ab_test_responses)}")
                
                if st.button("👍 Thumbs Up", key="thumbs_up"):
                    self._provide_feedback(current_index, 1)
                if st.button("👎 Thumbs Down", key="thumbs_down"):
                    self._provide_feedback(current_index, 0)
            else:
                st.success("A/B testing completed!")
                st.session_state.ab_test_completed = True
        
        # Step 4: View results
        if st.session_state.ab_test_completed:
            st.header("Step 4: View Results")
            
            # Calculate summary statistics
            if len(st.session_state.ab_test_data) > 0:
                # Convert to DataFrame for analysis
                df = pd.DataFrame([{
                    "Variant": data.variant,
                    "Prompt": data.prompt,
                    "Response": data.response,
                    "Feedback": data.feedback
                } for data in st.session_state.ab_test_data])
                
                # Create summary DataFrame
                summary_df = (
                    df.groupby("Variant")
                    .agg(
                        count=("Feedback", "count"),
                        score=("Feedback", "mean")
                    )
                    .reset_index()
                )
                
                st.subheader("Summary Results")
                st.dataframe(summary_df)
                
                # Create a bar chart to visualize the results
                st.subheader("A/B Test Results")
                st.bar_chart(summary_df.set_index("Variant")["score"])
                # Export results option
                if st.button("Export Results to Excel"):
                    results_path = FileService.export_ab_test_results(st.session_state.ab_test_data)
                    st.success(f"Results exported to {results_path}")
                    
                    # Add download button for the exported file
                    results_file_path = Path(results_path)
                    if results_file_path.exists():
                        with open(results_file_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Results",
                                data=file,
                                file_name="ab_test_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_ab_test_results"
                            )
            else:
                st.warning("No feedback data available.")

    def _process_ab_test_files(self):
        """Process the uploaded Excel files for A/B testing."""
        # Load both Excel files
        file_a = st.session_state.ab_test_excel_files['A']
        file_b = st.session_state.ab_test_excel_files['B']
        
        try:
            df_a = FileService.read_ab_testing_excel(file_a)
            df_b = FileService.read_ab_testing_excel(file_b)
            
            # Make sure both dataframes have 'Question' and 'Answer' columns (matching export format)
            required_columns = ['Question', 'File', 'Answer']
            
            for variant, df in [('A', df_a), ('B', df_b)]:
                # Convert column names to title case for consistency
                df.columns = [col.title() if isinstance(col, str) else col for col in df.columns]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Excel file {variant} is missing required columns: {', '.join(missing_columns)}. "
                           f"The file should have columns named 'Question' and 'Answer' like the export format.")
                    return
            
            # Clear existing responses
            st.session_state.ab_test_responses = []
            st.session_state.ab_test_data = []
            
            # Process answers from both files for A/B testing
            for variant, df in [('A', df_a), ('B', df_b)]:
                # Get rows with non-empty answers
                valid_rows = df[df['Answer'].notna()]
                
                for _, row in valid_rows.iterrows():
                    # Create test data using answers from the files
                    test_data = ABTestData(
                        variant=variant,
                        prompt=row['Question'],  # This is the question that was asked
                        file_name=row['File'],
                        response=row['Answer']   # This is the answer we're evaluating
                    )
                    
                    st.session_state.ab_test_responses.append(test_data)
                    st.session_state.ab_test_data.append(test_data)
            
            # Shuffle the responses for blind testing
            import random
            random.shuffle(st.session_state.ab_test_responses)
            
            # If no valid responses were found
            if not st.session_state.ab_test_responses:
                st.error("No valid answers found in the uploaded Excel files. "
                       "Make sure they contain 'Question' and 'Answer' columns with data.")
            
        except Exception as e:
            st.error(f"Error processing Excel files: {str(e)}")

    def _provide_feedback(self, index, feedback):
        """Provide feedback (thumbs up/down) for a response."""
        # Update the response in the responses list
        st.session_state.ab_test_responses[index].feedback = feedback
        
        # Find the corresponding data in ab_test_data and update it
        response_data = st.session_state.ab_test_responses[index]
        for i, data in enumerate(st.session_state.ab_test_data):
            if (data.variant == response_data.variant and 
                data.prompt == response_data.prompt and 
                data.response == response_data.response):
                st.session_state.ab_test_data[i].feedback = feedback
                break
        
        # Move to the next response
        st.session_state.ab_test_current_index += 1
        
        # Check if we've reached the end
        if st.session_state.ab_test_current_index >= len(st.session_state.ab_test_responses):
            st.session_state.ab_test_completed = True
        
        st.rerun()


# Main entry point
if __name__ == "__main__":
    app = ChatWithDocumentsApp()
    app.run()
