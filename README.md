# 📄 AI-Powered Sustainability Report Analyzer

## 🌍 Introduction

In recent years, the rise of corporate sustainability and human rights reporting has brought significant challenges in analyzing these documents consistently. Traditional methods, often manual and time-consuming, struggle to manage the complexity and diversity of these comprehensive reports—leading to inefficiencies and inconsistencies.

This thesis explores a solution using advanced **Prompt Engineering** techniques with **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** to streamline and improve the analysis process.

The primary objective is to build a system that:
- Utilizes LLMs to process and review corporate sustainability reports.
- Applies advanced prompt engineering strategies such as **Chain-of-Thought (CoT)** and **response formatting**.
- Automatically extracts and analyzes key information across different document formats.

---

## ⚙️ Methodology

The system leverages a **Retrieval-Augmented Generation (RAG)** approach and follows a step-by-step pipeline:

1. **Document Upload**  
   Users upload sustainability reports in PDF, DOCX, or TXT format.

2. **Chunking**  
   The system splits documents into manageable **text chunks** using character-based methods—preserving context and improving relevance during retrieval.

3. **Embedding Generation**  
   Chunks are transformed into **vector embeddings** using models from **OpenAI** and **Google AI**.

4. **Vector Store**  
   These embeddings are stored in a **vector database** for fast, semantic search and retrieval.

5. **Prompt Structuring**  
   Prompts are carefully designed to extract useful insights, improving the quality of answers from LLMs.

6. **User Interface (UI)**  
   - Upload and manage documents  
   - Configure model settings and hyperparameters  
   - Import structured spreadsheet prompts  
   - Export results as Excel files

This user-friendly interface supports structured workflows, making it easier to track compliance, evaluate results, and compare across different LLMs and embedding techniques.

---

## ✅ Evaluation

The system was evaluated on multiple performance dimensions, including:

- **Semantic Similarity**
- **Response Format Compliance**
- **Accuracy**

Key findings:
- **Gemini-1.5-Flash-8B** and **GPT-4-Turbo** delivered the most consistent performance.
- Prompt design, chunking strategies, and system instructions significantly influenced output quality.
- Model performance can vary depending on document characteristics, making **adaptive configuration essential**.

---

## 🚀 Conclusion & Future Work

This research demonstrates how LLMs and prompt engineering can automate and improve sustainability report analysis. Key benefits include:

- Faster, more reliable evaluations
- Scalable architecture for document processing
- Structured outputs for auditing and comparison

**Future improvements could include:**
- Smarter chunking strategies
- Multilingual support
- Automated evaluation pipelines

This project lays the foundation for **AI-driven, scalable, and objective analysis** in corporate sustainability reporting—moving away from manual processes toward an intelligent, efficient future.
