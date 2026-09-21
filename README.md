# Company AI Tool

## Overview

Company AI Tool is an AI-powered assistant designed to answer questions using information from company documents.

The project uses a **Retrieval-Augmented Generation (RAG)** approach. Company documents are processed and stored as searchable vector data. When a user asks a question, the system retrieves relevant information from the documents and provides it to the AI model to generate a contextual response.

The application is built with **FastAPI** and provides an API that can be tested through Swagger UI.

---

## Purpose

The main purpose of this project is to build a company-focused AI assistant that can:

- Understand user questions in natural language
- Search company documentation for relevant information
- Retrieve the most relevant document content
- Generate contextual AI responses
- Provide the functionality through a REST API

This approach allows the AI assistant to answer questions based on company-specific information rather than relying only on general model knowledge.

---

## How It Works

The application follows a Retrieval-Augmented Generation workflow:

```text
Company Documents
       │
       ▼
Document Processing
       │
       ▼
Text Chunking & Embeddings
       │
       ▼
ChromaDB Vector Database
       │
       ▼
User Question
       │
       ▼
Similarity Search
       │
       ▼
Relevant Document Context
       │
       ▼
AI Model
       │
       ▼
Generated Response

Project Structure
company-ai-tool/
│
├── app/
│   ├── api/
│   │   └── API routes and endpoints
│   │
│   ├── schemas/
│   │   └── Request and response models
│   │
│   ├── services/
│   │   └── Business logic and AI/RAG services
│   │
│   └── ...
│
├── documents/
│   └── Company documents used by the AI assistant
│
├── chroma_db/
│   └── Local vector database storage
│
├── .gitignore
│   └── Files and folders excluded from Git
│
├── requirements.txt
│   └── Python dependencies
│
├── pyproject.toml
│   └── Python project configuration
│
└── README.md
    └── Project documentation




<img width="1917" height="1023" alt="image" src="https://github.com/user-attachments/assets/668c780e-63e2-4b13-a76c-004d11741fa3" />
<img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/531f793e-0209-406e-8f9e-8ec85a641444" />
<img width="1917" height="991" alt="image" src="https://github.com/user-attachments/assets/08dfcbfb-c132-4911-9b0b-5162d684e97a" />

