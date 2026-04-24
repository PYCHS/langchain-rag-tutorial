# LangChain RAG Tutorial Practice

This repository is a hands-on learning project for building a basic Retrieval-Augmented Generation (RAG) workflow with LangChain.

I created this project while preparing for my upcoming Intel GPU Software Development Internship, where I expect to work with LLM-based debugging tools. The goal of this repository is to understand how external documents can be loaded, chunked, embedded, retrieved, and used as context for an LLM-generated answer.

## Project Motivation

Large Language Models are powerful, but they do not automatically know private, internal, or newly updated information. RAG helps solve this problem by retrieving relevant external knowledge at runtime and passing it into the model as context.

For a debugging assistant, this idea is especially important:

- Bug reports may require domain-specific knowledge
- Internal documents may contain previous solutions
- The model should answer based on evidence, not guess confidently
- Retrieval quality directly affects answer quality

This project is my first step toward understanding how to build more reliable LLM applications for technical workflows.

## What This Project Covers

This repository experiments with the core components of a RAG pipeline:

- Document loading
- Text chunking
- Embedding generation
- Vector database storage
- Semantic search
- Retrieval tools
- Agent-based answering with external context

## RAG Pipeline

The basic pipeline is:

```text
Raw Documents
     ↓
Document Loaders
     ↓
Text Splitter
     ↓
Embedding Model
     ↓
Vector Store
     ↓
Retriever / Tool
     ↓
LLM Answer with Retrieved Context


## Repository Structure

```text
langchain-rag-tutorial/
│
├── app.py                  # Main application entry point
├── example.py              # RAG / LangChain practice example
├── test_setup.py           # Environment and setup testing
├── .gitignore              # Files ignored by Git
│
├── loader/
│   ├── bugsDocument.csv    # Sample debugging-related CSV data
│   ├── loaderTest.py       # Loader testing script
│   ├── my_file.pdf         # Sample PDF document
│   ├── pdfLoader.py        # PDF loading practice
│   └── webLoader.py        # Web page loading practice
│
└── README.md
