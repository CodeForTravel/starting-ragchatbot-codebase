# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Google Gemini for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- A Gemini API key (for Gemini AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Running the Tests

Tests live in `backend/tests/` and use `pytest` (installed as a dev dependency via `uv sync`).

```bash
cd backend
uv run pytest tests -v
```

Run a single file or test:
```bash
uv run pytest tests/test_rag_system.py -v
uv run pytest tests -k "float_lesson"
```

Notes:
- Most tests use a fake Gemini client and a temporary ChromaDB, so they need no API key (the embedding model must be available locally).
- `tests/test_live_smoke.py` calls the real Gemini API and is skipped automatically unless `GEMINI_API_KEY` is set in `.env`.


Diagram of chat flow from fronend to backend
https://claude.ai/artifact/18BhpU3UZmhQfNKxsLUy5Q