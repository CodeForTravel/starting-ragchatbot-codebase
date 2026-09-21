# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for Python dependency management (Python >= 3.13, no separate lockfile-based build/lint/test tooling is configured).

```bash
# Install dependencies
uv sync

# Run the app (from repo root) — starts the backend and serves the frontend
chmod +x run.sh && ./run.sh

# Equivalent manual start (from backend/)
cd backend && uv run uvicorn app:app --reload --port 8000

# Run any one-off Python command in the project's environment
uv run python <script.py>
```

**Always use `uv` to run the server and any Python file (`uv run ...`). Never use `pip` directly, and never run bare `python`/`python3`.** Add dependencies with `uv add`.

The app requires a `GEMINI_API_KEY` in a `.env` file at the repo root (see `.env.example`).

Code quality (black, configured in `pyproject.toml`):

```bash
./scripts/format.sh   # auto-format backend/ and main.py with black
./scripts/check.sh    # black --check + pytest (skips the live smoke test)
```

Run `./scripts/format.sh` before committing. No linter or type-checker is configured.

- Web interface: http://localhost:8000
- API docs (FastAPI auto-generated): http://localhost:8000/docs

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. FastAPI backend + vanilla JS frontend, ChromaDB for vector storage, Google Gemini for generation.

```
frontend/ (static HTML/CSS/JS)  →  backend/app.py (FastAPI)  →  RAGSystem (orchestrator)
```

### Request flow

`POST /api/query` → `RAGSystem.query()` → `AIGenerator.generate_response()` calls Gemini with the `search_course_content` tool available → **if and only if** Gemini judges the question course-specific, it invokes the tool → `ToolManager` executes `CourseSearchTool` against `VectorStore` → results go back to Gemini, which may make a second, dependent tool call (e.g. `get_course_outline` then `search_course_content`) → final answer + collected sources return to the frontend. `generate_response` runs a loop of at most **2 sequential tool rounds** (`AIGenerator.MAX_ROUNDS`), one Gemini request per round; parallel calls in one response count as one round. After 2 rounds, or if a tool raises, one final text-only request (tool calling disabled via mode NONE) produces the answer.

Sources are tracked as a side effect: `CourseSearchTool.last_sources` (a deduplicated list, accumulated across the searches of one query, of `{text, url}` dicts, where `url` is the lesson link from `VectorStore.get_lesson_link`, rendered by the frontend as a new-tab link with no visible URL) is populated during `execute()`, read via `ToolManager.get_last_sources()` after generation, and explicitly reset (`reset_sources()`) each query so they don't leak across turns. This is stateful and easy to break if tools are added/changed without preserving that reset.

### Core backend modules (`backend/`)

- **`app.py`** — FastAPI app; `/api/query` and `/api/courses` endpoints; mounts `frontend/` as static files; on startup, loads all documents from `../docs` into the vector store (skipping courses that already exist by title).
- **`rag_system.py`** — `RAGSystem` wires together document processing, vector storage, AI generation, sessions, and tools. This is the main entry point to understand end-to-end query handling.
- **`document_processor.py`** — Parses course transcript `.txt` files with a specific expected format (see below) and produces sentence-aware, overlapping text chunks (`Config.CHUNK_SIZE` / `CHUNK_OVERLAP`).
- **`vector_store.py`** — Wraps ChromaDB with **two collections**:
  - `course_catalog` — one entry per course (title as ID), used only to fuzzy-resolve a user-provided course name to its exact title via embedding similarity.
  - `course_content` — the actual chunked text, searched with optional `course_title`/`lesson_number` filters.
  - Course name resolution always happens first (`_resolve_course_name`), then content search is filtered by the resolved title — a course name filter that doesn't match anything returns an explicit "no course found" error rather than falling back to unfiltered search.
- **`ai_generator.py`** — Wraps the Google GenAI (`google-genai`) client (model set in `config.py`). Contains the system prompt governing tool-use behavior (search only for course-specific questions, no meta-commentary about searching) and the sequential tool-round loop (`generate_response`, `_execute_calls`, `_final_text`).
- **`search_tools.py`** — `Tool` abstract base class, `CourseSearchTool` (the one registered tool, with tool schema in `get_tool_definition`), and `ToolManager` (registry + source tracking, designed so additional tools can be registered the same way).
- **`session_manager.py`** — Purely in-memory (`Dict[str, List[Message]]`), not persisted; history is capped to `Config.MAX_HISTORY` exchanges per session and formatted as a flat "Role: content" string spliced into the system prompt (not native multi-turn message history).
- **`models.py`** — Pydantic models: `Course`, `Lesson`, `CourseChunk`.
- **`config.py`** — All tunables in one dataclass: `GEMINI_MODEL`, `EMBEDDING_MODEL` (`all-MiniLM-L6-v2`), `CHUNK_SIZE`/`CHUNK_OVERLAP`, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`.

### Course document format

`document_processor.py` expects course `.txt` files (in `docs/`) shaped like:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

The course title is used as the unique ID/primary key across both Chroma collections, and lesson markers (`Lesson N: ...`) drive both chunk boundaries and the lesson metadata attached to each chunk. Changing this parsing logic affects both collections and chunk metadata simultaneously.

### Persisted state

`backend/chroma_db/` is the ChromaDB persistence directory (gitignored, regenerated on startup from `docs/`). Course loading on startup is idempotent by course title, so re-adding files with the same `Course Title:` line is a no-op.
