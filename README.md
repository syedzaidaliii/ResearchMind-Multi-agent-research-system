# ResearchMind: Multi-Agent Research System

ResearchMind turns a topic into a structured research report. It searches the web, reads several pages, retrieves the most relevant passages, writes a cited report, and then reviews its own work. You can use it from a Streamlit web app or straight from the terminal.

It runs entirely on free tiers: Google Gemini for the language model and Tavily for web search.

## How it works

ResearchMind uses retrieval-augmented generation (RAG). Instead of asking a model to read one page, it reads several pages, keeps only the passages that matter, and forces the writer to cite them.

| Step | What it does |
|------|--------------|
| 1. Search | Tavily returns up to 6 sources for the topic. |
| 2. Read pages | The pages are downloaded in parallel and cleaned. If a page cannot be read, the search snippet is used instead. |
| 3. Index | The text is split into overlapping chunks, embedded with Gemini embeddings, and stored in a fresh Chroma collection. |
| 4. Retrieve | The most relevant chunks are selected with MMR (maximal marginal relevance), which avoids returning the same point several times. |
| 5. Write | Gemini writes the report using only the retrieved chunks and cites each claim as [n]. The Sources section is built from the sources actually cited. |
| 6. Critic | A second prompt reviews the report and returns a score out of 10 with feedback. |

```
topic -> search -> read pages -> chunk + embed -> Chroma -> retrieve (MMR) -> writer [n] -> critic
```

### Safeguards against made-up answers

- The writer sees only the retrieved excerpts and is told to say when the sources do not cover something.
- After writing, the code checks every [n] against the real source list. Citations to sources that do not exist are flagged in the UI, and the Sources section lists only sources that were really cited.
- Page fetching refuses local and private network addresses, limits redirects and page size, and ignores non-HTML content.

## Features

- Streamlit interface with a live six-step progress tracker
- Metrics for critic score, sources cited and time taken
- Tabs for the report, critic feedback, the sources that were read, and the exact passages the writer was given
- Download of the finished report as a Markdown file
- Clear error messages for rate limits and API key problems
- Automatic retry with backoff when the embedding API is rate limited
- Falls back to an in-memory vector store if Chroma cannot start
- Offline tests and a setup check script
- Free to run: no paid API required

## Requirements

- Python 3.10 or newer (developed on Python 3.12)
- LangChain 1.0 or newer (the code uses `create_agent`)
- A free Google Gemini API key
- A free Tavily API key

## Setup

### 1. Get the code and create a virtual environment

```bash
git clone <your-repository-url>
cd Multi-agent-research-system-main

python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Get your API keys

- **Gemini:** open [Google AI Studio](https://aistudio.google.com), choose "Get API key", and create a key in a project that has no billing attached. The free tier does not require a payment method.
- **Tavily:** sign up at [tavily.com](https://tavily.com) and copy your API key from the dashboard. The free plan needs no card.

### 4. Create the `.env` file

Create a file named `.env` in the project root, in the same folder as `agents.py`:

```
GOOGLE_API_KEY=your_google_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Do not put the file inside `.venv`, do not add quotes you do not need, and never commit it to Git. Make sure your `.gitignore` contains:

```
.env
.venv/
__pycache__/
```

## Usage

### Web app

```bash
streamlit run app.py
```

If your terminal says the `streamlit` command is not recognized, run it through Python instead:

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`. Type a topic, press **Run research**, and watch the steps complete. When it finishes you can read the report, check the critic's feedback, inspect the raw research and download the report.

### Terminal

```bash
python pipeline.py
```

Enter a topic when prompted. The report, critic feedback and a citation summary are printed.

## Check your setup

```bash
python check_setup.py
```

This tests your keys, Tavily, the Gemini chat model, the embedding model and the vector store one by one, so you can see exactly what is misconfigured.

## Tests

```bash
python -m unittest discover tests
```

The tests need no API keys or internet. They cover chunking, text extraction, the private-address block, MMR retrieval, embedding retries and citation checking.

## Project structure

```
.
|-- app.py            Streamlit user interface
|-- pipeline.py       Runs the six steps (used by the UI and the terminal)
|-- rag.py            Search, scraping, chunking, embeddings, vector store, retrieval, citations
|-- agents.py         Gemini model setup and the critic prompt (older agent code is kept)
|-- tools.py          Older LangChain tools (no longer used by the pipeline)
|-- check_setup.py    Connectivity check
|-- tests/            Offline unit tests
|-- requirements.txt  Python dependencies
|-- .env              Your API keys (not committed)
`-- README.md
```

## Configuration

| What | Where | Default |
|------|-------|---------|
| Chat model | `agents.py`, the `ChatGoogleGenerativeAI(...)` line | `gemini-3.5-flash-lite`. Copy the current free-tier name from Google AI Studio if you see "model not found". |
| Embedding model | `EMBEDDING_MODEL` in `.env` | `gemini-embedding-001` |
| Sources per search | `RAG_MAX_RESULTS` in `.env` | 6 |
| Vector store | `VECTOR_STORE` in `.env` (`chroma` or `memory`) | `chroma` |
| Chunk size, overlap, top-k, MMR balance | constants at the top of `rag.py` | 1000, 150, 10, 0.6 |

## Troubleshooting

**API key required error on startup.** The `.env` file was not found or does not contain `GOOGLE_API_KEY`. Check that it sits next to `agents.py`, that it is named exactly `.env` (not `.env.txt`), and that you saved it.

**Rate limit or 429 error.** The Gemini free tier has low request limits, and one run makes several model calls. Wait a minute and try again.

**Embedding model not found.** Set `EMBEDDING_MODEL` in `.env` to a current embedding model from Google AI Studio, then run `python check_setup.py`.

**Model not found error.** The model name in `agents.py` is no longer available on the free tier. Choose a current Flash or Flash-Lite model in Google AI Studio and update the name.

**"None of the pages could be read".** Every site blocked scraping. Try another topic or run it again.

**Payment prompt in Google's console.** You started billing setup, which is not needed. Create the API key in a project without billing instead.

**The `streamlit` command is not recognized.** Use `python -m streamlit run app.py`, and make sure you installed the requirements with the same Python.

## Limitations

- Reports are only as good as the pages found. Always check the listed sources, especially for breaking news.
- Some websites block scraping or need JavaScript. Those pages fall back to the short search snippet.
- The critic's score is shown but not used to revise the report yet.
- The embedding model name and free-tier limits can change without notice.
- The code has been unit tested offline. Run `python check_setup.py` to confirm it works with your own keys.

## Roadmap

- Planner that splits a topic into sub-questions answered in parallel
- Groundedness check that verifies each claim against the retrieved text, with an automatic rewrite loop
- Evaluation script with real metrics on a fixed set of topics
- Docker image, public demo and a Groq fallback model

## Built with

- [LangChain](https://www.langchain.com) for prompts and model wrappers
- [Chroma](https://www.trychroma.com) for the vector store
- [Google Gemini](https://ai.google.dev) for the language model
- [Tavily](https://tavily.com) for web search
- [Streamlit](https://streamlit.io) for the user interface
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) and Requests for scraping
