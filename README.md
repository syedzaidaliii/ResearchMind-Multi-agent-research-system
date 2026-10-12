# ResearchMind: Multi-Agent Research System

ResearchMind turns a topic into a structured research report. A small team of AI steps searches the web, reads a source, writes the report, and then reviews its own work. You can use it from a Streamlit web app or straight from the terminal.

It runs entirely on free tiers: Google Gemini for the language model and Tavily for web search.

## How it works

The project runs four steps in order. Each step passes its output to the next one.

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | Search Agent | Calls the Tavily web search tool and returns titles, URLs and snippets for the topic. |
| 2 | Reader Agent | Picks the most relevant URL from the search results and scrapes the page text. |
| 3 | Writer | Combines the search results and scraped text into a report with an introduction, key findings, a conclusion and a source list. |
| 4 | Critic | Reviews the report and returns a score out of 10, strengths, areas to improve and a one-line verdict. |

```
topic --> Search Agent --> Reader Agent --> Writer --> Critic --> report + feedback
          (Tavily)         (web scraper)    (Gemini)   (Gemini)
```

The search agent and reader agent are tool-using agents built with LangChain. The writer and critic are prompt chains with no tools.

### Safeguard against made-up answers

Language models sometimes skip the search tool and answer from memory, which produces confident reports with no sources. The pipeline checks that the search agent really called the web search tool. If it did not, the pipeline retries once with a stricter instruction. If it still did not search, the run stops with a clear error instead of producing an unsupported report.

## Features

- Streamlit web interface with a live four-step progress tracker
- Critic score, time taken, and tabs for the report, the critic feedback and the raw research
- Download of the finished report as a Markdown file
- Clear error messages for rate limits and API key problems
- Terminal mode for quick runs without the browser
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

Enter a topic when prompted. Each step prints its output as it finishes.

## Project structure

```
.
|-- app.py            Streamlit user interface
|-- pipeline.py       Runs the four steps in order (used by the UI and the terminal)
|-- agents.py         Model setup, the two agents, and the writer and critic prompts
|-- tools.py          web_search (Tavily) and scrape_url (requests + BeautifulSoup)
|-- requirements.txt  Python dependencies
|-- .env              Your API keys (not committed)
`-- README.md
```

### Key functions

- `run_research_steps(topic)` in `pipeline.py` is a generator that yields an event when each step starts and finishes. The web app uses it to update the progress tracker live.
- `run_research_pipeline(topic)` in `pipeline.py` runs every step, prints the output and returns the final results as a dictionary with the keys `search_results`, `scraped_content`, `report` and `feedback`.

## Configuration

| What | Where | Notes |
|------|-------|-------|
| Language model | `agents.py`, the `ChatGoogleGenerativeAI(...)` line | Currently `gemini-3.5-flash-lite`. Free-tier model names change over time, so copy the current name from Google AI Studio if you see a "model not found" error. |
| Number of search results | `tools.py`, `max_results` in `web_search` | Default is 5. |
| Scraped text length | `tools.py`, the `[:3000]` slice in `scrape_url` | Default is 3000 characters. |
| Report and critic format | `agents.py`, `writer_prompt` and `critic_prompt` | The score parser in `app.py` expects the critic to start with `Score: X/10`. |

## Troubleshooting

**API key required error on startup.** The `.env` file was not found or does not contain `GOOGLE_API_KEY`. Check that it sits next to `agents.py`, that it is named exactly `.env` (not `.env.txt`), and that you saved it.

**Rate limit or 429 error.** The Gemini free tier has low request limits, and one run makes several model calls. Wait a minute and try again.

**Model not found error.** The model name in `agents.py` is no longer available on the free tier. Choose a current Flash or Flash-Lite model in Google AI Studio and update the name.

**"The search agent did not run a web search" error.** The model skipped the search tool twice. Run the topic again. If it keeps happening, check that your Tavily key is valid.

**Payment prompt in Google's console.** You started billing setup, which is not needed. Create the API key in a project without billing instead.

**The `streamlit` command is not recognized.** Use `python -m streamlit run app.py`, and make sure you installed the requirements with the same Python.

## Limitations

- This is a topic-report generator, not a fact-checker. A single search and one scraped page is thin evidence, so do not rely on it for breaking news or questions where the answer changed recently. Always check the sources it lists.
- The reader scrapes one page and keeps the first 3000 characters, so reports reflect a limited slice of what the search found.
- Some websites block scraping or require JavaScript, in which case the reader returns little or nothing.
- The critic's score is shown but not used to revise the report.
- Free-tier limits and model names can change without notice.

## Roadmap

- Search several sources in parallel and cross-check them before writing
- Pass sources as structured data so citations come from real results
- Let the critic's feedback trigger an automatic rewrite
- Add a fallback model (for example Groq) for when the Gemini free tier is exhausted
- Add automatic retries for rate limits
- Add tests and a cleaned, pinned `requirements.txt`

## Built with

- [LangChain](https://www.langchain.com) for the agents and prompt chains
- [Google Gemini](https://ai.google.dev) for the language model
- [Tavily](https://tavily.com) for web search
- [Streamlit](https://streamlit.io) for the user interface
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) and Requests for scraping