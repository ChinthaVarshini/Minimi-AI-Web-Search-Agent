# Minimi AI Web Search Agent

Minimi is a conversational research assistant that answers questions using web search and links its answers to the websites it consulted. It supports follow-up questions, keeps recent conversation context during the current session, displays message timestamps, and streams answers as they are generated.

## Features

- **Web search as the only agent tool:** Tavily search is the only external tool exposed to the assistant.
- **Website citations:** Search results are numbered, cited in answers, and linked in a source panel.
- **Conversation memory:** Recent questions and answers are used to interpret follow-ups.
- **Timestamps:** Each question and answer is timestamped in the browser's local timezone.
- **Fast responses:** Uses a lightweight default chat model, bounded search results, a short search timeout, and streamed output.
- **Prompt-injection awareness:** Search results are treated as untrusted content, not instructions.

## Requirements

- Python 3.10 or later
- An OpenAI API key for chat
- A Tavily API key for web search

## Setup

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/ChinthaVarshini/Minimi-AI-Web-Search-Agent.git
   cd Minimi-AI-Web-Search-Agent
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Configure credentials. Either copy `.env.example` to `.env` and add your keys:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   TAVILY_API_KEY=your-tavily-api-key
   OPENAI_MODEL=gpt-4o-mini
   ```

   Or set `OPENAI_API_KEY` and `TAVILY_API_KEY` in your shell before starting the app. `.env` is ignored by Git; never commit credentials.

5. Start Minimi:

   ```bash
   streamlit run app.py
   ```

   Open the local URL printed by Streamlit. You can also enter credentials and change the model from the app's sidebar.

## Run tests

```bash
python -m unittest discover -s tests
```

## How it works

The chat model receives recent conversation history and has access to one function: `web_search`. When it needs external facts, it formulates a query for Tavily. The application returns a small set of title, URL, and snippet records to the model. The answer is streamed to the interface, and the same numbered results are shown as clickable citations.

Conversation history is kept in Streamlit's session state. It is not stored in a database and may not survive a browser refresh or a new session.

For architecture, request flow, and design trade-offs, see [design.md](design.md).
