# LegalMate AI – Smart Legal Aid Assistant

An AI-powered legal information assistant built with Python, Flask, and the Groq API. It provides general legal information and document analysis for Indian law in plain, beginner-friendly English.

> ⚠️ **Disclaimer:** LegalMate AI provides general legal information only. It is **not** a substitute for advice from a qualified lawyer. Always verify important legal information with a qualified legal professional before making legal decisions.

---

## Features

- **Legal Knowledge Chat** – Ask general legal questions about Indian law (RTI, contracts, tenant rights, consumer protection, labour law, FIR, bail, and more).
- **Document Analyzer** – Upload PDF, DOCX, or TXT legal documents for AI-assisted summary and clause review.
- **Risk Flags** – Clauses are flagged as Low / Medium / High priority for your attention.
- **Document Q&A** – Ask specific questions about an uploaded document.
- **Built-in Knowledge Base** – 8 India-focused legal topics available without needing the API.

---

## Prerequisites

- Python 3.9 or newer
- A free Groq API key → [https://console.groq.com](https://console.groq.com)

---

## Setup & Installation (Windows PowerShell)

### Step 1 – Create a virtual environment

```powershell
python -m venv venv
```

### Step 2 – Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

> If you see an execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 – Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 – Configure your API key

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` in a text editor and add your Groq API key:

```
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=llama3-8b-8192
```

Get your free Groq API key at: [https://console.groq.com](https://console.groq.com)

### Step 5 – Start the application

```powershell
python app.py
```

### Step 6 – Open in your browser

```
http://127.0.0.1:5000
```

---

## Available Groq Models (free)

| Model | Best for |
|-------|----------|
| `llama3-8b-8192` | Fast responses, good for chat (default) |
| `llama3-70b-8192` | Higher quality, slightly slower |
| `mixtral-8x7b-32768` | Long documents (32k context) |
| `gemma2-9b-it` | Instruction-following tasks |

---

## Project Structure

```
legalmate-ai/
├── app.py              ← Complete Flask application (single file)
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
├── .env                ← Your local config (NOT committed to git)
├── .gitignore
└── README.md
```

---

## How It Works

1. You submit a question or upload a legal document.
2. A simple Python **orchestrator** routes your request to the correct agent.
3. The **Legal Knowledge Agent** searches the built-in knowledge base and queries the Groq LLM.
4. The **Document Analysis Agent** extracts text from your file and asks the LLM to analyze it.
5. Responses are returned as structured, plain-English results in the browser.

---

## Supported File Formats

| Format | Library used |
|--------|-------------|
| `.pdf` | PyMuPDF (`fitz`) |
| `.docx` | python-docx |
| `.txt` | Built-in Python |

Maximum upload size: **10 MB**

---

## Running Without an API Key

The application starts and the UI is fully functional without a Groq API key. When you send a question or analyze a document, you will see a clear message asking you to configure `GROQ_API_KEY`. The built-in knowledge base pills and UI are always available.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'groq'` | Run `pip install -r requirements.txt` |
| `⚠️ Invalid Groq API key` | Check your `.env` file has the correct key |
| `PDF parsing unavailable` | Run `pip install PyMuPDF` |
| `DOCX parsing unavailable` | Run `pip install python-docx` |
| PowerShell script execution error | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Port 5000 already in use | Change `port=5000` to another port (e.g. `5001`) in `app.py` |

---

## Legal & Ethical Notes

- LegalMate AI does **not** provide legal advice.
- The AI is instructed never to fabricate law section numbers, court judgments, or citations.
- Uploaded documents are processed in memory and **not stored** on disk.
- Document text is sent to the Groq API for analysis — do not upload highly sensitive personal documents.
- Always consult a qualified lawyer before signing or acting on any legal document.

---

## License

This project is for educational and demonstration purposes.
