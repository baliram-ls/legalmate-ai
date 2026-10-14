import os
import io
import json
import logging
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, session, render_template_string
from jinja2 import Environment, DictLoader
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

# ── Optional document parsers ──────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# ── Small built-in legal knowledge base (India-focused) ───────────────────────
LEGAL_KB = [
    {
        "topic": "Right to Information (RTI) Act, 2005",
        "summary": "Every citizen of India can request information from any public authority within 30 days. The Central Information Commission oversees compliance.",
        "source": "Right to Information Act, 2005"
    },
    {
        "topic": "Consumer Protection Act, 2019",
        "summary": "Consumers can file complaints against defective goods or deficient services. District Consumer Disputes Redressal Commissions hear cases up to ₹1 crore.",
        "source": "Consumer Protection Act, 2019"
    },
    {
        "topic": "Indian Contract Act, 1872",
        "summary": "A valid contract requires offer, acceptance, lawful consideration, capacity of parties, free consent, and lawful object. Agreements against public policy are void.",
        "source": "Indian Contract Act, 1872"
    },
    {
        "topic": "Fundamental Rights (Articles 12–35, Constitution of India)",
        "summary": "Citizens enjoy rights including equality (Article 14), freedom of speech (Article 19), right against exploitation (Article 23–24), and right to constitutional remedies (Article 32).",
        "source": "Constitution of India, Part III"
    },
    {
        "topic": "Tenant Rights – Rent Control Laws",
        "summary": "Most Indian states have Rent Control Acts that protect tenants from arbitrary eviction and regulate rent increases. Landlords must issue proper eviction notices.",
        "source": "State-specific Rent Control Acts"
    },
    {
        "topic": "Labour Law – Minimum Wages Act, 1948",
        "summary": "Employers must pay minimum wages as notified by the government for scheduled employments. Violation is a criminal offence.",
        "source": "Minimum Wages Act, 1948"
    },
    {
        "topic": "FIR Filing Process",
        "summary": "Any person may file a First Information Report (FIR) at a police station for cognizable offences. Under Section 154 CrPC the police are duty-bound to register it.",
        "source": "Code of Criminal Procedure, 1973, Section 154"
    },
    {
        "topic": "Right to Bail",
        "summary": "Bail is a legal right for bailable offences. For non-bailable offences, bail is at the court's discretion considering factors such as flight risk and severity of crime.",
        "source": "Code of Criminal Procedure, 1973, Sections 436–439"
    },
]

# ── Utility: check allowed file extension ─────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ── Utility: extract text from uploaded file bytes ────────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "txt":
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace")

    if ext == "pdf":
        if not HAS_PYMUPDF:
            return "[PDF parsing unavailable – please install PyMuPDF]"
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(pages)
        except Exception as exc:
            log.error("PDF extraction error: %s", type(exc).__name__)
            return "[Could not extract PDF text]"

    if ext == "docx":
        if not HAS_DOCX:
            return "[DOCX parsing unavailable – please install python-docx]"
        try:
            buf = io.BytesIO(file_bytes)
            doc = DocxDocument(buf)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:
            log.error("DOCX extraction error: %s", type(exc).__name__)
            return "[Could not extract DOCX text]"

    return "[Unsupported file type]"

# ── Core LLM helper ────────────────────────────────────────────────────────────
def generate_response(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    if not GROQ_API_KEY:
        return (
            "⚠️ GROQ_API_KEY is not configured. "
            "Please create a .env file with your Groq API key and restart the server."
        )
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        err = str(exc)
        log.error("Groq API error: %s", err[:200])
        if "authentication" in err.lower() or "api_key" in err.lower():
            return "⚠️ Invalid Groq API key. Please check your .env file."
        if "rate_limit" in err.lower():
            return "⚠️ Groq API rate limit reached. Please wait a moment and try again."
        if "model" in err.lower():
            return f"⚠️ Model '{GROQ_MODEL}' not found. Check GROQ_MODEL in your .env file."
        return f"⚠️ AI service error: {err[:300]}"

# ── Agent 1 – Legal Knowledge Agent ───────────────────────────────────────────
def legal_knowledge_agent(question: str) -> dict:
    kb_hits = [
        entry for entry in LEGAL_KB
        if any(word.lower() in entry["topic"].lower() or word.lower() in entry["summary"].lower()
               for word in question.split() if len(word) > 3)
    ]
    kb_context = ""
    if kb_hits:
        kb_context = "\n\nRelevant knowledge base entries:\n" + "\n".join(
            f"- [{e['source']}] {e['summary']}" for e in kb_hits[:3]
        )

    system_prompt = (
        "You are LegalMate AI's Legal Knowledge Agent — a helpful, accurate, and honest legal information assistant "
        "focused on Indian law. Your role is to provide general legal information in simple, beginner-friendly English.\n\n"
        "STRICT RULES:\n"
        "1. Never invent law section numbers, court judgments, or legal citations you are not certain about.\n"
        "2. If you are uncertain, clearly say 'This requires verification by a qualified lawyer.'\n"
        "3. Clearly distinguish general legal information from verified legal provisions.\n"
        "4. Use practical Indian examples to explain concepts.\n"
        "5. End every response with a short disclaimer reminding the user to consult a lawyer for their specific situation.\n"
        "6. Format responses with clear headings and bullet points where helpful.\n"
        "7. Do not claim to give legal advice — only general legal information.\n"
        + kb_context
    )
    answer = generate_response(system_prompt, question)
    return {
        "answer": answer,
        "kb_sources": [{"topic": e["topic"], "source": e["source"]} for e in kb_hits[:3]],
    }

# ── Agent 2 – Legal Document Analysis Agent ───────────────────────────────────
def legal_document_analysis_agent(doc_text: str, question: str = "", mode: str = "analyze") -> dict:
    truncated = doc_text[:6000]
    if len(doc_text) > 6000:
        truncated += "\n\n[... document truncated for analysis ...]"

    if mode == "question" and question:
        system_prompt = (
            "You are LegalMate AI's Document Analysis Agent. Answer the user's question using ONLY the text of the "
            "legal document provided. Do not invent facts. If the document does not contain sufficient information "
            "to answer the question, say so clearly. Format answers clearly and in plain English."
        )
        user_prompt = f"Document text:\n\"\"\"\n{truncated}\n\"\"\"\n\nUser question: {question}"
        answer = generate_response(system_prompt, user_prompt)
        return {"mode": "question", "answer": answer}

    # Default: full document analysis
    system_prompt = (
        "You are LegalMate AI's Document Analysis Agent. Analyze the provided legal document and return a JSON response "
        "with the following structure exactly (valid JSON only, no markdown fences):\n\n"
        "{\n"
        '  "summary": "Plain-language summary of the document (3-5 sentences)",\n'
        '  "document_type": "Type of document (e.g., Employment Agreement, Rental Agreement)",\n'
        '  "key_parties": ["Party 1", "Party 2"],\n'
        '  "clauses": [\n'
        "    {\n"
        '      "title": "Short clause title",\n'
        '      "excerpt": "Exact quote from the document (max 100 words)",\n'
        '      "concern": "Why this clause deserves attention",\n'
        '      "priority": "Low|Medium|High"\n'
        "    }\n"
        "  ],\n"
        '  "lawyer_questions": ["Question 1", "Question 2", "Question 3"],\n'
        '  "overall_risk": "Low|Medium|High",\n'
        '  "disclaimer": "This analysis is for informational purposes only and is not legal advice."\n'
        "}\n\n"
        "RULES:\n"
        "- Only use text present in the document. Do not invent content.\n"
        "- Identify clauses that are vague, one-sided, potentially unfair, or unusually broad.\n"
        "- Do not claim a contract is illegal or enforceable without clear evidence.\n"
        "- Set priority High only for clearly problematic clauses.\n"
        "- Return ONLY valid JSON."
    )
    raw = generate_response(system_prompt, f"Document:\n\"\"\"\n{truncated}\n\"\"\"", temperature=0.2)

    # Parse JSON response
    try:
        # Strip potential markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        result = json.loads(clean.strip())
    except (json.JSONDecodeError, ValueError):
        # Fallback: return raw text as summary
        result = {
            "summary": raw[:800] if not raw.startswith("⚠️") else "Analysis unavailable.",
            "document_type": "Unknown",
            "key_parties": [],
            "clauses": [],
            "lawyer_questions": [
                "What are my obligations under this contract?",
                "Are there any clauses I should negotiate before signing?",
                "What happens if either party breaches this agreement?",
            ],
            "overall_risk": "Unknown",
            "disclaimer": "This analysis is for informational purposes only and is not legal advice.",
            "_raw_error": raw if raw.startswith("⚠️") else "",
        }
    result["mode"] = "analyze"
    return result

# ── Orchestrator ───────────────────────────────────────────────────────────────
def orchestrator(action: str, **kwargs) -> dict:
    if action == "legal_question":
        question = kwargs.get("question", "").strip()
        if not question:
            return {"error": "Please enter a legal question."}
        return legal_knowledge_agent(question)

    if action == "analyze_document":
        doc_text = kwargs.get("doc_text", "").strip()
        if not doc_text:
            return {"error": "No document text provided."}
        if len(doc_text) < 50:
            return {"error": "The document appears too short to analyze."}
        return legal_document_analysis_agent(doc_text, mode="analyze")

    if action == "document_question":
        doc_text = kwargs.get("doc_text", "").strip()
        question = kwargs.get("question", "").strip()
        if not doc_text:
            return {"error": "No document text available. Please upload a document first."}
        if not question:
            return {"error": "Please enter a question about the document."}
        return legal_document_analysis_agent(doc_text, question=question, mode="question")

    return {"error": f"Unknown action: {action}"}

# ══════════════════════════════════════════════════════════════════════════════
#  HTML TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

BASE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{% block title %}LegalMate AI{% endblock %}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"/>
<style>
:root{
  --navy:#0d2b4e;--navy-light:#1a3f70;--blue:#1565c0;--blue-light:#e8f0fe;
  --accent:#0288d1;--success-soft:#e8f5e9;--warn-soft:#fff8e1;--danger-soft:#fce4ec;
  --text:#1a1a2e;--muted:#5f6368;--border:#dee2e6;--radius:12px;
}
body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;background:#f4f6fb;color:var(--text);}
/* Sidebar */
#sidebar{width:260px;min-height:100vh;background:var(--navy);position:fixed;top:0;left:0;z-index:1000;transition:.3s;}
#sidebar .brand{padding:1.4rem 1.2rem;border-bottom:1px solid rgba(255,255,255,.12);}
#sidebar .brand h5{color:#fff;font-weight:700;font-size:1.05rem;margin:0;}
#sidebar .brand small{color:rgba(255,255,255,.55);font-size:.75rem;}
#sidebar .nav-link{color:rgba(255,255,255,.75);padding:.7rem 1.2rem;border-radius:8px;margin:.1rem .6rem;font-size:.9rem;display:flex;align-items:center;gap:.55rem;transition:.2s;}
#sidebar .nav-link:hover,#sidebar .nav-link.active{background:rgba(255,255,255,.12);color:#fff;}
#sidebar .nav-link i{font-size:1rem;width:1.2rem;text-align:center;}
#sidebar .disclaimer-sidebar{padding:.8rem 1rem;margin:.8rem;background:rgba(255,255,255,.07);border-radius:8px;font-size:.7rem;color:rgba(255,255,255,.45);line-height:1.4;}
/* Main */
#main{margin-left:260px;min-height:100vh;transition:.3s;}
.topbar{background:#fff;border-bottom:1px solid var(--border);padding:.75rem 1.5rem;display:flex;align-items:center;justify-content:space-between;}
.topbar .toggle-btn{display:none;}
.page-content{padding:1.8rem 1.5rem;}
/* Cards */
.card{border:none;border-radius:var(--radius);box-shadow:0 1px 6px rgba(0,0,0,.07);}
.card-feature{border-left:4px solid var(--blue);transition:.2s;}
.card-feature:hover{box-shadow:0 4px 16px rgba(0,0,0,.1);transform:translateY(-2px);}
/* Chat */
.chat-box{height:420px;overflow-y:auto;background:#f8fafd;border:1px solid var(--border);border-radius:var(--radius);padding:1rem;}
.msg-user{background:var(--blue);color:#fff;border-radius:18px 18px 4px 18px;padding:.65rem 1rem;max-width:80%;margin-left:auto;margin-bottom:.8rem;font-size:.9rem;}
.msg-ai{background:#fff;border:1px solid var(--border);border-radius:18px 18px 18px 4px;padding:.65rem 1rem;max-width:85%;margin-bottom:.8rem;font-size:.9rem;line-height:1.6;}
.msg-ai pre{white-space:pre-wrap;font-family:inherit;margin:0;}
/* Priority badges */
.badge-high{background:#fce4ec;color:#b71c1c;border:1px solid #ef9a9a;}
.badge-medium{background:#fff3e0;color:#e65100;border:1px solid #ffcc80;}
.badge-low{background:#e8f5e9;color:#1b5e20;border:1px solid #a5d6a7;}
/* Clause card */
.clause-card{border-left:4px solid #ccc;margin-bottom:1rem;}
.clause-card.high{border-color:#e53935;}
.clause-card.medium{border-color:#fb8c00;}
.clause-card.low{border-color:#43a047;}
.excerpt-box{background:#f8f9fa;border-left:3px solid var(--accent);padding:.5rem .75rem;font-size:.85rem;color:var(--muted);border-radius:0 6px 6px 0;}
/* Upload zone */
.upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:2.5rem;text-align:center;cursor:pointer;transition:.2s;background:#fff;}
.upload-zone:hover{border-color:var(--blue);background:var(--blue-light);}
/* Spinner */
.spinner-overlay{display:none;position:fixed;inset:0;background:rgba(255,255,255,.7);z-index:9999;align-items:center;justify-content:center;}
.spinner-overlay.show{display:flex;}
/* Responsive */
@media(max-width:768px){
  #sidebar{left:-260px;}
  #sidebar.open{left:0;}
  #main{margin-left:0;}
  .topbar .toggle-btn{display:block;}
}
/* KB source pills */
.kb-pill{background:var(--blue-light);color:var(--blue);border-radius:20px;padding:.2rem .7rem;font-size:.75rem;margin:.1rem;}
/* Risk indicator */
.risk-low{color:#2e7d32;font-weight:600;}
.risk-medium{color:#e65100;font-weight:600;}
.risk-high{color:#b71c1c;font-weight:600;}
.risk-unknown{color:var(--muted);font-weight:600;}
</style>
</head>
<body>

<!-- Spinner -->
<div class="spinner-overlay" id="spinnerOverlay">
  <div class="text-center">
    <div class="spinner-border text-primary" style="width:3rem;height:3rem;" role="status"></div>
    <div class="mt-2 fw-semibold text-primary">Processing…</div>
  </div>
</div>

<!-- Sidebar -->
<div id="sidebar">
  <div class="brand">
    <h5><i class="bi bi-balance-scale-left me-2"></i>LegalMate AI</h5>
    <small>Smart Legal Aid Assistant</small>
  </div>
  <nav class="mt-3">
    <a href="/" class="nav-link {% if page=='home' %}active{% endif %}">
      <i class="bi bi-house-door"></i> Dashboard
    </a>
    <a href="/chat" class="nav-link {% if page=='chat' %}active{% endif %}">
      <i class="bi bi-chat-dots"></i> Legal Knowledge
    </a>
    <a href="/analyze" class="nav-link {% if page=='analyze' %}active{% endif %}">
      <i class="bi bi-file-earmark-text"></i> Document Analyzer
    </a>
    <a href="/about" class="nav-link {% if page=='about' %}active{% endif %}">
      <i class="bi bi-info-circle"></i> About
    </a>
  </nav>
  <div class="disclaimer-sidebar mt-auto">
    <i class="bi bi-exclamation-triangle me-1"></i>
    General information only. Not legal advice. Consult a qualified lawyer for your specific situation.
  </div>
</div>

<!-- Main -->
<div id="main">
  <div class="topbar">
    <button class="btn btn-sm btn-outline-secondary toggle-btn" onclick="toggleSidebar()">
      <i class="bi bi-list"></i>
    </button>
    <span class="fw-semibold text-secondary">{% block topbar_title %}LegalMate AI{% endblock %}</span>
    <span class="badge bg-primary-subtle text-primary border border-primary-subtle" style="font-size:.75rem;">
      <i class="bi bi-cpu me-1"></i>{{ model }}
    </span>
  </div>

  <div class="page-content">
    <!-- Disclaimer banner -->
    <div class="alert alert-warning d-flex align-items-start gap-2 py-2 mb-3" style="font-size:.82rem;border-radius:10px;" role="alert">
      <i class="bi bi-exclamation-triangle-fill flex-shrink-0 mt-1"></i>
      <div><strong>Disclaimer:</strong> LegalMate AI provides general legal information and document-review assistance.
      It is <strong>not</strong> a substitute for advice from a qualified lawyer. AI-generated answers may be incomplete
      or incorrect. Verify important legal information using reliable sources before making legal decisions.</div>
    </div>

    {% block content %}{% endblock %}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
function toggleSidebar(){document.getElementById('sidebar').classList.toggle('open');}
function showSpinner(){document.getElementById('spinnerOverlay').classList.add('show');}
function hideSpinner(){document.getElementById('spinnerOverlay').classList.remove('show');}
function showAlert(msg,type='danger',container='alertContainer'){
  const c=document.getElementById(container);
  if(!c)return;
  c.innerHTML=`<div class="alert alert-${type} alert-dismissible fade show" role="alert">
    ${msg}<button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>`;
}
function mdToHtml(t){
  if(!t)return '';
  return t
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/^### (.+)$/gm,'<h6 class="fw-bold mt-2 mb-1">$1</h6>')
    .replace(/^## (.+)$/gm,'<h5 class="fw-bold mt-2 mb-1">$1</h5>')
    .replace(/^# (.+)$/gm,'<h4 class="fw-bold mt-2 mb-1">$1</h4>')
    .replace(/^[-•] (.+)$/gm,'<li>$1</li>')
    .replace(/(<li>[\s\S]+?<\/li>)/g,'<ul class="mb-1 ps-3">$1</ul>')
    .replace(/\n{2,}/g,'</p><p>').replace(/\n/g,'<br/>')
    .replace(/^(.)/,'<p>$1').replace(/(.)$/,'$1</p>');
}
</script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

# ── Home / Dashboard ───────────────────────────────────────────────────────────
HOME_TEMPLATE = r"""
{% extends 'base' %}
{% block title %}Dashboard – LegalMate AI{% endblock %}
{% block topbar_title %}Dashboard{% endblock %}
{% block content %}
<div class="row mb-4">
  <div class="col-12">
    <div class="card p-4" style="background:linear-gradient(135deg,#0d2b4e 0%,#1565c0 100%);color:#fff;border-radius:var(--radius);">
      <div class="row align-items-center">
        <div class="col-md-8">
          <h2 class="fw-bold mb-1"><i class="bi bi-balance-scale-left me-2"></i>Welcome to LegalMate AI</h2>
          <p class="mb-3" style="opacity:.85;">Your AI-powered legal information assistant. Ask legal questions in plain English or upload a document for AI-assisted review.</p>
          <div class="d-flex gap-2 flex-wrap">
            <a href="/chat" class="btn btn-light btn-sm fw-semibold"><i class="bi bi-chat-dots me-1"></i>Ask a Legal Question</a>
            <a href="/analyze" class="btn btn-outline-light btn-sm fw-semibold"><i class="bi bi-file-earmark-text me-1"></i>Analyze a Document</a>
          </div>
        </div>
        <div class="col-md-4 text-end d-none d-md-block">
          <i class="bi bi-shield-check" style="font-size:5rem;opacity:.25;"></i>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="row g-3 mb-4">
  <div class="col-md-6 col-lg-4">
    <div class="card card-feature h-100 p-3">
      <div class="d-flex align-items-center gap-3 mb-2">
        <div class="rounded-circle d-flex align-items-center justify-content-center" style="width:44px;height:44px;background:var(--blue-light);">
          <i class="bi bi-chat-text" style="color:var(--blue);font-size:1.25rem;"></i>
        </div>
        <div>
          <h6 class="fw-bold mb-0">Legal Knowledge Agent</h6>
          <small class="text-muted">Ask any legal question</small>
        </div>
      </div>
      <p class="text-muted small mb-3">Get plain-English explanations of Indian laws, rights, contracts, and legal procedures with practical examples.</p>
      <a href="/chat" class="btn btn-primary btn-sm mt-auto">Start Chat <i class="bi bi-arrow-right ms-1"></i></a>
    </div>
  </div>
  <div class="col-md-6 col-lg-4">
    <div class="card card-feature h-100 p-3">
      <div class="d-flex align-items-center gap-3 mb-2">
        <div class="rounded-circle d-flex align-items-center justify-content-center" style="width:44px;height:44px;background:#f3e5f5;">
          <i class="bi bi-file-earmark-search" style="color:#7b1fa2;font-size:1.25rem;"></i>
        </div>
        <div>
          <h6 class="fw-bold mb-0">Document Analysis Agent</h6>
          <small class="text-muted">Upload PDF, DOCX, TXT</small>
        </div>
      </div>
      <p class="text-muted small mb-3">Upload a legal document and receive a plain-language summary, risk flags, and suggested questions for your lawyer.</p>
      <a href="/analyze" class="btn btn-primary btn-sm mt-auto">Upload Document <i class="bi bi-arrow-right ms-1"></i></a>
    </div>
  </div>
  <div class="col-md-6 col-lg-4">
    <div class="card card-feature h-100 p-3">
      <div class="d-flex align-items-center gap-3 mb-2">
        <div class="rounded-circle d-flex align-items-center justify-content-center" style="width:44px;height:44px;background:var(--success-soft);">
          <i class="bi bi-book" style="color:#2e7d32;font-size:1.25rem;"></i>
        </div>
        <div>
          <h6 class="fw-bold mb-0">Built-in Knowledge Base</h6>
          <small class="text-muted">India-focused legal info</small>
        </div>
      </div>
      <p class="text-muted small mb-3">Covers RTI Act, Consumer Protection, Contract Law, Fundamental Rights, Labour Law, tenant rights and more.</p>
      <a href="/about" class="btn btn-outline-secondary btn-sm mt-auto">Learn More <i class="bi bi-arrow-right ms-1"></i></a>
    </div>
  </div>
</div>

<div class="row g-3">
  <div class="col-md-6">
    <div class="card p-3">
      <h6 class="fw-bold mb-3"><i class="bi bi-lightning-charge text-warning me-2"></i>Quick Questions to Try</h6>
      <div class="d-flex flex-column gap-2">
        {% for q in sample_questions %}
        <a href="/chat?q={{ q|urlencode }}" class="btn btn-outline-secondary btn-sm text-start">
          <i class="bi bi-arrow-right-circle text-primary me-2"></i>{{ q }}
        </a>
        {% endfor %}
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card p-3">
      <h6 class="fw-bold mb-3"><i class="bi bi-info-circle text-primary me-2"></i>How It Works</h6>
      <ol class="mb-0 ps-3" style="font-size:.9rem;line-height:2;">
        <li>Ask a legal question or upload a document.</li>
        <li>The AI orchestrator routes your request to the right agent.</li>
        <li>The agent analyzes using Groq LLM + a local knowledge base.</li>
        <li>You receive a clear, plain-English response.</li>
        <li>Always verify important information with a qualified lawyer.</li>
      </ol>
    </div>
  </div>
</div>
{% endblock %}
"""

# ── Legal Knowledge Chat ───────────────────────────────────────────────────────
CHAT_TEMPLATE = r"""
{% extends 'base' %}
{% block title %}Legal Knowledge Chat – LegalMate AI{% endblock %}
{% block topbar_title %}Legal Knowledge Chat{% endblock %}
{% block content %}
<div class="row">
  <div class="col-lg-8">
    <div class="card p-3 mb-3">
      <h5 class="fw-bold mb-1"><i class="bi bi-chat-dots text-primary me-2"></i>Legal Knowledge Agent</h5>
      <p class="text-muted small mb-0">Ask any general legal question. Responses are based on Indian law. For document-specific questions, use the Document Analyzer.</p>
    </div>

    <!-- Chat window -->
    <div class="card p-0 mb-3">
      <div class="chat-box" id="chatBox">
        <div class="msg-ai">
          <strong>LegalMate AI</strong><br/>
          👋 Hello! I'm your Legal Knowledge Agent. Ask me anything about Indian law — rights, contracts, procedures, or legal terms — and I'll explain it in simple language.<br/><br/>
          <em style="font-size:.8rem;color:var(--muted);">I provide general information only, not legal advice.</em>
        </div>
      </div>
      <div class="p-3 border-top d-flex gap-2">
        <input type="text" id="questionInput" class="form-control" placeholder="Type your legal question here…" autocomplete="off"/>
        <button class="btn btn-primary px-3" id="sendBtn" onclick="sendQuestion()">
          <i class="bi bi-send"></i>
        </button>
      </div>
    </div>
    <div id="alertContainer"></div>
  </div>

  <div class="col-lg-4">
    <div class="card p-3 mb-3">
      <h6 class="fw-bold mb-3"><i class="bi bi-lightbulb text-warning me-1"></i> Example Questions</h6>
      <div class="d-flex flex-column gap-2">
        {% for q in sample_questions %}
        <button class="btn btn-outline-secondary btn-sm text-start" onclick="fillQuestion(this.textContent.trim())">{{ q }}</button>
        {% endfor %}
      </div>
    </div>
    <div class="card p-3">
      <h6 class="fw-bold mb-2"><i class="bi bi-database text-info me-1"></i> Knowledge Base Topics</h6>
      <div class="d-flex flex-wrap gap-1">
        {% for t in kb_topics %}
        <span class="kb-pill">{{ t }}</span>
        {% endfor %}
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
const prefillQ = {{ prefill_q | tojson }};
window.addEventListener('DOMContentLoaded', () => {
  if(prefillQ){ document.getElementById('questionInput').value = prefillQ; sendQuestion(); }
});

function fillQuestion(q){
  document.getElementById('questionInput').value = q;
  document.getElementById('questionInput').focus();
}

function appendMsg(role, html){
  const box = document.getElementById('chatBox');
  const div = document.createElement('div');
  div.className = role === 'user' ? 'msg-user' : 'msg-ai';
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendQuestion(){
  const inp = document.getElementById('questionInput');
  const q = inp.value.trim();
  if(!q) return;
  inp.value = '';
  document.getElementById('sendBtn').disabled = true;
  appendMsg('user', escHtml(q));
  appendMsg('ai', '<span class="text-muted"><span class="spinner-border spinner-border-sm me-1"></span>Thinking…</span>');
  const chatBox = document.getElementById('chatBox');
  const loadingMsg = chatBox.lastElementChild;
  try {
    const res = await fetch('/api/legal_question', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await res.json();
    let html = '';
    if(data.error){
      html = `<span class="text-danger"><i class="bi bi-exclamation-circle me-1"></i>${escHtml(data.error)}</span>`;
    } else {
      html = `<strong>LegalMate AI</strong><br/>${mdToHtml(data.answer)}`;
      if(data.kb_sources && data.kb_sources.length){
        html += '<div class="mt-2 pt-2 border-top"><small class="text-muted fw-semibold">Sources from knowledge base:</small><div class="mt-1">';
        data.kb_sources.forEach(s => {
          html += `<span class="kb-pill"><i class="bi bi-book-half me-1"></i>${escHtml(s.source)}</span>`;
        });
        html += '</div></div>';
      }
    }
    loadingMsg.innerHTML = html;
    chatBox.scrollTop = chatBox.scrollHeight;
  } catch(e) {
    loadingMsg.innerHTML = '<span class="text-danger">Network error. Please try again.</span>';
  }
  document.getElementById('sendBtn').disabled = false;
}

function escHtml(t){
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

document.getElementById('questionInput').addEventListener('keydown', e => {
  if(e.key === 'Enter') sendQuestion();
});
</script>
{% endblock %}
"""

# ── Document Analyzer ──────────────────────────────────────────────────────────
ANALYZE_TEMPLATE = r"""
{% extends 'base' %}
{% block title %}Document Analyzer – LegalMate AI{% endblock %}
{% block topbar_title %}Document Analyzer{% endblock %}
{% block content %}
<div class="row">
  <div class="col-lg-7">
    <!-- Upload -->
    <div class="card p-3 mb-3" id="uploadSection">
      <h5 class="fw-bold mb-1"><i class="bi bi-file-earmark-arrow-up text-primary me-2"></i>Upload Legal Document</h5>
      <p class="text-muted small mb-3">Supported formats: PDF, DOCX, TXT — Max 10 MB</p>
      <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
        <i class="bi bi-cloud-upload" style="font-size:2.5rem;color:var(--blue);"></i>
        <p class="mt-2 mb-0 fw-semibold">Click to upload or drag & drop</p>
        <small class="text-muted">PDF, DOCX, or TXT</small>
      </div>
      <input type="file" id="fileInput" accept=".pdf,.docx,.txt" style="display:none;" onchange="handleFile(event)"/>
      <div id="fileInfo" class="mt-2" style="display:none;">
        <div class="d-flex align-items-center gap-2 p-2 bg-light rounded">
          <i class="bi bi-file-earmark-text text-primary fs-5"></i>
          <div class="flex-grow-1">
            <div class="fw-semibold" id="fileName"></div>
            <small class="text-muted" id="fileSize"></small>
          </div>
          <button class="btn btn-sm btn-outline-danger" onclick="clearFile()"><i class="bi bi-x"></i></button>
        </div>
      </div>
    </div>

    <!-- Text preview + Analyze -->
    <div class="card p-3 mb-3" id="previewSection" style="display:none;">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <h6 class="fw-bold mb-0"><i class="bi bi-eye me-1 text-secondary"></i>Document Preview</h6>
        <button class="btn btn-primary btn-sm" id="analyzeBtn" onclick="analyzeDocument()">
          <i class="bi bi-search me-1"></i>Analyze Document
        </button>
      </div>
      <textarea class="form-control" id="docPreview" rows="8" style="font-size:.82rem;font-family:monospace;" readonly></textarea>
      <small class="text-muted mt-1 d-block">First 3000 characters shown. Full document is sent for analysis.</small>
    </div>

    <div id="alertContainer"></div>

    <!-- Results -->
    <div id="resultsSection" style="display:none;">
      <!-- Summary -->
      <div class="card p-3 mb-3">
        <div class="d-flex justify-content-between align-items-start mb-2">
          <h6 class="fw-bold mb-0"><i class="bi bi-card-text text-primary me-1"></i>Document Summary</h6>
          <span id="overallRiskBadge"></span>
        </div>
        <p id="docType" class="text-muted small mb-1"></p>
        <p id="summaryText" class="mb-1" style="font-size:.9rem;"></p>
        <div id="partiesList"></div>
      </div>

      <!-- Clauses -->
      <div class="card p-3 mb-3" id="clausesSection">
        <h6 class="fw-bold mb-3"><i class="bi bi-flag text-warning me-1"></i>Clauses to Review</h6>
        <div id="clausesList"></div>
      </div>

      <!-- Lawyer questions -->
      <div class="card p-3 mb-3">
        <h6 class="fw-bold mb-3"><i class="bi bi-patch-question text-info me-1"></i>Questions to Ask Your Lawyer</h6>
        <ol id="lawyerQuestions" class="ps-3" style="font-size:.9rem;line-height:2;"></ol>
      </div>
    </div>
  </div>

  <div class="col-lg-5">
    <!-- Q&A on document -->
    <div class="card p-3 mb-3" id="docQASection" style="display:none;">
      <h6 class="fw-bold mb-2"><i class="bi bi-chat-left-quote text-primary me-1"></i>Ask About This Document</h6>
      <p class="text-muted small mb-2">Ask specific questions about the uploaded document.</p>
      <div class="chat-box mb-2" id="docChatBox" style="height:300px;">
        <div class="msg-ai">
          Document loaded! Ask me anything about its contents — parties, obligations, terms, or specific clauses.
        </div>
      </div>
      <div class="d-flex gap-2">
        <input type="text" id="docQuestionInput" class="form-control form-control-sm" placeholder="Ask about the document…"/>
        <button class="btn btn-primary btn-sm" id="docSendBtn" onclick="askDocQuestion()"><i class="bi bi-send"></i></button>
      </div>
    </div>

    <!-- How to read results -->
    <div class="card p-3">
      <h6 class="fw-bold mb-2"><i class="bi bi-info-circle text-primary me-1"></i>Understanding Results</h6>
      <div class="d-flex align-items-start gap-2 mb-2">
        <span class="badge badge-high px-2 py-1 flex-shrink-0">HIGH</span>
        <small>Clause needs urgent attention before signing. Consult a lawyer.</small>
      </div>
      <div class="d-flex align-items-start gap-2 mb-2">
        <span class="badge badge-medium px-2 py-1 flex-shrink-0">MEDIUM</span>
        <small>Clause is potentially ambiguous or one-sided. Discuss with a lawyer.</small>
      </div>
      <div class="d-flex align-items-start gap-2 mb-3">
        <span class="badge badge-low px-2 py-1 flex-shrink-0">LOW</span>
        <small>Clause is notable but not immediately alarming. Still worth reviewing.</small>
      </div>
      <div class="alert alert-warning py-2 mb-0" style="font-size:.8rem;">
        <i class="bi bi-exclamation-triangle me-1"></i>
        AI analysis is not a substitute for legal advice. Always consult a qualified lawyer before signing or acting on any legal document.
      </div>
    </div>
  </div>
</div>
{% endblock %}
{% block scripts %}
<script>
let docText = '';

// Drag & drop
const zone = document.getElementById('uploadZone');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor='var(--blue)'; });
zone.addEventListener('dragleave', () => { zone.style.borderColor=''; });
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.style.borderColor='';
  const f = e.dataTransfer.files[0];
  if(f) processFile(f);
});

function handleFile(event){ processFile(event.target.files[0]); }

function processFile(file){
  if(!file) return;
  const ext = file.name.split('.').pop().toLowerCase();
  if(!['pdf','docx','txt'].includes(ext)){
    showAlert('Invalid file type. Please upload a PDF, DOCX, or TXT file.','danger');
    return;
  }
  if(file.size > 10*1024*1024){
    showAlert('File too large. Maximum size is 10 MB.','danger');
    return;
  }
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileSize').textContent = (file.size/1024).toFixed(1)+' KB';
  document.getElementById('fileInfo').style.display = 'block';
  uploadFile(file);
}

async function uploadFile(file){
  showSpinner();
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/upload', { method:'POST', body: fd });
    const data = await res.json();
    hideSpinner();
    if(data.error){ showAlert(data.error, 'danger'); return; }
    docText = data.text;
    document.getElementById('docPreview').value = data.text.substring(0,3000);
    document.getElementById('previewSection').style.display = 'block';
    document.getElementById('docQASection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('alertContainer').innerHTML = '';
  } catch(e) {
    hideSpinner();
    showAlert('Upload failed. Please try again.','danger');
  }
}

function clearFile(){
  docText='';
  document.getElementById('fileInput').value='';
  document.getElementById('fileInfo').style.display='none';
  document.getElementById('previewSection').style.display='none';
  document.getElementById('docQASection').style.display='none';
  document.getElementById('resultsSection').style.display='none';
}

async function analyzeDocument(){
  if(!docText){ showAlert('No document loaded.','danger'); return; }
  showSpinner();
  document.getElementById('analyzeBtn').disabled = true;
  try {
    const res = await fetch('/api/analyze', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({doc_text: docText})
    });
    const data = await res.json();
    hideSpinner();
    document.getElementById('analyzeBtn').disabled = false;
    if(data.error){ showAlert(data.error,'danger'); return; }
    renderResults(data);
  } catch(e){
    hideSpinner();
    document.getElementById('analyzeBtn').disabled = false;
    showAlert('Analysis failed. Please try again.','danger');
  }
}

function renderResults(d){
  // Summary
  document.getElementById('summaryText').textContent = d.summary || 'No summary available.';
  document.getElementById('docType').textContent = d.document_type ? '📄 ' + d.document_type : '';
  const risk = (d.overall_risk||'Unknown').toLowerCase();
  const riskEl = document.getElementById('overallRiskBadge');
  const riskClasses = {high:'risk-high',medium:'risk-medium',low:'risk-low',unknown:'risk-unknown'};
  riskEl.innerHTML = `<span class="${riskClasses[risk]||'risk-unknown'}"><i class="bi bi-shield-exclamation me-1"></i>Overall Risk: ${d.overall_risk||'Unknown'}</span>`;
  const parties = document.getElementById('partiesList');
  parties.innerHTML = '';
  if(d.key_parties && d.key_parties.length){
    parties.innerHTML = '<div class="mt-1"><small class="fw-semibold text-muted">Key Parties: </small>'
      + d.key_parties.map(p=>`<span class="badge bg-secondary-subtle text-secondary border me-1">${escHtml(p)}</span>`).join('')
      + '</div>';
  }
  // Clauses
  const list = document.getElementById('clausesList');
  list.innerHTML = '';
  if(!d.clauses || !d.clauses.length){
    list.innerHTML = '<p class="text-muted small">No specific clauses flagged for review.</p>';
  } else {
    d.clauses.forEach(c => {
      const p = (c.priority||'Low').toLowerCase();
      const badgeHtml = {high:`<span class="badge badge-high">HIGH</span>`,medium:`<span class="badge badge-medium">MEDIUM</span>`,low:`<span class="badge badge-low">LOW</span>`}[p]||`<span class="badge bg-secondary">?</span>`;
      list.innerHTML += `
        <div class="card clause-card ${p} p-3">
          <div class="d-flex justify-content-between align-items-start mb-2">
            <h6 class="fw-bold mb-0">${escHtml(c.title||'Clause')}</h6>
            ${badgeHtml}
          </div>
          ${c.excerpt ? `<div class="excerpt-box mb-2"><i class="bi bi-quote me-1 text-muted"></i>${escHtml(c.excerpt)}</div>` : ''}
          <p class="small mb-0 text-muted">${escHtml(c.concern||'')}</p>
        </div>`;
    });
  }
  // Lawyer questions
  const ql = document.getElementById('lawyerQuestions');
  ql.innerHTML = '';
  (d.lawyer_questions||[]).forEach(q => {
    ql.innerHTML += `<li>${escHtml(q)}</li>`;
  });
  document.getElementById('resultsSection').style.display = 'block';
  document.getElementById('resultsSection').scrollIntoView({behavior:'smooth'});
}

async function askDocQuestion(){
  const q = document.getElementById('docQuestionInput').value.trim();
  if(!q || !docText) return;
  document.getElementById('docQuestionInput').value = '';
  document.getElementById('docSendBtn').disabled = true;
  appendDocMsg('user', escHtml(q));
  appendDocMsg('ai', '<span class="text-muted"><span class="spinner-border spinner-border-sm me-1"></span>Searching document…</span>');
  const box = document.getElementById('docChatBox');
  const loadingMsg = box.lastElementChild;
  try {
    const res = await fetch('/api/doc_question', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({doc_text: docText, question: q})
    });
    const data = await res.json();
    loadingMsg.innerHTML = data.error
      ? `<span class="text-danger">${escHtml(data.error)}</span>`
      : mdToHtml(data.answer||'');
    box.scrollTop = box.scrollHeight;
  } catch(e) {
    loadingMsg.innerHTML = '<span class="text-danger">Network error.</span>';
  }
  document.getElementById('docSendBtn').disabled = false;
}

function appendDocMsg(role, html){
  const box = document.getElementById('docChatBox');
  const div = document.createElement('div');
  div.className = role === 'user' ? 'msg-user' : 'msg-ai';
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function escHtml(t){ return t ? String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : ''; }

document.getElementById('docQuestionInput').addEventListener('keydown', e => {
  if(e.key==='Enter') askDocQuestion();
});
</script>
{% endblock %}
"""

# ── About Page ─────────────────────────────────────────────────────────────────
ABOUT_TEMPLATE = r"""
{% extends 'base' %}
{% block title %}About – LegalMate AI{% endblock %}
{% block topbar_title %}About LegalMate AI{% endblock %}
{% block content %}
<div class="row justify-content-center">
  <div class="col-lg-9">

    <div class="card p-4 mb-4">
      <h4 class="fw-bold mb-3"><i class="bi bi-info-circle text-primary me-2"></i>About LegalMate AI</h4>
      <p>LegalMate AI is an AI-powered legal aid assistant designed to make basic legal information accessible to everyone in India. It helps you understand legal terminology, your rights, contracts, and common legal procedures in plain, simple English — without needing a law degree.</p>
      <p class="mb-0">LegalMate AI is intended as a starting point for legal understanding. It is <strong>not</strong> a lawyer, and its responses should never replace professional legal advice.</p>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6 class="fw-bold mb-3" style="color:var(--blue);"><i class="bi bi-robot me-2"></i>Legal Knowledge Agent</h6>
          <ul class="small mb-0 ps-3" style="line-height:2;">
            <li>Answers general legal-information questions.</li>
            <li>Explains legal terminology and rights.</li>
            <li>Covers Indian laws and procedures.</li>
            <li>Searches a built-in India-focused knowledge base.</li>
            <li>Clearly flags uncertainty.</li>
            <li>Never invents law sections or court judgments.</li>
          </ul>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card p-3 h-100">
          <h6 class="fw-bold mb-3" style="color:#7b1fa2;"><i class="bi bi-file-earmark-search me-2"></i>Document Analysis Agent</h6>
          <ul class="small mb-0 ps-3" style="line-height:2;">
            <li>Extracts and analyzes PDF, DOCX, and TXT files.</li>
            <li>Summarizes documents in plain language.</li>
            <li>Flags risky, vague, or one-sided clauses.</li>
            <li>Assigns Low / Medium / High review priority.</li>
            <li>Suggests questions for your lawyer.</li>
            <li>Answers questions about the uploaded document.</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="card p-4 mb-4">
      <h6 class="fw-bold mb-3"><i class="bi bi-diagram-3 text-success me-2"></i>How It Works</h6>
      <div class="row g-3">
        <div class="col-md-4 text-center">
          <div class="rounded-circle mx-auto d-flex align-items-center justify-content-center mb-2" style="width:52px;height:52px;background:var(--blue-light);">
            <span class="fw-bold" style="color:var(--blue);">1</span>
          </div>
          <strong class="d-block small">You Submit</strong>
          <small class="text-muted">Type a legal question or upload a document (PDF/DOCX/TXT).</small>
        </div>
        <div class="col-md-4 text-center">
          <div class="rounded-circle mx-auto d-flex align-items-center justify-content-center mb-2" style="width:52px;height:52px;background:#f3e5f5;">
            <span class="fw-bold" style="color:#7b1fa2;">2</span>
          </div>
          <strong class="d-block small">Orchestrator Routes</strong>
          <small class="text-muted">A simple Python orchestrator selects the correct AI agent.</small>
        </div>
        <div class="col-md-4 text-center">
          <div class="rounded-circle mx-auto d-flex align-items-center justify-content-center mb-2" style="width:52px;height:52px;background:var(--success-soft);">
            <span class="fw-bold" style="color:#2e7d32;">3</span>
          </div>
          <strong class="d-block small">AI Responds</strong>
          <small class="text-muted">The agent queries Groq LLM and returns a structured, plain-English answer.</small>
        </div>
      </div>
    </div>

    <div class="card p-4 mb-4 border-warning">
      <h6 class="fw-bold mb-3 text-warning"><i class="bi bi-exclamation-triangle me-2"></i>Limitations & Responsible Use</h6>
      <ul class="small mb-0 ps-3" style="line-height:2;">
        <li><strong>Not a lawyer:</strong> LegalMate AI does not provide legal advice and cannot represent you in any legal matter.</li>
        <li><strong>AI may be wrong:</strong> Large language models can make mistakes, hallucinate facts, or miss important nuances.</li>
        <li><strong>India-specific:</strong> The knowledge base focuses on Indian law. Laws vary by state and may change over time.</li>
        <li><strong>Verify everything:</strong> Always verify critical legal information using official government sources or a qualified lawyer.</li>
        <li><strong>No citations invented:</strong> The AI is instructed never to fabricate law sections, court judgments, or citations.</li>
        <li><strong>Document privacy:</strong> Uploaded documents are processed in memory and not stored. However, document text is sent to the Groq API for analysis.</li>
        <li><strong>No identity verification:</strong> LegalMate AI does not verify any person's identity, criminal record, or legal standing.</li>
      </ul>
    </div>

    <div class="card p-4">
      <h6 class="fw-bold mb-3"><i class="bi bi-stack text-info me-2"></i>Technology Stack</h6>
      <div class="row g-2">
        {% for tech in tech_stack %}
        <div class="col-md-4">
          <div class="d-flex align-items-center gap-2 p-2 bg-light rounded">
            <i class="bi bi-check-circle-fill text-success"></i>
            <small>{{ tech }}</small>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>

  </div>
</div>
{% endblock %}
"""

# ── 404 ────────────────────────────────────────────────────────────────────────
ERROR_TEMPLATE = r"""
{% extends 'base' %}
{% block title %}{{ code }} – LegalMate AI{% endblock %}
{% block topbar_title %}Error {{ code }}{% endblock %}
{% block content %}
<div class="text-center py-5">
  <i class="bi bi-exclamation-circle" style="font-size:4rem;color:var(--muted);"></i>
  <h3 class="mt-3 fw-bold">{{ code }} – {{ message }}</h3>
  <p class="text-muted">{{ description }}</p>
  <a href="/" class="btn btn-primary mt-2">Return to Dashboard</a>
</div>
{% endblock %}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_QUESTIONS = [
    "What are my rights if my landlord refuses to return the security deposit?",
    "How do I file an RTI application?",
    "What is the difference between a void and a voidable contract?",
    "Can my employer deduct salary without notice?",
    "What should I do if police refuse to register my FIR?",
    "How do consumer courts work in India?",
]

KB_TOPICS = [e["topic"].split("–")[0].split("(")[0].strip() for e in LEGAL_KB]

TECH_STACK = [
    "Python + Flask (backend)",
    "Groq API – LLaMA 3 LLM",
    "Bootstrap 5 (UI framework)",
    "PyMuPDF (PDF extraction)",
    "python-docx (DOCX extraction)",
    "python-dotenv (config)",
    "Bootstrap Icons",
    "Vanilla JavaScript (frontend)",
    "render_template_string (templates)",
]

def _render(template_str, **kwargs):
    """Render a child template string that extends 'base' via an in-memory DictLoader."""
    env = Environment(loader=DictLoader({
        'base': BASE_TEMPLATE,
        'child': template_str,
    }), autoescape=False)
    # Expose the urlencode filter (used in HOME_TEMPLATE)
    from urllib.parse import quote_plus
    env.filters['urlencode'] = lambda s: quote_plus(s)
    tmpl = env.get_template('child')
    return tmpl.render(model=GROQ_MODEL, **kwargs)


@app.route("/")
def home():
    return _render(HOME_TEMPLATE, page="home", sample_questions=SAMPLE_QUESTIONS[:4])


@app.route("/chat")
def chat():
    prefill = request.args.get("q", "")
    return _render(
        CHAT_TEMPLATE,
        page="chat",
        sample_questions=SAMPLE_QUESTIONS,
        kb_topics=KB_TOPICS,
        prefill_q=prefill,
    )


@app.route("/analyze")
def analyze():
    return _render(ANALYZE_TEMPLATE, page="analyze")


@app.route("/about")
def about():
    return _render(ABOUT_TEMPLATE, page="about", tech_stack=TECH_STACK)


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/legal_question", methods=["POST"])
def api_legal_question():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if len(question) > 2000:
        return jsonify({"error": "Question is too long (max 2000 characters)."}), 400
    result = orchestrator("legal_question", question=question)
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Invalid file type. Please upload PDF, DOCX, or TXT."}), 400
    file_bytes = file.read()
    if len(file_bytes) == 0:
        return jsonify({"error": "The uploaded file is empty."}), 400
    text = extract_text(file_bytes, filename)
    if not text or len(text.strip()) < 20:
        return jsonify({"error": "Could not extract readable text from the document."}), 400
    return jsonify({"text": text, "filename": filename, "chars": len(text)})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    doc_text = data.get("doc_text", "").strip()
    if not doc_text:
        return jsonify({"error": "No document text provided."}), 400
    result = orchestrator("analyze_document", doc_text=doc_text)
    return jsonify(result)


@app.route("/api/doc_question", methods=["POST"])
def api_doc_question():
    data = request.get_json(silent=True) or {}
    doc_text = data.get("doc_text", "").strip()
    question = data.get("question", "").strip()
    if not doc_text:
        return jsonify({"error": "No document text provided."}), 400
    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    result = orchestrator("document_question", doc_text=doc_text, question=question)
    return jsonify(result)


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return _render(
        ERROR_TEMPLATE,
        page="",
        code=404,
        message="Page Not Found",
        description="The page you are looking for does not exist.",
    ), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 10 MB."}), 413


@app.errorhandler(500)
def server_error(e):
    return _render(
        ERROR_TEMPLATE,
        page="",
        code=500,
        message="Server Error",
        description="An unexpected error occurred. Please try again.",
    ), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Ensure UTF-8 output on Windows consoles
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not GROQ_API_KEY:
        print("\n[WARNING] GROQ_API_KEY is not set.")
        print("  Create a .env file with your Groq API key to enable AI features.")
        print("  See .env.example for the required format.\n")
    if not HAS_PYMUPDF:
        print("[INFO] PyMuPDF not installed - PDF support disabled.")
    if not HAS_DOCX:
        print("[INFO] python-docx not installed - DOCX support disabled.")
    print(f"[OK] Starting LegalMate AI with model: {GROQ_MODEL}")
    print("[OK] Open: http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
