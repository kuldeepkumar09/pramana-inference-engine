"""Generate Pramana Engine Judge Presentation PDF."""
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT = "Pramana_Engine_Presentation.pdf"
W = 190  # usable page width

class PDF(FPDF):
    def header(self):
        self.set_fill_color(25, 47, 90)
        self.rect(0, 0, 210, 14, 'F')
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 3)
        self.cell(190, 8, "Pramana Engine  -  Judge Presentation Guide", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def section_title(self, text):
        self.ln(4)
        self.set_fill_color(25, 47, 90)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.set_x(10)
        self.cell(W, 8, f"  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def sub_title(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(25, 47, 90)
        self.set_x(10)
        self.cell(W, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_x(10)
        self.multi_cell(W, 5, text)
        self.ln(1)

    def bullet(self, label, text):
        self.set_font("Helvetica", "B", 9)
        self.set_x(12)
        self.cell(4, 5, "-")
        self.set_font("Helvetica", "B", 9)
        self.cell(42, 5, label + ":")
        self.set_font("Helvetica", "", 9)
        x = self.get_x()
        y = self.get_y()
        remaining = W - (x - 10)
        self.multi_cell(remaining, 5, text)
        self.ln(1)

    def table_header(self, col1w, col1, col2):
        self.set_fill_color(25, 47, 90)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        self.set_x(10)
        self.cell(col1w, 6, col1, border=1, fill=True)
        self.cell(W - col1w, 6, col2, border=1, fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def table_row(self, col1w, col1, col2, fill=False):
        self.set_font("Helvetica", "", 9)
        if fill:
            self.set_fill_color(235, 240, 250)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_x(10)
        self.cell(col1w, 6, col1, border=1, fill=True)
        self.cell(W - col1w, 6, col2, border=1, fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code_block(self, text):
        self.set_fill_color(245, 245, 245)
        self.set_font("Courier", "", 8)
        self.set_draw_color(180, 180, 180)
        self.set_x(10)
        self.multi_cell(W, 4.5, text, border=1, fill=True)
        self.set_draw_color(0, 0, 0)
        self.ln(2)

    def info_box(self, text):
        self.set_fill_color(230, 245, 255)
        self.set_font("Helvetica", "I", 9)
        self.set_draw_color(100, 160, 220)
        self.set_x(10)
        self.multi_cell(W, 5, text, border=1, fill=True)
        self.set_draw_color(0, 0, 0)
        self.ln(2)


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(10, 18, 10)
pdf.add_page()

# ===== COVER =====
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(25, 47, 90)
pdf.ln(6)
pdf.cell(W, 12, "Pramana Engine", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "B", 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(W, 8, "Neuro-Symbolic AI for Indian Philosophical Reasoning",
         align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(120, 120, 120)
pdf.cell(W, 6, "Judge Presentation Guide  |  BCA Project Defense",
         align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

pdf.set_fill_color(240, 244, 255)
pdf.set_draw_color(25, 47, 90)
pdf.set_font("Helvetica", "B", 10)
pdf.set_x(10)
pdf.multi_cell(W, 6,
    "One-Line Summary:\n"
    "An AI reasoning system that combines Nyaya philosophy with modern machine learning.\n"
    "It gives answers AND explains WHY, using 6 classical Pramanas as justification.",
    border=1, fill=True)
pdf.set_draw_color(0, 0, 0)
pdf.ln(4)

# ===== PART 1 =====
pdf.section_title("PART 1 - Opening Statement (What to say first)")
pdf.info_box(
    '"I have built an AI-powered reasoning system called the Pramana Engine. It is inspired by '
    'ancient Indian Nyaya philosophy and combines modern AI techniques - specifically Large '
    'Language Models and semantic search - with classical logical inference rules. The system '
    'can answer philosophy questions, validate logical arguments, and explain WHY it accepted '
    'or rejected a conclusion."'
)
pdf.body("Tell judges this in the first 30 seconds. It sets the context clearly.")

# ===== PART 2 =====
pdf.section_title("PART 2 - The Problem You Solved")
pdf.body(
    "Modern AI like ChatGPT gives answers - but cannot explain its reasoning in a verifiable way.\n"
    "In Nyaya Shastra, every claim must be justified using a specific Pramana (source of knowledge).\n"
    "This system forces the AI to justify every answer - or refuse to give one."
)
pdf.sub_title("The 6 Pramanas (sources of valid knowledge):")
pdf.ln(1)
pdf.table_header(65, "Pramana", "Simple English Meaning")
pramanas = [
    ("Pratyaksha", "Direct perception - what you see/hear/touch yourself"),
    ("Anumana", "Inference - 'there is smoke, so there must be fire'"),
    ("Shabda", "Testimony - trusting a reliable teacher, text, or authority"),
    ("Upamana", "Comparison - 'a gavaya animal is like a cow'"),
    ("Arthapatti", "Postulation - 'he never eats at home, so he eats outside'"),
    ("Anupalabdhi", "Non-perception - 'there is no pot on the table'"),
]
for i, (p, d) in enumerate(pramanas):
    pdf.table_row(65, p, d, fill=(i % 2 == 0))
pdf.ln(2)

# ===== PART 3 =====
pdf.section_title("PART 3 - How the System Works (Architecture)")
pdf.sub_title("System Flow:")
pdf.code_block(
    "User Question\n"
    "      |\n"
    "      v\n"
    "Flask Web API  (Input Validation + Rate Limiting)\n"
    "      |\n"
    "  ----+----------------------------+-------------------\n"
    "  |                              |                   |\n"
    "  v                              v                   v\n"
    "Symbolic Engine          RAG Pipeline          QA Solver\n"
    "(Pramana logic,          (Hybrid Search        (151-rule\n"
    " Hetvabhasa,              BM25 + FAISS)         knowledge\n"
    " belief revision)         + Ollama LLM)         base)\n"
    "  |                              |\n"
    "  +------------------+-----------+\n"
    "                     |\n"
    "                     v\n"
    "           Pramana Validator\n"
    "      (Is this answer justified?)\n"
    "                     |\n"
    "                     v\n"
    "   Final Answer: VALID / SUSPENDED / INVALID\n"
    "   + confidence score + citations + reasoning trace"
)

pdf.sub_title("Explain each block to judges:")
blocks = [
    ("Flask Web API",
     "The front door. Validates input, limits requests per minute (rate limiting), "
     "routes to the correct component. Built with Python Flask framework."),
    ("Symbolic Engine",
     "The logical brain. Checks if an inference follows Nyaya rules. "
     "Detects 5 types of logical fallacies (Hetvabhasa). "
     "Returns VALID, SUSPENDED, UNJUSTIFIED, or INVALID."),
    ("RAG Pipeline",
     "Retrieval-Augmented Generation - searches the knowledge base FIRST, "
     "then generates an answer based on evidence. Prevents AI hallucination. "
     "Uses two search methods combined (BM25 + FAISS)."),
    ("BM25 Search",
     "Keyword matching algorithm (like Google keyword search). "
     "Finds documents containing the exact words in the question."),
    ("FAISS Semantic Search",
     "Vector similarity search by Meta/Facebook. Converts text to "
     "mathematical vectors representing meaning. 'agni' and 'fire' treated as similar."),
    ("Ollama + Mistral 7B",
     "A 7-billion parameter AI language model running 100% OFFLINE. "
     "No internet needed, no API cost. Handles complex natural language questions."),
    ("QA Solver (Rule Bank)",
     "151 hand-crafted rules covering all Nyaya concepts. Fast - no LLM needed. "
     "Answers deterministically when a question matches a known rule."),
    ("Pramana Validator",
     "Final check: which Pramana justifies this answer? Is confidence high enough? "
     "If evidence is too weak, returns SUSPENDED instead of VALID."),
]
for label, desc in blocks:
    pdf.bullet(label, desc)

# ===== PART 4 =====
pdf.add_page()
pdf.section_title("PART 4 - Technologies Used")
pdf.table_header(60, "Technology", "Purpose in the Project")
tech = [
    ("Python 3.11", "Core programming language"),
    ("Flask", "REST API web framework - exposes engine as HTTP endpoints"),
    ("FAISS (Meta/Facebook)", "Semantic vector search - finds meaning, not just keywords"),
    ("Sentence Transformers", "Converts sentences to 384-dim vectors for semantic search"),
    ("BM25", "Classical keyword ranking algorithm (used by search engines)"),
    ("Ollama + Mistral 7B", "Local offline AI model - no internet required"),
    ("Docker + Compose", "Containerized deployment - runs with one command"),
    ("pytest (67 tests)", "Automated test suite - verifies every component works"),
    ("JSON Rule Bank (151)", "Symbolic Nyaya knowledge base"),
    ("python-dotenv", "Environment variable management (.env config file)"),
    ("flask-limiter", "Rate limiting - prevents API abuse"),
    ("RRF Algorithm", "Reciprocal Rank Fusion - combines BM25 + FAISS scores"),
]
for i, (t, p) in enumerate(tech):
    pdf.table_row(60, t, p, fill=(i % 2 == 0))
pdf.ln(3)

# ===== PART 5 =====
pdf.section_title("PART 5 - Key Technical Concepts to Explain")

pdf.sub_title("RAG - Retrieval Augmented Generation")
pdf.info_box(
    '"RAG solves the hallucination problem in AI. Instead of making the model memorize everything '
    '(which causes wrong confident answers), RAG makes the model look up information first, then '
    'answer based on evidence. Think of it as open-book exam vs closed-book exam."'
)

pdf.sub_title("Hetvabhasa - 5 Logical Fallacies the Engine Detects:")
pdf.table_header(55, "Fallacy", "Meaning")
fallacies = [
    ("Savyabhicara", "Too broad - reason applies where conclusion does not hold"),
    ("Viruddha", "Contradictory - reason actually proves the OPPOSITE"),
    ("Satpratipaksha", "Countered - an equal opposing reason exists"),
    ("Asiddha", "Unproven - the premise itself is doubtful"),
    ("Badhita", "Contradicted - stronger evidence disproves it"),
]
for i, (f, m) in enumerate(fallacies):
    pdf.table_row(55, f, m, fill=(i % 2 == 0))
pdf.ln(2)

pdf.sub_title("Epistemic Status - 4 Possible Outcomes:")
pdf.table_header(55, "Status", "Meaning")
statuses = [
    ("VALID", "Inference is well-grounded and accepted"),
    ("SUSPENDED", "Insufficient evidence - system refuses to commit (honest!)"),
    ("UNJUSTIFIED", "No valid pramana supports this claim"),
    ("INVALID", "Logical fallacy detected - argument rejected"),
]
for i, (s, m) in enumerate(statuses):
    pdf.table_row(55, s, m, fill=(i % 2 == 0))
pdf.ln(2)

pdf.sub_title("Hybrid Search - BM25 + FAISS combined with RRF:")
pdf.body(
    "BM25 Score + FAISS Score merged using Reciprocal Rank Fusion (RRF):\n"
    "   final_score = (0.6 x semantic_score) + (0.4 x keyword_score)\n"
    "Semantic meaning weighted 60%, exact keywords 40%.\n"
    "Result: better retrieval than either method alone."
)

# ===== PART 6 =====
pdf.add_page()
pdf.section_title("PART 6 - Live Demo Script (5 Minutes)")

pdf.sub_title("Start the server (no Ollama needed for demo):")
pdf.code_block("set PRAMANA_DEMO_MODE=1 && python -m pramana_engine.web")
pdf.body("Server starts at http://localhost:5000")

pdf.sub_title("Demo 1 - Classic Nyaya Inference (Smoke -> Fire):")
pdf.code_block(
    "curl -X POST http://localhost:5000/api/infer \\\n"
    "  -H \"Content-Type: application/json\" \\\n"
    "  -d '{\n"
    "    \"paksha\": \"hill\",\n"
    "    \"sadhya\": \"fire\",\n"
    "    \"hetu\": \"smoke\",\n"
    "    \"hetuConf\": 0.9,\n"
    "    \"vyaptiStr\": 0.95,\n"
    "    \"pramanaType\": \"Anumana\"\n"
    "  }'"
)
pdf.info_box(
    'Say: "Paksha = subject (hill). Sadhya = what we want to prove (fire). '
    'Hetu = reason (smoke). Vyapti = universal rule at 95% confidence. '
    'Result: VALID - this is a correct Anumana inference."'
)

pdf.sub_title("Demo 2 - Fallacy Detection:")
pdf.code_block(
    "curl -X POST http://localhost:5000/api/infer \\\n"
    "  -H \"Content-Type: application/json\" \\\n"
    "  -d '{\n"
    "    \"paksha\": \"hill\",\n"
    "    \"sadhya\": \"fire\",\n"
    "    \"hetu\": \"wetness\",\n"
    "    \"hetuConf\": 0.9,\n"
    "    \"vyaptiStr\": 0.95,\n"
    "    \"pramanaType\": \"Anumana\"\n"
    "  }'"
)
pdf.info_box(
    'Say: "Wetness is a sign of WATER, not fire - this is Viruddha Hetvabhasa '
    '(contradictory fallacy). The system detects and rejects it. '
    'This is the key feature: it does NOT blindly accept weak arguments."'
)

pdf.sub_title("Demo 3 - Natural Language Philosophy Question:")
pdf.code_block(
    "curl -X POST http://localhost:5000/api/rag/answer \\\n"
    "  -H \"Content-Type: application/json\" \\\n"
    "  -d '{\"question\": \"What is the difference between pratyaksha and anumana?\"}'"
)
pdf.info_box(
    'Say: "This goes through the full RAG pipeline - searches the knowledge base, '
    'retrieves relevant chunks, generates an answer. Notice: confidence score, '
    'citations of source text, epistemic status, and pramana used are all returned."'
)

pdf.sub_title("Demo 4 - System Health Check:")
pdf.code_block("curl http://localhost:5000/api/health")
pdf.info_box(
    'Say: "This diagnostic endpoint checks all components - FAISS vector store, LLM, '
    'rule bank, required packages. Returns 200 if healthy, 207 if degraded."'
)

# ===== PART 7 =====
pdf.section_title("PART 7 - What Makes This Project Unique")
unique = [
    ("Neuro-Symbolic AI",
     "Combines neural AI (LLM, embeddings) with symbolic logic (rules, Nyaya constraints). "
     "Most AI is one or the other - this project uses both together."),
    ("Epistemic Honesty",
     "Returns SUSPENDED when evidence is insufficient instead of always giving an answer. "
     "This is philosophically correct - modern AI almost never does this."),
    ("151-Rule Knowledge Base",
     "Hand-crafted rules covering all major Nyaya concepts: Pancavayava syllogism, "
     "four types of Abhava, Pararthanumana vs Svarthanumana, school comparisons."),
    ("Runs 100% Offline",
     "No cloud dependency, no API cost. Mistral 7B runs locally via Ollama. "
     "Works anywhere, even without internet."),
    ("Demo Mode",
     "If Ollama is unavailable, falls back to pure symbolic reasoning - still gives "
     "structured answers. No single point of failure."),
    ("67 Automated Tests",
     "Full test suite covering unit, integration, and end-to-end scenarios. "
     "Proves the system works reliably, not just in demos."),
]
for label, desc in unique:
    pdf.bullet(label, desc)

# ===== PART 8 =====
pdf.add_page()
pdf.section_title("PART 8 - Answering Tough Judge Questions")

qa_pairs = [
    ("Why Nyaya philosophy specifically?",
     "Nyaya is one of the oldest formal logic systems - older than Aristotle. "
     "Every claim needs a justified knowledge source (Pramana). "
     "This makes it ideal for building AI that reasons transparently."),
    ("What is RAG and why did you use it?",
     "RAG = Retrieval Augmented Generation. Solves AI hallucination: "
     "the model looks up information first, then answers based on evidence. "
     "Like open-book vs closed-book exam."),
    ("How accurate is the system?",
     "Rule-based questions (151 rules): 100% accurate, deterministic. "
     "RAG questions: epistemic status ensures VALID is only returned when evidence "
     "supports it. Confidence score shows exact certainty."),
    ("Difference between /api/infer and /api/rag/answer?",
     "/api/infer = structured inference, you provide paksha/sadhya/hetu explicitly. "
     "/api/rag/answer = natural language, system searches, generates, then validates."),
    ("Can you improve this further?",
     "Yes - extend to all 6 Indian philosophical schools "
     "(Nyaya, Vaisheshika, Samkhya, Yoga, Mimamsa, Vedanta). "
     "Add graph database for storing inference chains over time."),
    ("What is FAISS?",
     "Facebook AI Similarity Search. Converts text to 384-dimensional vectors "
     "representing meaning, finds similar vectors very fast across millions of documents."),
    ("What is Docker used for?",
     "Docker packages the entire application into containers. "
     "Anyone can run the project with: docker-compose up. "
     "No installation problems, works on any machine."),
    ("Why did you use Mistral 7B?",
     "Mistral 7B is a state-of-the-art open-source model that runs locally "
     "via Ollama. 7 billion parameters - strong reasoning, no cost, fully private."),
]
for q, a in qa_pairs:
    pdf.set_fill_color(230, 245, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(10)
    pdf.multi_cell(W, 5, f"Q: {q}", fill=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(255, 255, 255)
    pdf.set_x(10)
    pdf.multi_cell(W, 5, f"A: {a}")
    pdf.ln(2)

# ===== PART 9 =====
pdf.section_title("PART 9 - One-Slide Project Summary")
pdf.set_fill_color(25, 47, 90)
pdf.set_text_color(255, 255, 255)
pdf.set_font("Courier", "B", 9)
summary = (
    "PROJECT:  Pramana Engine\n"
    "=============================================================\n\n"
    "PROBLEM:  AI gives answers without justification.\n"
    "          Nyaya philosophy demands justified knowledge.\n\n"
    "SOLUTION: Neuro-Symbolic AI enforcing Pramana validation\n"
    "          on every inference - accepts, suspends, or rejects.\n\n"
    "HOW:      Flask API -> Hybrid Search (BM25 + FAISS)\n"
    "                    -> Local LLM (Mistral 7B via Ollama)\n"
    "                    -> Symbolic Validator (151 rules)\n"
    "                    -> Epistemic Status (VALID/SUSPENDED/INVALID)\n\n"
    "UNIQUE:   6 Pramanas | 5 Hetvabhasa | 151 Rules | Offline LLM\n"
    "          67 automated tests | Demo mode | Docker deployment\n\n"
    "RESULT:   Philosophically honest AI - says 'I don't know'\n"
    "          when evidence is insufficient."
)
pdf.set_x(10)
pdf.multi_cell(W, 5, summary, border=0, fill=True)
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

# ===== PART 10 =====
pdf.section_title("PART 10 - Commands to Start the Project")
pdf.sub_title("Option 1 - Demo Mode (no Ollama, instant start on Windows):")
pdf.code_block("set PRAMANA_DEMO_MODE=1 && python -m pramana_engine.web")

pdf.sub_title("Option 2 - Full mode (Ollama must be running):")
pdf.code_block("python -m pramana_engine.web")

pdf.sub_title("Option 3 - Docker full stack (recommended for judges):")
pdf.code_block("docker-compose up")

pdf.sub_title("Key API Endpoints:")
pdf.table_header(60, "Endpoint", "Purpose")
endpoints = [
    ("POST /api/infer", "Structured Nyaya inference with pramana validation"),
    ("POST /api/rag/answer", "Natural language question answering (RAG + LLM)"),
    ("POST /api/compare", "Pramana-constrained vs baseline comparison"),
    ("GET /api/health", "Full system health check - all components"),
    ("GET /api/version", "API version information"),
    ("GET /api/rag/status", "RAG pipeline status + demo mode flag"),
    ("GET /judge", "Judge Dashboard - upload dataset for batch evaluation"),
    ("GET /", "Main Unified Workspace UI"),
]
for i, (e, p) in enumerate(endpoints):
    pdf.table_row(60, e, p, fill=(i % 2 == 0))

pdf.ln(6)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(100, 100, 100)
pdf.set_x(10)
pdf.cell(W, 6, "Generated for BCA Project Defense  |  Pramana Engine  |  IIT Delhi",
         align="C")

# Save
pdf.output(OUTPUT)
print(f"PDF saved: {os.path.abspath(OUTPUT)}")
