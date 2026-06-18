import os
# Establish writable configurations for serverless hosting environments before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp"

import re
import io
import json
import uuid
import zipfile
import base64
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai
import pandas as pd
import fitz  # PyMuPDF
import docx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "sciwrite_secret_session_key_2026")

MODEL_NAME = "gemini-2.5-flash-lite"

SECTION_LIST = [
    "Abstract",
    "Introduction",
    "Literature Review",
    "Methodology",
    "Results",
    "Discussion",
    "Conclusion",
]

DEFAULT_PROJECT = {
    "title": "",
    "authors": [],
    "affiliations": [],
    "journal": "",
    "journal_template": "Custom",
    "paper_type": "Literature Review",
    "citation_style": "APA",
    "research_area": "",
    "research_interest": "",
    "refined_topic": None,
    "research_questions": [],
    "objectives": [],
    "references": [],
    "outline": {},
    "sections": {},
    "figures": [],
    "tables": [],
}

maxDuration = 60

# --- Helper State Management ---
def get_project_state():
    if "project" not in session:
        session["project"] = json.loads(json.dumps(DEFAULT_PROJECT))
    return session["project"]

def save_project_state(state):
    session["project"] = state
    session.modified = True

# --- Architecture Utility Parsers ---
def extract_json(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        return json.loads(candidate)
    except Exception:
        return None

def ask_gemini(system_prompt, user_prompt, temperature=0.2, json_mode=False):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        config_args = {"temperature": temperature}
        if json_mode:
            config_args["response_mime_type"] = "application/json"
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                **config_args
            )
        )
        if not response or not getattr(response, "text", None):
            return None
        return response.text
    except Exception:
        return None

def extract_pdf_text(file_bytes, max_pages=2):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = min(max_pages, doc.page_count)
        text = ""
        for i in range(pages):
            text += doc[i].get_text()
        doc.close()
        return text.strip()
    except Exception:
        return ""

def extract_docx_text(file_bytes):
    try:
        f = io.BytesIO(file_bytes)
        d = docx.Document(f)
        return "\n".join(p.text for p in d.paragraphs).strip()
    except Exception:
        return ""

def parse_reference_with_gemini(text_snippet):
    system_prompt = (
        "You extract bibliographic metadata from academic text. "
        "Return valid JSON only, no markdown, no commentary. "
        "Use only the supplied text. Never invent metadata. "
        'If a field cannot be determined, use the string "Unknown".'
    )
    user_prompt = (
        "Extract the following fields as JSON with keys "
        '"title","authors","year","journal","abstract","keywords":\n\n'
        f"{text_snippet[:6000]}"
    )
    raw = ask_gemini(system_prompt, user_prompt, temperature=0.0, json_mode=True)
    parsed = extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return {
            "title": "Unknown",
            "authors": "Unknown",
            "year": "Unknown",
            "journal": "Unknown",
            "abstract": "Unknown",
            "keywords": "Unknown",
        }
    for key in ["title", "authors", "year", "journal", "abstract", "keywords"]:
        if key not in parsed or not parsed[key]:
            parsed[key] = "Unknown"
        if isinstance(parsed[key], list):
            parsed[key] = ", ".join(str(x) for x in parsed[key])
    return parsed

def reference_completeness(ref):
    fields = ["title", "authors", "year", "journal"]
    known = sum(1 for f in fields if ref.get(f) and ref.get(f) != "Unknown")
    return known / len(fields)

def project_completeness(project):
    checks = [
        bool(project.get("title")),
        bool(project.get("authors")),
        bool(project.get("paper_type")),
        bool(project.get("journal")),
        bool(project.get("research_questions")),
        bool(project.get("objectives")),
        bool(project.get("outline")),
        bool(project.get("references")),
    ]
    return sum(checks) / len(checks)

def format_authors_apa(authors_str):
    parts = [a.strip() for a in re.split(r",| and |&", authors_str) if a.strip()]
    if not parts or parts == ["Unknown"]:
        return "Unknown"
    formatted = []
    for p in parts:
        bits = p.split()
        if len(bits) >= 2:
            last = bits[-1]
            initials = " ".join(f"{b[0]}." for b in bits[:-1])
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(p)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"

def format_citation(ref, style):
    title = ref.get("title", "Unknown")
    authors = ref.get("authors", "Unknown")
    year = ref.get("year", "Unknown")
    journal = ref.get("journal", "Unknown")
    first_author_last = "Unknown"
    if authors and authors != "Unknown":
        first_bits = re.split(r",| and |&", authors)[0].strip().split()
        if first_bits:
            first_author_last = first_bits[-1]
    if style == "APA":
        in_text = f"({first_author_last}, {year})"
        ref_list = f"{format_authors_apa(authors)} ({year}). {title}. {journal}."
    elif style == "MLA":
        in_text = f"({first_author_last} {year})"
        ref_list = f"{authors}. \"{title}.\" {journal}, {year}."
    elif style == "IEEE":
        in_text = "[1]"
        ref_list = f"{authors}, \"{title},\" {journal}, {year}."
    else:
        in_text = f"({first_author_last} {year})"
        ref_list = f"{authors}. {year}. {title}. {journal}."
    return in_text, ref_list

def bib_key(ref, idx):
    base = "Unknown"
    if ref.get("authors") and ref.get("authors") != "Unknown":
        base = re.split(r",| and |&", ref["authors"])[0].strip().split()[-1]
    year = ref.get("year", "Unknown")
    return f"{base}{year}_{idx}".replace(" ", "")

def ref_to_bibtex(ref, idx):
    key = bib_key(ref, idx)
    return (
        f"@article{{{key},\n"
        f"  title={{{ref.get('title','Unknown')}}},\n"
        f"  author={{{ref.get('authors','Unknown')}}},\n"
        f"  year={{{ref.get('year','Unknown')}}},\n"
        f"  journal={{{ref.get('journal','Unknown')}}}\n"
        f"}}\n"
    )

def latex_escape(s):
    if not isinstance(s, str):
        s = str(s)
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def build_latex_preamble(template, title, authors):
    auth_str = " \\and ".join(latex_escape(a) for a in authors) if authors else "Author Name"
    if template == "IEEE":
        doc_class = "\\documentclass[conference]{IEEEtran}"
    elif template == "ACM":
        doc_class = "\\documentclass[sigconf]{acmart}"
    elif template == "Springer":
        doc_class = "\\documentclass{llncs}"
    elif template == "Elsevier":
        doc_class = "\\documentclass{elsarticle}"
    else:
        doc_class = "\\documentclass[12pt]{article}"
    return (
        f"{doc_class}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{cite}\n"
        f"\\title{{{latex_escape(title) if title else 'Untitled Research Paper'}}}\n"
        f"\\author{{{auth_str}}}\n"
        "\\date{}\n"
    )

def build_latex_document(project):
    title = project.get("title", "")
    authors = project.get("authors", [])
    template = project.get("journal_template", "Custom")
    sections = project.get("sections", {})
    figures = project.get("figures", [])

    preamble = build_latex_preamble(template, title, authors)
    body = ["\\begin{document}", "\\maketitle"]

    abstract_text = sections.get("Abstract", {}).get("notes", "") if isinstance(sections.get("Abstract"), dict) else ""
    body.append("\\begin{abstract}")
    body.append(latex_escape(abstract_text) if abstract_text else "Abstract content goes here.")
    body.append("\\end{abstract}")

    for sec in SECTION_LIST:
        if sec == "Abstract":
            continue
        body.append(f"\\section{{{latex_escape(sec)}}}")
        sec_data = sections.get(sec)
        notes = ""
        if isinstance(sec_data, dict):
            notes = sec_data.get("notes", "")
        body.append(latex_escape(notes) if notes else "Content to be drafted by the author.")

    if figures:
        for i, fig in enumerate(figures, start=1):
            body.append("\\begin{figure}[h]")
            body.append("\\centering")
            body.append(f"\\includegraphics[width=0.8\\textwidth]{{figures/{fig.get('filename','')}}}")
            body.append(f"\\caption{{{latex_escape(fig.get('caption',''))}}}")
            body.append(f"\\label{{fig:{fig.get('label', f'fig{i}')}}}")
            body.append("\\end{figure}")

    body.append("\\bibliographystyle{plain}")
    body.append("\\bibliography{references}")
    body.append("\\end{document}")

    return preamble + "\n".join(body)

def build_bib_file(references):
    return "\n".join(ref_to_bibtex(r, i) for i, r in enumerate(references))

def build_reference_context(references, limit=8):
    if not references:
        return "No references have been uploaded yet."
    lines = []
    for r in references[:limit]:
        lines.append(
            f"- {r.get('title','Unknown')} ({r.get('authors','Unknown')}, {r.get('year','Unknown')}), "
            f"{r.get('journal','Unknown')}"
        )
    return "\n".join(lines)

# --- Controller Routes ---

@app.route("/")
def index():
    get_project_state()
    return render_template("index.html")

@app.route("/api/state", methods=["GET"])
def get_state():
    proj = get_project_state()
    clean_proj = json.loads(json.dumps(proj))
    clean_proj["completeness"] = int(project_completeness(proj) * 100)
    return jsonify(clean_proj)

@app.route("/api/update_setup", methods=["POST"])
def update_setup():
    proj = get_project_state()
    data = request.json
    proj["title"] = data.get("title", "")
    proj["authors"] = [a.strip() for a in data.get("authors", "").split("\n") if a.strip()]
    proj["affiliations"] = [a.strip() for a in data.get("affiliations", "").split("\n") if a.strip()]
    proj["journal"] = data.get("journal", "")
    proj["paper_type"] = data.get("paper_type", "Literature Review")
    proj["citation_style"] = data.get("citation_style", "APA")
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/upload_references", methods=["POST"])
def upload_references():
    proj = get_project_state()
    files = request.files.getlist("files")
    file_type = request.form.get("type")

    for f in files:
        if not f.filename:
            continue
        file_bytes = f.read()
        text = ""
        if file_type == "pdf" and f.filename.lower().endswith(".pdf"):
            text = extract_pdf_text(file_bytes, max_pages=2)
        elif file_type == "docx" and f.filename.lower().endswith(".docx"):
            text = extract_docx_text(file_bytes)
        
        if not text:
            continue
        
        meta = parse_reference_with_gemini(text)
        meta["id"] = str(uuid.uuid4())
        meta["source_file"] = f.filename
        proj["references"].append(meta)

    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/add_manual_reference", methods=["POST"])
def add_manual_reference():
    proj = get_project_state()
    data = request.json
    new_ref = {
        "id": str(uuid.uuid4()),
        "title": data.get("title") or "Unknown",
        "authors": data.get("authors") or "Unknown",
        "year": data.get("year") or "Unknown",
        "journal": data.get("journal") or "Unknown",
        "abstract": "Unknown",
        "keywords": "Unknown",
        "doi": data.get("doi") or "Unknown",
        "url": data.get("url") or "Unknown",
        "source_file": "Manual entry",
    }
    proj["references"].append(new_ref)
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/edit_references", methods=["POST"])
def edit_references():
    proj = get_project_state()
    data = request.json
    proj["references"] = data.get("references", [])
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/remove_reference", methods=["POST"])
def remove_reference():
    proj = get_project_state()
    idx = int(request.json.get("index", 0))
    if 0 <= idx < len(proj["references"]):
        proj["references"].pop(idx)
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/refine_topic", methods=["POST"])
def refine_topic():
    proj = get_project_state()
    data = request.json
    proj["research_area"] = data.get("research_area", "")
    proj["research_interest"] = data.get("research_interest", "")
    
    system_prompt = (
        "You are a research mentor helping a first-year undergraduate student "
        "choose a manageable research topic. Be encouraging and clear. "
        "Return valid JSON only with keys: refined_topic, scope, difficulty, why_manageable."
    )
    user_prompt = (
        f"Research area: {proj['research_area']}\n"
        f"Student interest: {proj['research_interest']}\n"
        f"Paper type: {proj.get('paper_type', 'Not specified')}"
    )
    raw = ask_gemini(system_prompt, user_prompt, temperature=0.3, json_mode=True)
    parsed = extract_json(raw)
    if parsed:
        proj["refined_topic"] = parsed
        save_project_state(proj)
        return jsonify({"success": True, "refined": parsed})
    return jsonify({"success": False, "error": "Could not parse AI response."}), 500

@app.route("/api/apply_title", methods=["POST"])
def apply_title():
    proj = get_project_state()
    if proj.get("refined_topic") and proj["refined_topic"].get("refined_topic"):
        proj["title"] = proj["refined_topic"]["refined_topic"]
        save_project_state(proj)
        return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route("/api/generate_questions", methods=["POST"])
def generate_questions():
    proj = get_project_state()
    topic_for_rq = proj.get("refined_topic", {}).get("refined_topic") if proj.get("refined_topic") else proj.get("title")
    if not topic_for_rq:
        return jsonify({"success": False, "error": "Set a research title or refine a topic first."}), 400
    
    system_prompt = (
        "You are a research mentor for a first-year undergraduate. "
        "Generate clear, specific, answerable research questions and matching objectives. "
        "Return valid JSON only with keys: research_questions (array of strings), "
        "objectives (array of strings)."
    )
    user_prompt = (
        f"Topic: {topic_for_rq}\n"
        f"Paper type: {proj.get('paper_type', 'Not specified')}"
    )
    raw = ask_gemini(system_prompt, user_prompt, temperature=0.3, json_mode=True)
    parsed = extract_json(raw)
    if parsed:
        proj["research_questions"] = parsed.get("research_questions", [])
        proj["objectives"] = parsed.get("objectives", [])
        save_project_state(proj)
        return jsonify({"success": True, "data": parsed})
    return jsonify({"success": False, "error": "Could not parse AI response."}), 500

@app.route("/api/save_questions", methods=["POST"])
def save_questions():
    proj = get_project_state()
    data = request.json
    proj["research_questions"] = [q.strip() for q in data.get("research_questions", "").split("\n") if q.strip()]
    proj["objectives"] = [o.strip() for o in data.get("objectives", "").split("\n") if o.strip()]
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/generate_outline", methods=["POST"])
def generate_outline():
    proj = get_project_state()
    if not proj.get("title") or not proj.get("objectives"):
        return jsonify({"success": False, "error": "Please set a title and define objectives first."}), 400
    
    system_prompt = (
        "You are a research mentor creating an academic paper outline for a first-year "
        "undergraduate. Return valid JSON only: an object whose keys are section names "
        "and whose values are arrays of short subsection or bullet-point strings."
    )
    user_prompt = (
        f"Title: {proj['title']}\n"
        f"Objectives: {'; '.join(proj['objectives'])}\n"
        f"Paper type: {proj.get('paper_type', 'Not specified')}"
    )
    raw = ask_gemini(system_prompt, user_prompt, temperature=0.3, json_mode=True)
    parsed = extract_json(raw)
    if parsed and isinstance(parsed, dict):
        proj["outline"] = parsed
        save_project_state(proj)
        return jsonify({"success": True, "outline": parsed})
    return jsonify({"success": False, "error": "Could not parse AI response."}), 500

@app.route("/api/save_outline", methods=["POST"])
def save_outline():
    proj = get_project_state()
    data = request.json
    proj["outline"] = data.get("outline", {})
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/writing_guidance", methods=["POST"])
def writing_guidance():
    proj = get_project_state()
    section = request.json.get("section")
    if section not in SECTION_LIST:
        return jsonify({"success": False, "error": "Invalid Section Reference."}), 400
        
    system_prompt = (
        "You are a writing coach for a first-year undergraduate student. "
        "You NEVER write the student's paper text for them. "
        "Use only the supplied references and project context. "
        "If information is unavailable, respond with 'INSUFFICIENT EVIDENCE' for that field. "
        "Do not invent studies, citations, authors, journals, years, or findings. "
        "Return valid JSON only with keys: purpose, what_to_include, suggested_structure, "
        "example_framework, writing_prompts (array of strings)."
    )
    user_prompt = (
        f"Section: {section}\n"
        f"Paper title: {proj.get('title', 'Unknown')}\n"
        f"Objectives: {'; '.join(proj.get('objectives', [])) or 'Unknown'}\n"
        f"Outline for this section: {proj.get('outline', {}).get(section, 'Unknown')}\n"
        f"Available references:\n{build_reference_context(proj.get('references', []))}"
    )
    raw = ask_gemini(system_prompt, user_prompt, temperature=0.2, json_mode=True)
    parsed = extract_json(raw)
    if parsed:
        if section not in proj["sections"] or not isinstance(proj["sections"][section], dict):
            proj["sections"][section] = {"guidance": {}, "notes": ""}
        proj["sections"][section]["guidance"] = parsed
        save_project_state(proj)
        return jsonify({"success": True, "guidance": parsed})
    return jsonify({"success": False, "error": "Could not parse AI guidance response."}), 500

@app.route("/api/save_notes", methods=["POST"])
def save_notes():
    proj = get_project_state()
    data = request.json
    section = data.get("section")
    notes = data.get("notes", "")
    if section in SECTION_LIST:
        if section not in proj["sections"] or not isinstance(proj["sections"][section], dict):
            proj["sections"][section] = {"guidance": {}, "notes": ""}
        proj["sections"][section]["notes"] = notes
        save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/add_figure", methods=["POST"])
def add_figure():
    proj = get_project_state()
    img_file = request.files.get("image")
    caption = request.form.get("caption", "Untitled figure")
    source = request.form.get("source", "Unknown")
    
    if img_file and img_file.filename:
        filename = secure_filename(img_file.filename)
        file_bytes = img_file.read()
        fig_num = len(proj["figures"]) + 1
        
        b64_string = base64.b64encode(file_bytes).decode("utf-8")
        
        proj["figures"].append({
            "id": str(uuid.uuid4()),
            "filename": filename,
            "b64_bytes": b64_string,
            "caption": caption,
            "source": source,
            "number": fig_num,
            "label": f"fig{fig_num}"
        })
        save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/remove_figure", methods=["POST"])
def remove_figure():
    proj = get_project_state()
    fig_id = request.json.get("id")
    proj["figures"] = [f for f in proj["figures"] if f.get("id") != fig_id]
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/update_figure_meta", methods=["POST"])
def update_figure_meta():
    proj = get_project_state()
    data = request.json
    fig_id = data.get("id")
    for f in proj["figures"]:
        if f.get("id") == fig_id:
            f["caption"] = data.get("caption", f["caption"])
            f["source"] = data.get("source", f["source"])
            break
    save_project_state(proj)
    return jsonify({"success": True})

@app.route("/api/citations_view", methods=["GET"])
def citations_view():
    proj = get_project_state()
    style = proj.get("citation_style", "APA")
    output = []
    for ref in proj.get("references", []):
        in_text, ref_list = format_citation(ref, style)
        output.append({
            "title": ref.get("title", "Unknown"),
            "completeness": reference_completeness(ref),
            "in_text": in_text,
            "ref_list": ref_list
        })
    return jsonify({"style": style, "citations": output})

@app.route("/api/export_preview", methods=["POST"])
def export_preview():
    proj = get_project_state()
    proj["journal_template"] = request.json.get("journal_template", "Custom")
    save_project_state(proj)
    
    tex_content = build_latex_document(proj)
    bib_content = build_bib_file(proj.get("references", []))
    return jsonify({"tex": tex_content, "bib": bib_content})

@app.route("/api/download_zip", methods=["GET"])
def download_zip():
    proj = get_project_state()
    tex_content = build_latex_document(proj)
    bib_content = build_bib_file(proj.get("references", []))
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("paper.tex", tex_content)
        zf.writestr("references.bib", bib_content)
        for fig in proj.get("figures", []):
            if "b64_bytes" in fig:
                raw_bytes = base64.b64decode(fig["b64_bytes"].encode("utf-8"))
                zf.writestr(f"figures/{fig['filename']}", raw_bytes)
                
    buffer.seek(0)
    filename = f"{(proj.get('title') or 'research_project').strip().replace(' ', '_')[:40]}.zip"
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )

if __name__ == "__main__":
    app.run(debug=True)