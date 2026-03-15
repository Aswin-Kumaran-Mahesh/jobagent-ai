"""
Job Search Agent — Web Interface
Multi-provider LLM support:
  - Anthropic (Claude)  — paid, best reasoning
  - Groq                — FREE tier, very fast (Llama 3 70B)
  - Google Gemini       — FREE tier (gemini-1.5-flash)
"""

import os, json, csv, queue, threading, re, io, base64
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)

JOBS_CSV = os.path.join(os.path.dirname(__file__), "jobs.csv")

PROVIDER_MODELS = {
    "anthropic": "claude-opus-4-5",
    "groq":      "llama3-70b-8192",
    "gemini":    "gemini-2.0-flash",
    "openai":    "gpt-4o-mini",
}

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CANDIDATE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PROFILE = {
    "name": "Aswin Kumaran Mahesh Kannan",
    "degree": "Master of Engineering in Data Science",
    "university": "University of Houston",
    "gpa": 3.85, "graduation": "May 2026",
    "skills": [
        "Python","LangGraph","LangChain","LlamaIndex","RAG","LLMs",
        "prompt engineering","agentic workflows","tool integration",
        "ChromaDB","FAISS","Neo4j","vector search","hybrid retrieval",
        "PyTorch","scikit-learn","deep learning","transformers",
        "MLOps","CI/CD","Docker","REST APIs","Git","AWS","GCP",
        "RLHF","instruction tuning","context engineering","MCP",
        "federated learning","SHAP","feature engineering"
    ],
    "years_experience": 1,
    "location_preference": ["Remote","Houston TX","New York NY","San Francisco CA","Any"],
    "excluded_companies": [],
    "max_experience_required": 3,
}

DEFAULT_RESUME = """ASWIN KUMARAN MAHESH KANNAN
maswinkumaran@gmail.com | linkedin.com/in/aswin-kumaran-mahesh

SUMMARY
MS Data Science candidate (GPA 3.85, UH, May 2026) with hands-on experience building
GenAI applications, agentic workflows, and RAG pipelines. Skilled in LangGraph, LangChain,
ChromaDB, PyTorch, and MLOps. Published federated learning research (Springer, 2026).

EDUCATION
Master of Engineering in Data Science - University of Houston (GPA: 3.85) | 2024-2026
B.Tech in AI and Data Science - M. Kumarasamy College of Engineering | 2020-2024

EXPERIENCE
Data Science Intern - Sterling Software | Feb 2024 - Jun 2024
* Improved predictive model accuracy by 20% using Python and scikit-learn.
* Built and evaluated deep learning pipelines for classification tasks.
* Collaborated with cross-functional teams to integrate ML solutions into production.

Software Engineer Intern - Mistral Solutions | Jul 2022 - Aug 2022
* Developed Python-based AI tools for autonomous vision and data processing.
* Built and consumed REST APIs to integrate AI capabilities into software pipelines.

PROJECTS
RareDx: Agentic GenAI System | LangGraph, MCP, ChromaDB
* Built GenAI copilot using LangGraph with agentic workflows and tool integration.
* Implemented context engineering, memory/state management, hybrid retrieval.

ClimateRAG: RAG Pipeline | LangChain, FAISS, LLMs
* Designed RAG pipeline with hybrid retrieval, vector search, and reranking.

Fed-AI: Federated Learning | Springer Book Chapter, 2026
* Federated ML framework: MSE=0.039, R2=0.89. Integrated SHAP and Grad-CAM."""

# ─────────────────────────────────────────────────────────────────────────────
# TOOL LOGIC (pure Python, provider-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def load_jobs():
    jobs = []
    with open(JOBS_CSV, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row["id"] = i + 1
            row["Years Experience"] = int(row.get("Years Experience", 0))
            jobs.append(dict(row))
    return jobs

def filter_jobs(jobs, location_preference, max_experience_years, excluded_companies):
    passed, rejected = [], []
    loc_lower = [l.lower() for l in location_preference]
    for job in jobs:
        reasons = []
        job_loc = job.get("Location", "").lower()
        loc_ok = (
            "any" in loc_lower or "remote" in job_loc
            or any(p in job_loc for p in loc_lower if p not in ("any","remote"))
        )
        if not loc_ok:
            reasons.append(f"Location '{job['Location']}' not in preferences")
        if job.get("Years Experience", 0) > max_experience_years:
            reasons.append(f"Requires {job['Years Experience']} yrs (max: {max_experience_years})")
        if job.get("Company","") in excluded_companies:
            reasons.append("Company excluded")
        if reasons:
            rejected.append({"job_id": job["id"], "title": job["Job Title"],
                             "company": job["Company"], "reasons": reasons})
        else:
            passed.append(job)
    return {"filtered_jobs": passed, "rejected_count": len(rejected),
            "passed_count": len(passed), "rejection_log": rejected}

def rank_jobs(jobs, candidate_skills, candidate_experience_years, jd_text=""):
    skills_lower = {s.lower() for s in candidate_skills}
    jd_keywords = set()
    if jd_text:
        words = re.findall(r"[a-zA-Z][\w\+\#\.]*", jd_text.lower())
        stopwords = {"the","and","or","for","with","this","that","are","was","you","have",
                     "will","your","our","their","from","into","about","which","must","should",
                     "able","work","team","role","experience","skills","strong","required","preferred"}
        jd_keywords = {w for w in words if len(w) > 3 and w not in stopwords}
    scored = []
    for job in jobs:
        req = [s.strip().lower() for s in job.get("Required Skills","").split(",")]
        matched = [s for s in req if any(cs in s or s in cs for cs in skills_lower)]
        skill_score = round((len(matched)/len(req))*60, 1) if req else 0
        exp_diff = abs(candidate_experience_years - job.get("Years Experience",0))
        exp_score = {0:30,1:24,2:15}.get(exp_diff,5)
        loc = job.get("Location","").lower()
        loc_score = 10 if "remote" in loc else (7 if "hybrid" in loc else 3)
        jd_score = 0
        if jd_keywords:
            job_words = set(re.findall(r"[a-zA-Z][\w\+\#\.]*",
                (job.get("Job Description","")+job.get("Required Skills","")).lower()))
            overlap = jd_keywords & job_words
            jd_score = round(min(len(overlap)/max(len(jd_keywords),1)*15, 15), 1)
        total = round(skill_score + exp_score + loc_score + jd_score, 1)
        scored.append({**job, "score": total, "skill_score": skill_score,
                       "exp_score": exp_score, "loc_score": loc_score,
                       "jd_score": jd_score, "matched_skills": matched,
                       "skill_match_pct": round((len(matched)/len(req))*100 if req else 0, 1),
                       "jd_boost_active": bool(jd_text)})
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    return {"ranked_jobs": ranked, "top_3": ranked[:3], "jd_boost_active": bool(jd_text)}

# ─────────────────────────────────────────────────────────────────────────────
# RESUME TAILORING — simple LLM call, works with any provider
# ─────────────────────────────────────────────────────────────────────────────

TAILOR_PROMPT_TPL = """You are a professional resume writer. Tailor the candidate's resume for this specific job.

JOB: {title} @ {company}
Required Skills: {skills}
{jd_section}
RESUME:
{resume}

Produce ONLY this exact format:

--- TAILORED PROFESSIONAL SUMMARY ---
[3-4 sentences, ATS-optimized, written for this role]

--- 2 MODIFIED EXPERIENCE BULLETS ---
* [enhance an existing bullet to match this role]
* [enhance another existing bullet to match this role]

--- HIGHLIGHTED ALIGNED SKILLS ---
[5-7 comma-separated skills most relevant to this job]

Do not rewrite the full resume. Do not fabricate new facts."""

def tailor_with_anthropic(prompt, api_key):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    r = client.messages.create(model=PROVIDER_MODELS["anthropic"], max_tokens=900,
                                messages=[{"role":"user","content":prompt}])
    return r.content[0].text

def tailor_with_openai(prompt, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(model=PROVIDER_MODELS["openai"], max_tokens=900,
        messages=[{"role":"user","content":prompt}])
    return r.choices[0].message.content

def tailor_with_groq(prompt, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    r = client.chat.completions.create(model=PROVIDER_MODELS["groq"], max_tokens=900,
        messages=[{"role":"user","content":prompt}])
    return r.choices[0].message.content

def tailor_with_gemini(prompt, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(PROVIDER_MODELS["gemini"])
    r = model.generate_content(prompt)
    return r.text

TAILOR_FNS = {"anthropic": tailor_with_anthropic, "openai": tailor_with_openai,
               "groq": tailor_with_groq, "gemini": tailor_with_gemini}

def tailor_resume_llm(job, resume, candidate_skills, api_key, provider, jd_text=""):
    jd_section = (f"\nFULL JOB DESCRIPTION:\n{jd_text.strip()}\n"
                  if jd_text and jd_text.strip()
                  else f"\nJob Description:\n{job['Job Description']}\n")
    prompt = TAILOR_PROMPT_TPL.format(
        title=job["Job Title"], company=job["Company"],
        skills=job["Required Skills"], jd_section=jd_section, resume=resume)
    text = TAILOR_FNS[provider](prompt, api_key)
    return {"tailored_output": text, "job_tailored_for": job["Job Title"],
            "company": job["Company"], "used_custom_jd": bool(jd_text and jd_text.strip())}

# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC AGENTIC LOOP (native tool-calling)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {"name":"filter_jobs","description":"Filter jobs by location, experience cap, and exclusions.",
     "input_schema":{"type":"object","properties":{
         "jobs":{"type":"array","items":{"type":"object"}},
         "location_preference":{"type":"array","items":{"type":"string"}},
         "max_experience_years":{"type":"integer"},
         "excluded_companies":{"type":"array","items":{"type":"string"}}},
     "required":["jobs","location_preference","max_experience_years","excluded_companies"]}},
    {"name":"rank_jobs","description":"Score and rank filtered jobs by skill match, experience, location, and optional JD.",
     "input_schema":{"type":"object","properties":{
         "jobs":{"type":"array","items":{"type":"object"}},
         "candidate_skills":{"type":"array","items":{"type":"string"}},
         "candidate_experience_years":{"type":"integer"},
         "jd_text":{"type":"string"}},
     "required":["jobs","candidate_skills","candidate_experience_years"]}},
    {"name":"tailor_resume","description":"Tailor Professional Summary and 2 bullets for the top job.",
     "input_schema":{"type":"object","properties":{
         "job":{"type":"object"},
         "resume":{"type":"string"},
         "candidate_skills":{"type":"array","items":{"type":"string"}},
         "jd_text":{"type":"string"}},
     "required":["job","resume","candidate_skills"]}},
]

def run_anthropic_agent(profile, resume, api_key, emit, all_jobs, jd_text):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    jd_instruction = ""
    if jd_text:
        jd_instruction = f"\nCUSTOM JD PROVIDED — pass jd_text to rank_jobs and tailor_resume:\n{jd_text}\n"

    system = ("You are an intelligent job search agent. Follow this pipeline:\n"
              "Step 1: filter_jobs | Step 2: rank_jobs | Step 3: tailor_resume\n"
              "Be concise. Explain your reasoning at each step.")
    user = (f"Run the full job search pipeline.\nPROFILE: {json.dumps(profile)}\n"
            f"RESUME:\n{resume}\nJOBS: {len(all_jobs)} loaded.\n{jd_instruction}\n"
            "Pipeline: filter → rank → tailor. Give a brief final summary.")

    messages = [{"role":"user","content":user}]
    step = 0
    while True:
        step += 1
        emit("step", {"step":step,"message":"Agent thinking..."})
        resp = client.messages.create(model=PROVIDER_MODELS["anthropic"], max_tokens=4096,
                                       system=system, tools=TOOLS, messages=messages)
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                emit("reasoning", {"text":b.text.strip(),"step":step})

        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        if not tool_blocks or resp.stop_reason == "end_turn":
            emit("final", {"text":" ".join(b.text for b in resp.content if b.type=="text")})
            break

        results = []
        for tb in tool_blocks:
            emit("tool_call", {"tool":tb.name,"step":step})
            if tb.name == "filter_jobs":
                r = filter_jobs(all_jobs, tb.input["location_preference"],
                                tb.input["max_experience_years"], tb.input["excluded_companies"])
                emit("filter_result", r)
            elif tb.name == "rank_jobs":
                r = rank_jobs(tb.input["jobs"], tb.input["candidate_skills"],
                              tb.input["candidate_experience_years"],
                              tb.input.get("jd_text", jd_text))
                emit("rank_result", r)
            elif tb.name == "tailor_resume":
                r = tailor_resume_llm(tb.input["job"], tb.input["resume"],
                                      tb.input["candidate_skills"], api_key, "anthropic",
                                      tb.input.get("jd_text", jd_text))
                emit("tailor_result", r)
            else:
                r = {"error": f"Unknown tool: {tb.name}"}
            results.append({"type":"tool_result","tool_use_id":tb.id,"content":json.dumps(r)})

        messages.append({"role":"assistant","content":resp.content})
        messages.append({"role":"user","content":results})

# ─────────────────────────────────────────────────────────────────────────────
# GROQ / GEMINI AGENTIC LOOP (simulated tool-calling via structured prompting)
# These models get a step-by-step prompt that makes them decide and execute
# each pipeline stage, with the Python tools doing the actual computation.
# ─────────────────────────────────────────────────────────────────────────────

def call_llm_simple(provider, api_key, prompt):
    """Single-turn LLM call, returns text. Used for non-Anthropic providers."""
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI(api_key=api_key).chat.completions.create(
            model=PROVIDER_MODELS["openai"], max_tokens=1200,
            messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content
    if provider == "groq":
        from groq import Groq
        r = Groq(api_key=api_key).chat.completions.create(
            model=PROVIDER_MODELS["groq"], max_tokens=1200,
            messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(PROVIDER_MODELS["gemini"]).generate_content(prompt).text
    raise ValueError(f"Unknown provider: {provider}")

def run_simple_agent(profile, resume, api_key, provider, emit, all_jobs, jd_text):
    """
    Simulated agentic loop for Groq/Gemini:
    - Python handles all filtering and ranking (deterministic)
    - LLM is called for reasoning explanations and resume tailoring
    - This keeps costs at zero for free-tier providers
    """
    emit("reasoning", {"text": f"Using {provider.upper()} ({PROVIDER_MODELS[provider]}) — running pipeline with LLM reasoning.", "step":1})

    # ── Step 1: Filter ──
    emit("step", {"step":1, "message":"Filtering jobs..."})
    emit("tool_call", {"tool":"filter_jobs","step":1})
    filter_result = filter_jobs(
        all_jobs,
        profile.get("location_preference", ["Any"]),
        profile.get("max_experience_required", 3),
        profile.get("excluded_companies", [])
    )
    emit("filter_result", filter_result)

    # Ask LLM to explain the filter decision
    filter_prompt = (
        f"You are a job search agent. You just filtered {len(all_jobs)} jobs and "
        f"{filter_result['passed_count']} passed based on: location={profile.get('location_preference')}, "
        f"max_experience={profile.get('max_experience_required')} years, "
        f"excluded_companies={profile.get('excluded_companies')}. "
        f"Write 2 sentences explaining why these filters make sense for this candidate: {profile.get('name')}."
    )
    reason1 = call_llm_simple(provider, api_key, filter_prompt)
    emit("reasoning", {"text": reason1, "step":1})

    # ── Step 2: Rank ──
    emit("step", {"step":2, "message":"Ranking jobs..."})
    emit("tool_call", {"tool":"rank_jobs","step":2})
    rank_result = rank_jobs(
        filter_result["filtered_jobs"],
        profile.get("skills", []),
        profile.get("years_experience", 1),
        jd_text
    )
    emit("rank_result", rank_result)

    top = rank_result["top_3"]
    rank_prompt = (
        f"You are a job search agent. After scoring {filter_result['passed_count']} jobs, "
        f"the top 3 are: "
        f"1) {top[0]['Job Title']} @ {top[0]['Company']} (score {top[0]['score']}), "
        f"2) {top[1]['Job Title']} @ {top[1]['Company']} (score {top[1]['score']}), "
        f"3) {top[2]['Job Title']} @ {top[2]['Company']} (score {top[2]['score']}). "
        f"The candidate has these skills: {', '.join(profile.get('skills',[])[:10])}. "
        f"Write 2-3 sentences explaining why the top job is the best match."
    )
    reason2 = call_llm_simple(provider, api_key, rank_prompt)
    emit("reasoning", {"text": reason2, "step":2})

    # ── Step 3: Tailor ──
    emit("step", {"step":3, "message":"Tailoring resume..."})
    emit("tool_call", {"tool":"tailor_resume","step":3})
    top_job = rank_result["top_3"][0]
    tailor_result = tailor_resume_llm(top_job, resume, profile.get("skills",[]),
                                      api_key, provider, jd_text)
    emit("tailor_result", tailor_result)

    # Final summary
    summary_prompt = (
        f"Summarize in 3 sentences: a job search agent filtered {len(all_jobs)} jobs to "
        f"{filter_result['passed_count']}, ranked them, and the top match is "
        f"{top_job['Job Title']} at {top_job['Company']} with a score of {top_job['score']}/100. "
        f"The resume was tailored for this role. Make it sound like an agent completing its mission."
    )
    summary = call_llm_simple(provider, api_key, summary_prompt)
    emit("final", {"text": summary})

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_streaming(profile, resume, api_key, provider, event_queue, jd_text="", custom_jobs=None):
    def emit(event_type, data):
        event_queue.put({"type": event_type, "data": data})
    try:
        all_jobs = custom_jobs if custom_jobs else load_jobs()
        source = "uploaded CSV" if custom_jobs else "built-in dataset"
        emit("status", {"message": f"Loaded {len(all_jobs)} jobs from {source} | Provider: {provider.upper()}", "step":0})
        if jd_text:
            emit("status", {"message":"Custom JD detected — boosting ranking + tailoring", "step":0})

        if provider == "anthropic":
            run_anthropic_agent(profile, resume, api_key, emit, all_jobs, jd_text)
        else:
            run_simple_agent(profile, resume, api_key, provider, emit, all_jobs, jd_text)

        emit("done", {"message":"Pipeline complete"})
    except Exception as e:
        emit("error", {"message": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────


# ── Upload helpers ────────────────────────────────────────────────────────────

@app.route("/api/extract_pdf", methods=["POST"])
def extract_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(f.read())) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        text = text.strip()
        if not text:
            return jsonify({"error": "Could not extract text from PDF"}), 400
        word_count = len(text.split())
        return jsonify({"text": text, "word_count": word_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/parse_csv", methods=["POST"])
def parse_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "File must be a CSV"}), 400
    try:
        content_bytes = f.read().decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(content_bytes))
        rows = list(reader)
        required = ["Job Title", "Company", "Location", "Required Skills",
                    "Years Experience", "Job Description", "URL"]
        missing = [r for r in required if r not in reader.fieldnames]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        for i, row in enumerate(rows):
            row["id"] = i + 1
            try:
                row["Years Experience"] = int(row.get("Years Experience", 0))
            except:
                row["Years Experience"] = 0
        return jsonify({"jobs": rows, "count": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate_resume_pdf", methods=["POST"])
def generate_resume_pdf():
    import io, re, os, subprocess, tempfile

    data            = request.json
    tailored_output = data.get("tailored_output", "")
    original_resume = data.get("resume", "")
    job_title       = data.get("job_title", "")
    company         = data.get("company", "")
    fmt             = data.get("format", "pdf")

    # ── helpers ──────────────────────────────────────────────────────────
    def tex(s):
        s = str(s)
        for old, new in [
            ("\\", "BACKSLASH_PLACEHOLDER"),
            ("&",  "\\&"), ("%", "\\%"), ("$", "\\$"),
            ("#",  "\\#"), ("_", "\\_"), ("{", "\\{"),
            ("}",  "\\}"), ("~", "\\textasciitilde{}"),
            ("^",  "\\textasciicircum{}"),
            ("BACKSLASH_PLACEHOLDER", "\\textbackslash{}"),
        ]:
            s = s.replace(old, new)
        return s

    def extract(text, marker):
        parts = re.split(r"-{2,}\s*[A-Z][A-Z ]*[A-Z]\s*-{2,}", text)
        headers = re.findall(r"-{2,}\s*([A-Z][A-Z ]*[A-Z])\s*-{2,}", text)
        for i, h in enumerate(headers):
            if marker.upper() in h.upper() and i + 1 < len(parts):
                return parts[i + 1].strip()
        return ""

    summary  = extract(tailored_output, "TAILORED PROFESSIONAL SUMMARY")
    bul_raw  = extract(tailored_output, "MODIFIED EXPERIENCE BULLETS")
    skills   = extract(tailored_output, "HIGHLIGHTED ALIGNED SKILLS")

    # Clean summary — strip any stray bullet lines
    summary  = re.sub(r"(?m)^[*•\-].*$", "", summary).strip()
    summary  = re.sub(r"\s+", " ", summary).strip()

    bullets  = [l.strip().lstrip("*•- ").strip()
                for l in bul_raw.splitlines()
                if l.strip().startswith(("*", "•", "-"))]

    # ── parse resume ─────────────────────────────────────────────────────
    lines = [l.strip() for l in original_resume.splitlines()
             if l.strip() and not l.strip().startswith("(cid:")]

    name_line    = lines[0] if lines else "Candidate"
    contact_line = ""
    for l in lines[1:5]:
        if "@" in l or "linkedin" in l.lower() or "|" in l:
            contact_line = l
            break

    SKIP = {"SUMMARY", "OBJECTIVE", "PROFILE"}
    SECS = ["EDUCATION", "EXPERIENCE", "PROJECT", "SKILL", "PUBLICATION", "CERTIFICATION"]
    sections, cur_sec, cur_lines = {}, None, []
    for line in lines:
        upper = line.upper()
        if any(kw in upper for kw in SKIP):
            continue
        if any(kw in upper for kw in SECS) and len(line) < 45:
            if cur_sec:
                sections[cur_sec] = cur_lines
            cur_sec, cur_lines = upper, []
        elif cur_sec:
            cur_lines.append(line)
    if cur_sec:
        sections[cur_sec] = cur_lines

    # ── build LaTeX using list of strings ────────────────────────────────
    L = []

    def add(*args):
        for a in args:
            L.append(a)

    add(
        r"\documentclass[letterpaper,10pt]{article}",
        r"\usepackage[left=0.55in,right=0.55in,top=0.45in,bottom=0.45in]{geometry}",
        r"\usepackage{latexsym,titlesec,enumitem,hyperref,fancyhdr,tabularx}",
        r"\usepackage[usenames,dvipsnames]{color}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\input{glyphtounicode}",
        r"\hypersetup{colorlinks=false}",
        r"\pagestyle{fancy}\fancyhf{}\fancyfoot{}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\renewcommand{\footrulewidth}{0pt}",
        r"\raggedbottom\raggedright",
        r"\setlength{\tabcolsep}{0in}",
        r"\titleformat{\section}{\vspace{-4pt}\scshape\raggedright\large}{}{0em}{}[\color{black}\titlerule\vspace{-5pt}]",
        r"\pdfgentounicode=1",
        r"\newcommand{\resumeItem}[1]{\item\small{#1\vspace{-2pt}}}",
        r"\newcommand{\resumeSubheading}[4]{\vspace{-2pt}\item",
        r"  \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}",
        r"    \textbf{#1} & #2 \\",
        r"    \textit{\small#3} & \textit{\small#4} \\",
        r"  \end{tabular*}\vspace{-7pt}}",
        r"\newcommand{\resumeProjectHeading}[2]{\item",
        r"  \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}",
        r"    \small#1 & #2 \\",
        r"  \end{tabular*}\vspace{-7pt}}",
        r"\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in,label={}]}",
        r"\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}",
        r"\newcommand{\resumeItemListStart}{\begin{itemize}}",
        r"\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}",
        r"\begin{document}",
    )

    # Header
    add(r"\begin{center}")
    add(f"  {{\\Huge \\scshape {tex(name_line)}}}")
    if contact_line.strip():
        add(f"  \\\\[1pt] \\small {tex(contact_line)}")
    add(f"  \\\\[1pt] \\textit{{\\small Tailored for: {tex(job_title)} at {tex(company)}}}")
    add(r"\end{center}", r"\vspace{-8pt}")

    # Summary
    if summary:
        add(r"\section{Summary}", r"\resumeSubHeadingListStart")
        add(f"  \\item \\small{{{tex(summary)}}}")
        add(r"\resumeSubHeadingListEnd")

    # Education
    edu_key = next((k for k in sections if "EDUCATION" in k), None)
    if edu_key:
        add(r"\section{Education}", r"\resumeSubHeadingListStart")
        edu_lines = sections[edu_key]
        i = 0
        while i < len(edu_lines):
            l = edu_lines[i]
            if not l:
                i += 1; continue
            nxt = edu_lines[i+1] if i+1 < len(edu_lines) else ""
            dm = re.search(r"(\w[\w\s]*\d{4})\s*[–-]+\s*(\w[\w\s]*\d{4})", l)
            if any(x in l for x in ["University","College","Institute","School"]):
                date_str = dm.group(0) if dm else ""
                inst = l.replace(date_str,"").strip(" –-") if date_str else l
                degree = nxt if nxt and not any(x in nxt.upper() for x in ["UNIVERSITY","COLLEGE"]) else ""
                add(f"  \\resumeSubheading{{{tex(inst)}}}{{{tex(date_str)}}}{{{tex(degree)}}}{{}}")
                i += 2 if degree else 1
            else:
                i += 1
        add(r"\resumeSubHeadingListEnd")

    # Experience
    exp_key = next((k for k in sections if "EXPERIENCE" in k), None)
    if exp_key:
        add(r"\section{Experience}", r"\resumeSubHeadingListStart")
        injected = 0
        in_items = False
        for l in sections[exp_key]:
            if not l: continue
            is_bul = l.startswith(("*","•","-","·"))
            dm = re.search(r"(\w{3}\s*\d{4})\s*[–-]+\s*(\w{3}\s*\d{4}|Present|present)", l)
            if is_bul:
                if not in_items:
                    add(r"  \resumeItemListStart")
                    in_items = True
                if injected < len(bullets):
                    add(f"    \\resumeItem{{{tex(bullets[injected])}}}")
                    injected += 1
                else:
                    add(f"    \\resumeItem{{{tex(l.lstrip('*•-· '))}}}")
            else:
                if in_items:
                    add(r"  \resumeItemListEnd")
                    in_items = False
                if dm:
                    rest = l.replace(dm.group(0),"").strip(" –-")
                    add(f"  \\resumeSubheading{{{tex(rest)}}}{{{tex(dm.group(0))}}}{{}}{{}}")
                elif any(x in l for x in ["Intern","Engineer","Developer","Analyst","Scientist","Manager"]):
                    add(f"  \\resumeSubheading{{{tex(l)}}}{{}}{{}}{{}}")
                else:
                    add(f"  \\resumeSubheading{{{tex(l)}}}{{}}{{}}{{}}")
        if in_items:
            add(r"  \resumeItemListEnd")
        add(r"\resumeSubHeadingListEnd")

    # Projects
    proj_key = next((k for k in sections if "PROJECT" in k), None)
    if proj_key:
        add(r"\section{Projects}", r"\resumeSubHeadingListStart")
        in_items = False
        for l in sections[proj_key]:
            if not l: continue
            if l.startswith(("*","•","-","·")):
                if not in_items:
                    add(r"  \resumeItemListStart")
                    in_items = True
                add(f"    \\resumeItem{{{tex(l.lstrip('*•-· '))}}}")
            else:
                if in_items:
                    add(r"  \resumeItemListEnd")
                    in_items = False
                add(f"  \\resumeProjectHeading{{\\textbf{{{tex(l)}}}}}{{}}")
        if in_items:
            add(r"  \resumeItemListEnd")
        add(r"\resumeSubHeadingListEnd")

    # Skills
    if skills:
        add(r"\section{Technical Skills}", r"\resumeSubHeadingListStart")
        add(f"  \\item \\small{{\\textbf{{Aligned Skills}}: {tex(skills)}}}")
        add(r"\resumeSubHeadingListEnd")

    add(r"\end{document}")

    latex = "\n".join(L)

    # Return LaTeX
    if fmt == "latex":
        from flask import make_response
        resp = make_response(latex)
        resp.headers["Content-Type"] = "text/plain; charset=utf-8"
        resp.headers["Content-Disposition"] = "attachment; filename=resume_tailored.tex"
        return resp

    # Compile to PDF
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "resume.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex)
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                capture_output=True, timeout=30
            )
            pdf_path = os.path.join(tmpdir, "resume.pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                from flask import make_response
                resp = make_response(pdf_bytes)
                resp.headers["Content-Type"] = "application/pdf"
                resp.headers["Content-Disposition"] = f"attachment; filename=Resume_{job_title.replace(' ','-')}_Tailored.pdf"
                return resp
            else:
                # fallback to latex
                from flask import make_response
                resp = make_response(latex)
                resp.headers["Content-Type"] = "text/plain; charset=utf-8"
                resp.headers["Content-Disposition"] = "attachment; filename=resume_tailored.tex"
                return resp
    except Exception as e:
        from flask import make_response
        resp = make_response(latex)
        resp.headers["Content-Type"] = "text/plain; charset=utf-8"
        resp.headers["Content-Disposition"] = "attachment; filename=resume_tailored.tex"
        return resp


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/defaults")
def defaults():
    return jsonify({"profile": DEFAULT_PROFILE, "resume": DEFAULT_RESUME})

@app.route("/api/run", methods=["POST"])
def run():
    data     = request.json
    api_key  = data.get("api_key","").strip()
    provider = data.get("provider","anthropic").strip().lower()
    if not api_key:
        return jsonify({"error":"API key required"}), 400
    if provider not in PROVIDER_MODELS:
        return jsonify({"error":f"Unknown provider: {provider}"}), 400

    profile     = data.get("profile", DEFAULT_PROFILE)
    resume      = data.get("resume",  DEFAULT_RESUME)
    jd_text     = data.get("jd_text","").strip()
    custom_jobs = data.get("custom_jobs", None)  # uploaded CSV jobs

    q = queue.Queue()
    def stream():
        t = threading.Thread(target=run_agent_streaming,
                             args=(profile, resume, api_key, provider, q, jd_text))
        t.daemon = True; t.start()
        while True:
            try:
                event = q.get(timeout=120)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("done","error"): break
            except queue.Empty:
                yield f"data: {json.dumps({'type':'error','data':{'message':'Timeout'}})}\n\n"
                break
    return Response(stream_with_context(stream()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/jobs")
def jobs():
    return jsonify(load_jobs())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
