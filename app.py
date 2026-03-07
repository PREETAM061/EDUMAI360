"""EduMate 360 v5.1 — All bugs fixed"""
import os, json, re, random, sqlite3
from datetime import datetime, date
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DB_PATH = "edumate360.db"

def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS stats (
            session_id TEXT PRIMARY KEY, streak INTEGER DEFAULT 0,
            wrong_in_row INTEGER DEFAULT 0, accuracy REAL DEFAULT 0,
            total_q INTEGER DEFAULT 0, correct_q INTEGER DEFAULT 0,
            total_time REAL DEFAULT 0, attempts INTEGER DEFAULT 0,
            emotion TEXT DEFAULT 'focused', current_level_idx INTEGER DEFAULT 2,
            xp INTEGER DEFAULT 0, score INTEGER DEFAULT 0,
            badges TEXT DEFAULT '[]', topics_studied TEXT DEFAULT '[]',
            weak_topics TEXT DEFAULT '[]', study_dates TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
            topic TEXT, question TEXT, correct INTEGER, time_sec REAL,
            difficulty TEXT, timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS spaced_repetition (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
            topic TEXT, next_review REAL, interval_days INTEGER DEFAULT 1,
            ease_factor REAL DEFAULT 2.5, repetitions INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
            score INTEGER, xp INTEGER, accuracy REAL, topics INTEGER, timestamp TEXT,
            session_id TEXT DEFAULT ""
        );
        """)
        # Migrate existing leaderboard table if session_id column missing
        try:
            db.execute("ALTER TABLE leaderboard ADD COLUMN session_id TEXT DEFAULT ''")
            db.commit()
        except Exception:
            pass  # Column already exists
init_db()

def load_stats(sid):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM stats WHERE session_id=?", (sid,)).fetchone()
        if not row:
            db.execute("INSERT INTO stats (session_id) VALUES (?)", (sid,))
            db.commit()
            return {"session_id":sid,"streak":0,"wrong_in_row":0,"accuracy":0.0,
                    "total_q":0,"correct_q":0,"total_time":0.0,"attempts":0,
                    "emotion":"focused","current_level_idx":2,"xp":0,"score":0,
                    "badges":[],"topics_studied":[],"weak_topics":[],"study_dates":[]}
        d = dict(row)
        for f in ["badges","topics_studied","weak_topics","study_dates"]:
            try: d[f] = json.loads(d[f]) if d[f] else []
            except: d[f] = []
        return d

def save_stats(s):
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""INSERT OR REPLACE INTO stats
            (session_id,streak,wrong_in_row,accuracy,total_q,correct_q,total_time,
             attempts,emotion,current_level_idx,xp,score,badges,topics_studied,weak_topics,study_dates)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (s["session_id"],s["streak"],s["wrong_in_row"],s["accuracy"],s["total_q"],
             s["correct_q"],s["total_time"],s["attempts"],s["emotion"],
             s["current_level_idx"],s["xp"],s["score"],
             json.dumps(s["badges"]),json.dumps(s["topics_studied"]),
             json.dumps(s["weak_topics"]),json.dumps(s["study_dates"])))
        db.commit()

def log_answer(sid, topic, question, correct, time_sec, difficulty):
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT INTO answers (session_id,topic,question,correct,time_sec,difficulty,timestamp) VALUES (?,?,?,?,?,?,?)",
            (sid, topic, question, int(correct), time_sec, difficulty, datetime.now().isoformat()))
        db.commit()

def get_progress_data(sid):
    """Accuracy over time — per answer, in order."""
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            "SELECT correct, topic FROM answers WHERE session_id=? ORDER BY timestamp ASC LIMIT 100",
            (sid,)).fetchall()
    data = []; cc = 0
    for i, row in enumerate(rows):
        if row[0]: cc += 1
        data.append({"n": i+1, "accuracy": round(cc/(i+1)*100, 1), "topic": row[1]})
    return data

def get_topic_stats(sid):
    """Per-topic breakdown — each topic separate."""
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """SELECT topic, COUNT(*) as total, SUM(correct) as correct_count, AVG(time_sec) as avg_time
               FROM answers WHERE session_id=?
               GROUP BY topic ORDER BY total DESC LIMIT 15""",
            (sid,)).fetchall()
    result = []
    for row in rows:
        total = row[1]; correct = int(row[2] or 0)
        result.append({
            "topic": row[0], "total": total, "correct": correct,
            "wrong": total - correct,
            "mastery": round(correct/total*100, 1) if total > 0 else 0,
            "avg_time": round(row[3] or 0, 1)
        })
    return result

def update_sr(sid, topic, correct):
    """SM-2 spaced repetition. next_review stored as REAL (unix timestamp)."""
    now = datetime.now().timestamp()
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute(
            "SELECT id, interval_days, ease_factor, repetitions FROM spaced_repetition WHERE session_id=? AND topic=?",
            (sid, topic)).fetchone()
        if row:
            rid, interval, ef, reps = row
            q = 5 if correct else 2
            ef = max(1.3, ef + 0.1 - (5-q)*(0.08 + (5-q)*0.02))
            if correct:
                reps += 1
                interval = 1 if reps == 1 else 6 if reps == 2 else max(1, round(interval * ef))
            else:
                reps = 0; interval = 1
            next_review = now + interval * 86400
            db.execute(
                "UPDATE spaced_repetition SET next_review=?,interval_days=?,ease_factor=?,repetitions=? WHERE id=?",
                (next_review, interval, ef, reps, rid))
        else:
            # FIX BUG 12: First time seeing topic — due in 1 day
            next_review = now + 86400
            db.execute(
                "INSERT INTO spaced_repetition (session_id,topic,next_review,interval_days,ease_factor,repetitions) VALUES (?,?,?,1,2.5,0)",
                (sid, topic, next_review))
        db.commit()

def get_due(sid):
    """Topics whose review date has passed."""
    now = datetime.now().timestamp()
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            "SELECT topic, interval_days FROM spaced_repetition WHERE session_id=? AND next_review <= ?",
            (sid, now)).fetchall()
    return [{"topic": r[0], "interval_days": r[1]} for r in rows]

class AdaptiveEngine:
    LEVELS = ["beginner", "easy", "medium", "hard", "expert"]
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
        self._train()
    def _train(self):
        random.seed(42); X, y = [], []
        for _ in range(60): X.append([random.randint(0,1),random.randint(2,6),random.uniform(10,42),random.uniform(55,120),random.randint(1,8),random.randint(1,2)]); y.append(0)
        for _ in range(80): X.append([random.randint(1,3),random.randint(0,2),random.uniform(43,74),random.uniform(22,55),random.randint(3,12),random.randint(0,1)]); y.append(1)
        for _ in range(60): X.append([random.randint(4,12),0,random.uniform(76,100),random.uniform(4,20),random.randint(5,20),0]); y.append(2)
        X = np.array(X); self.scaler.fit(X); self.clf.fit(self.scaler.transform(X), y)
    def recommend(self, s):
        e = {"focused":0,"confused":1,"stressed":2}.get(s.get("emotion","focused"), 0)
        feat = np.array([[s.get("streak",0), s.get("wrong_in_row",0), s.get("accuracy",50),
                          min(s.get("avg_time_sec",30),120), s.get("attempts",1), e]])
        action = int(self.clf.predict(self.scaler.transform(feat))[0])
        conf = float(max(self.clf.predict_proba(self.scaler.transform(feat))[0]))
        cur = s.get("current_level_idx", 2)
        if e >= 1 and action == 2: action = 1
        new_idx = max(0, min(4, cur + (1 if action==2 else -1 if action==0 else 0)))
        reasons = {0:"Several errors — reducing difficulty 💪", 1:"Steady performance — maintaining 📊", 2:"Strong streak — increasing difficulty 🚀"}
        return {"action":["easier","stay","harder"][action], "new_level":self.LEVELS[new_idx],
                "new_level_idx":new_idx, "confidence":round(conf,3), "reason":reasons[action]}

engine = AdaptiveEngine()

def make_prompt(topic, difficulty, num_q, q_type):
    type_desc = {
        "mcq": f"{num_q} multiple choice questions with 4 distinct options each",
        "short": f"{num_q} short answer questions",
        "truefalse": f"{num_q} true/false questions",
        "mixed": f"{num_q} mixed questions (some MCQ, some true/false)"
    }.get(q_type, f"{num_q} MCQ questions")
    return f"""You are an expert educational assessment creator.
Topic: {topic}
Difficulty: {difficulty}
Task: Generate {type_desc}

CRITICAL RULES — FOLLOW EXACTLY:
1. The "answer" field must be the EXACT text of one option, copied word-for-word
2. Never use "A", "B", "C", "D" as the answer value
3. Never use "All of the above" or "Both A and B" as options
4. Each question must have a clear single correct answer
5. Return ONLY valid JSON — no markdown, no code fences, no extra text

Required JSON format:
{{"topic":"{topic}","difficulty":"{difficulty}","questions":[{{"id":1,"type":"mcq","question":"...","options":["Option text 1","Option text 2","Option text 3","Option text 4"],"answer":"Option text 1","explanation":"Why this is correct","concept":"Key concept"}}]}}

For true/false: options must be exactly ["True","False"]
For short answer: options must be []"""

def safe_parse(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    s = raw.find("{"); e = raw.rfind("}") + 1
    if s >= 0 and e > s: raw = raw[s:e]
    return json.loads(raw)

def ai_generate(topic, difficulty, num_q=5, q_type="mcq"):
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set", "questions": []}
    for attempt in range(3):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":make_prompt(topic,difficulty,num_q,q_type)}],
                temperature=0.4, max_tokens=4000)
            data = safe_parse(r.choices[0].message.content)
            if "questions" not in data: raise ValueError("No questions key in response")
            # Validate and fix each question's answer
            for q in data["questions"]:
                opts = q.get("options", [])
                ans = q.get("answer", "")
                if opts and ans not in opts:
                    # Try case-insensitive match
                    matched = next((o for o in opts if o.lower() == ans.lower()), None)
                    q["answer"] = matched if matched else opts[0]
            return data
        except json.JSONDecodeError:
            if attempt == 2: return {"error": "AI returned malformed JSON — please retry", "questions": []}
        except Exception as ex:
            if attempt == 2: return {"error": str(ex), "questions": []}
    return {"error": "Failed after 3 attempts", "questions": []}

def ai_from_text(text, difficulty, num_q=5):
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set", "questions": []}
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Generate {num_q} {difficulty}-level MCQ questions ONLY from this material. Do not use outside knowledge.
Material: \"\"\"{text[:4000]}\"\"\"
Rules: answer must exactly match one option text. Return ONLY raw JSON:
{{"topic":"infer the topic","difficulty":"{difficulty}","questions":[{{"id":1,"type":"mcq","question":"...","options":["opt1","opt2","opt3","opt4"],"answer":"opt1","explanation":"...","concept":"..."}}]}}"""
        r = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=4000)
        data = safe_parse(r.choices[0].message.content)
        for q in data.get("questions", []):
            opts = q.get("options", [])
            ans = q.get("answer", "")
            if opts and ans not in opts:
                matched = next((o for o in opts if o.lower() == ans.lower()), None)
                q["answer"] = matched if matched else opts[0]
        return data
    except Exception as ex: return {"error": str(ex), "questions": []}

def ai_followup(topic, question, student_ans, correct_ans, explanation, was_correct):
    if not GROQ_AVAILABLE or not GROQ_API_KEY: return ""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        context = "Correct!" if was_correct else f"Incorrect (answered: '{student_ans}')"
        r = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":f"Topic:{topic}\nQ:{question}\nAnswer:{correct_ans}\n{context}\nExplanation:{explanation}\nWrite ONE follow-up question max 20 words. Return ONLY the question."}],
            temperature=0.8, max_tokens=60)
        return r.choices[0].message.content.strip()
    except: return ""

def award_badges(sess):
    new_b = []; have = sess["badges"]
    def give(b):
        if b not in have: have.append(b); new_b.append(b)
    if sess["streak"] >= 3: give("🔥 Hot Streak")
    if sess["streak"] >= 7: give("⚡ Lightning Mind")
    if sess["accuracy"] >= 90 and sess["total_q"] >= 5: give("🎯 Sharpshooter")
    if sess["score"] >= 500: give("💯 Score Master")
    if sess["correct_q"] >= 20: give("🎓 Scholar")
    if sess["xp"] >= 1000: give("🏆 Champion")
    if len(sess["topics_studied"]) >= 5: give("🌍 Explorer")
    if len(sess["study_dates"]) >= 7: give("📅 Consistent")
    return new_b

# ── ROUTES ──────────────────────────────────────────────────────────

@app.route("/")
def index(): return send_from_directory("frontend", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","groq":bool(GROQ_API_KEY),"model":"llama-3.3-70b-versatile",
                    "ml":"GradientBoostingClassifier","pdf":PDF_AVAILABLE,"db":"sqlite","version":"5.1"})

@app.route("/api/generate", methods=["POST"])
def generate():
    d = request.json or {}
    topic = d.get("topic","").strip(); difficulty = d.get("difficulty","medium")
    num_q = min(max(int(d.get("num_questions",5)),1),15)  # FIX BUG 11: allow up to 15
    q_type = d.get("type","mcq"); sid = d.get("session_id","default")
    if not topic: return jsonify({"error":"Topic required"}), 400
    if difficulty not in engine.LEVELS: difficulty = "medium"
    result = ai_generate(topic, difficulty, num_q, q_type)
    if not result.get("questions"): return jsonify({"error":result.get("error","Failed")}), 500
    sess = load_stats(sid); sess["attempts"] += 1
    if topic not in sess["topics_studied"]: sess["topics_studied"].append(topic)
    today = date.today().isoformat()
    if today not in sess["study_dates"]: sess["study_dates"].append(today)
    save_stats(sess)
    return jsonify({"topic":topic,"difficulty":difficulty,"questions":result["questions"],"total":len(result["questions"])})

@app.route("/api/generate-from-text", methods=["POST"])
def gen_from_text():
    d = request.json or {}
    text = d.get("text","").strip(); difficulty = d.get("difficulty","medium")
    num_q = min(max(int(d.get("num_questions",5)),1),15); sid = d.get("session_id","default")
    if len(text) < 50: return jsonify({"error":"Provide at least 50 characters"}), 400
    result = ai_from_text(text, difficulty, num_q)
    if not result.get("questions"): return jsonify({"error":result.get("error","Failed")}), 500
    sess = load_stats(sid); sess["attempts"] += 1
    topic = result.get("topic","Notes")
    if topic not in sess["topics_studied"]: sess["topics_studied"].append(topic)
    today = date.today().isoformat()
    if today not in sess["study_dates"]: sess["study_dates"].append(today)
    save_stats(sess)
    return jsonify({"topic":topic,"difficulty":difficulty,"questions":result["questions"],"total":len(result["questions"])})

@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files: return jsonify({"error":"No PDF uploaded"}), 400
    if not PDF_AVAILABLE: return jsonify({"error":"PyPDF2 not installed"}), 500
    file = request.files["pdf"]; difficulty = request.form.get("difficulty","medium")
    num_q = min(max(int(request.form.get("num_questions",5)),1),15)
    sid = request.form.get("session_id","default")
    try:
        reader = PyPDF2.PdfReader(file); text = ""
        for page in reader.pages[:10]: text += page.extract_text() or ""
        if len(text.strip()) < 50: return jsonify({"error":"Could not extract text from PDF"}), 400
        result = ai_from_text(text, difficulty, num_q)
        if not result.get("questions"): return jsonify({"error":result.get("error","Failed")}), 500
        sess = load_stats(sid); sess["attempts"] += 1
        topic = result.get("topic","PDF Upload")
        if topic not in sess["topics_studied"]: sess["topics_studied"].append(topic)
        save_stats(sess)
        return jsonify({"topic":topic,"difficulty":difficulty,"questions":result["questions"],
                        "total":len(result["questions"]),"pages_read":len(reader.pages)})
    except Exception as ex: return jsonify({"error":str(ex)}), 500

@app.route("/api/submit", methods=["POST"])
def submit():
    d = request.json or {}
    sid = d.get("session_id","default")
    correct = bool(d.get("correct", False))
    time_sec = float(d.get("time_sec", 30))
    topic = d.get("topic","").strip()

    sess = load_stats(sid)
    sess["total_q"] += 1
    sess["total_time"] += time_sec

    PTS = {"beginner":5,"easy":10,"medium":20,"hard":30,"expert":50}
    cur_diff = engine.LEVELS[sess["current_level_idx"]]

    if correct:
        sess["streak"] += 1
        sess["correct_q"] += 1
        sess["wrong_in_row"] = 0
        pts = PTS.get(cur_diff, 20)
        sess["score"] += pts
        sess["xp"] += pts
        # FIX BUG 7: remove topic from weak_topics when answered correctly
        if topic and topic in sess["weak_topics"]:
            sess["weak_topics"].remove(topic)
    else:
        sess["streak"] = 0
        sess["wrong_in_row"] += 1
        # Only add to weak topics if got wrong 2+ times (not just once)
        # FIX BUG 7: Only mark as weak if accuracy on this topic < 50%
        if topic:
            with sqlite3.connect(DB_PATH) as db:
                row = db.execute(
                    "SELECT COUNT(*), SUM(correct) FROM answers WHERE session_id=? AND topic=?",
                    (sid, topic)).fetchone()
            total_t = row[0] or 0; correct_t = int(row[1] or 0)
            # After this answer, topic is weak if <50% accuracy and at least 2 attempts
            if total_t >= 1:  # current answer not logged yet
                new_acc = correct_t / (total_t + 1) * 100  # include current wrong
                if new_acc < 50 and topic not in sess["weak_topics"]:
                    sess["weak_topics"].append(topic)

    # Recalculate accuracy correctly
    sess["accuracy"] = round(sess["correct_q"] / sess["total_q"] * 100, 1) if sess["total_q"] > 0 else 0.0
    avg_time = sess["total_time"] / sess["total_q"]

    ml = engine.recommend({
        "streak": sess["streak"], "wrong_in_row": sess["wrong_in_row"],
        "accuracy": sess["accuracy"], "avg_time_sec": avg_time,
        "attempts": sess["attempts"], "emotion": sess["emotion"],
        "current_level_idx": sess["current_level_idx"]
    })
    sess["current_level_idx"] = ml["new_level_idx"]

    # Update study_dates (for streak calendar)
    today = date.today().isoformat()
    if today not in sess["study_dates"]:
        sess["study_dates"].append(today)
        # Keep only last 365 days
        sess["study_dates"] = sess["study_dates"][-365:]

    # Update topics_studied (for home stats + review screen)
    if topic and topic not in sess["topics_studied"]:
        sess["topics_studied"].append(topic)

    # Log answer BEFORE calling update_sr so topic stats are correct
    log_answer(sid, topic, d.get("question",""), correct, time_sec, cur_diff)
    update_sr(sid, topic, correct)

    followup = ai_followup(topic, d.get("question",""), d.get("student_answer",""),
                           d.get("correct_answer",""), d.get("explanation",""), correct)
    new_badges = award_badges(sess)
    save_stats(sess)

    return jsonify({
        "correct": correct,
        "new_difficulty": ml["new_level"],
        "ml_action": ml["action"],
        "ml_reason": ml["reason"],
        "ml_confidence": ml["confidence"],
        "followup": followup,
        "new_badges": new_badges,
        "due_topics": get_due(sid),
        "stats": {
            "streak": sess["streak"],
            "accuracy": sess["accuracy"],
            "score": sess["score"],
            "xp": sess["xp"],
            "level": sess["xp"] // 200 + 1,
            "total_q": sess["total_q"],
            "correct_q": sess["correct_q"],
            "weak_topics": sess["weak_topics"][-5:],
            "badges": sess["badges"]
        }
    })

@app.route("/api/emotion", methods=["POST"])
def emotion():
    d = request.json or {}
    sess = load_stats(d.get("session_id","default"))
    sess["emotion"] = d.get("emotion","focused")
    save_stats(sess)
    return jsonify({"ok": True})

@app.route("/api/stats/<sid>")
def stats(sid):
    s = load_stats(sid)
    progress = get_progress_data(sid)
    topic_stats = get_topic_stats(sid)
    due = get_due(sid)
    # FIX BUG 10: recalculate accuracy from DB directly to ensure it's correct
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("SELECT COUNT(*), SUM(correct) FROM answers WHERE session_id=?", (sid,)).fetchone()
    total_from_db = row[0] or 0
    correct_from_db = int(row[1] or 0)
    # Sync stats with DB if out of sync
    if total_from_db > 0 and (s["total_q"] != total_from_db or s["correct_q"] != correct_from_db):
        s["total_q"] = total_from_db
        s["correct_q"] = correct_from_db
        s["accuracy"] = round(correct_from_db / total_from_db * 100, 1)
        save_stats(s)
    return jsonify({
        "streak": s["streak"], "accuracy": s["accuracy"], "score": s["score"],
        "xp": s["xp"], "level": s["xp"]//200+1,
        "total_q": s["total_q"], "correct_q": s["correct_q"],
        "badges": s["badges"], "weak_topics": s["weak_topics"],
        "topics_studied": s["topics_studied"], "current_diff": engine.LEVELS[s["current_level_idx"]],
        "study_dates": s["study_dates"], "progress_data": progress,
        "topic_stats": topic_stats, "due_topics": due
    })

@app.route("/api/leaderboard", methods=["GET","POST"])
def leaderboard():
    if request.method == "POST":
        d = request.json or {}
        name  = str(d.get("name","Anonymous"))[:30].strip()
        sid   = str(d.get("session_id", ""))[:40]
        score = int(d.get("score", 0))
        xp    = int(d.get("xp", 0))
        acc   = float(d.get("accuracy", 0))
        tops  = int(d.get("topics", 0))
        if not name:
            return jsonify({"error":"Name required"}), 400
        with sqlite3.connect(DB_PATH) as db:
            # Check if this session already has an entry
            existing = db.execute(
                "SELECT id, score FROM leaderboard WHERE session_id=?", (sid,)
            ).fetchone() if sid else None

            if existing:
                # Only update if new score is higher
                if score > existing[1]:
                    db.execute(
                        "UPDATE leaderboard SET name=?,score=?,xp=?,accuracy=?,topics=?,timestamp=? WHERE id=?",
                        (name, score, xp, acc, tops, datetime.now().isoformat(), existing[0]))
                    db.commit()
                # Return rank based on current (possibly unchanged) score
                final_score = max(score, existing[1])
            else:
                # New entry — add session_id column value
                db.execute(
                    "INSERT INTO leaderboard (name,score,xp,accuracy,topics,timestamp,session_id) VALUES (?,?,?,?,?,?,?)",
                    (name, score, xp, acc, tops, datetime.now().isoformat(), sid))
                db.commit()
                final_score = score
            rank_row = db.execute(
                "SELECT COUNT(*) FROM leaderboard WHERE score > ?", (final_score,)
            ).fetchone()
        return jsonify({"status":"saved", "rank": int(rank_row[0]) + 1})
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            "SELECT name,score,xp,accuracy,topics FROM leaderboard ORDER BY score DESC LIMIT 10").fetchall()
    return jsonify({"leaderboard":[
        {"rank":i+1,"name":r[0],"score":r[1],"xp":r[2],"accuracy":r[3],"topics":r[4]}
        for i,r in enumerate(rows)
    ]})

@app.route("/api/due-topics/<sid>")
def due_topics(sid): return jsonify({"due": get_due(sid)})

@app.route("/api/export-report/<sid>")
def export_report(sid):
    s = load_stats(sid)
    # Sync from DB before export
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("SELECT COUNT(*), SUM(correct) FROM answers WHERE session_id=?", (sid,)).fetchone()
    total = row[0] or 0; correct = int(row[1] or 0)
    acc = round(correct/total*100,1) if total > 0 else 0
    return jsonify({
        "generated": datetime.now().isoformat(),
        "session": sid,
        "version": "5.1",
        "summary": {
            "score": s["score"], "xp": s["xp"],
            "level": s["xp"]//200+1,
            "accuracy": acc,
            "total_questions": total,
            "correct": correct,
            "wrong": total - correct
        },
        "topics_studied": s["topics_studied"],
        "weak_topics": s["weak_topics"],
        "study_dates": s["study_dates"],
        "topic_stats": get_topic_stats(sid),
        "badges_earned": s["badges"]
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
