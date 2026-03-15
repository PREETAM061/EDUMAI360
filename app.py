"""EduMate 360 v7.0 — build:20260315-110149"""
import os, json, re, random, sqlite3, hashlib, hmac, base64, time
from functools import wraps
from datetime import datetime, date
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    import warnings; warnings.filterwarnings('ignore')
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

app = Flask(__name__)
CORS(app)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Module-level Groq client — used by ai_chat() route
_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY","") else None
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
        CREATE TABLE IF NOT EXISTS mistakes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     TEXT,
            topic          TEXT,
            concept        TEXT,
            question       TEXT,
            your_answer    TEXT,
            correct_answer TEXT,
            explanation    TEXT,
            date           TEXT,
            retried        INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS emotion_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     TEXT,
            quiz_date      TEXT,
            dominant       TEXT,
            focused_pct    INTEGER DEFAULT 0,
            fatigued_pct   INTEGER DEFAULT 0,
            anxious_pct    INTEGER DEFAULT 0,
            overloaded_pct INTEGER DEFAULT 0,
            loading_pct    INTEGER DEFAULT 0,
            total_q        INTEGER DEFAULT 0,
            topic          TEXT
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role       TEXT,
            content    TEXT,
            timestamp  TEXT
        );
        CREATE TABLE IF NOT EXISTS password_resets (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT NOT NULL,
            token      TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used       INTEGER DEFAULT 0
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

# Per-session cognitive log {sid: [{emotion, topic, concept, correct, qIdx}]}
cog_log = {}

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
    global _groq_client
    if not _groq_client:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(3):
        try:
            r = _groq_client.chat.completions.create(
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
        global _groq_client
        if not _groq_client:
            _groq_client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Generate {num_q} {difficulty}-level MCQ questions ONLY from this material. Do not use outside knowledge.
Material: \"\"\"{text[:4000]}\"\"\"
Rules: answer must exactly match one option text. Return ONLY raw JSON:
{{"topic":"infer the topic","difficulty":"{difficulty}","questions":[{{"id":1,"type":"mcq","question":"...","options":["opt1","opt2","opt3","opt4"],"answer":"opt1","explanation":"...","concept":"..."}}]}}"""
        r = _groq_client.chat.completions.create(model="llama-3.3-70b-versatile",
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
        global _groq_client
        if not _groq_client:
            _groq_client = Groq(api_key=GROQ_API_KEY)
        context = "Correct!" if was_correct else f"Incorrect (answered: '{student_ans}')"
        r = _groq_client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":f"Topic:{topic}\nQ:{question}\nAnswer:{correct_ans}\n{context}\nExplanation:{explanation}\nWrite ONE follow-up question max 20 words. Return ONLY the question."}],
            temperature=0.8, max_tokens=60)
        return r.choices[0].message.content.strip()
    except: return ""


def ai_topic_guide(topic, concept, question, correct_answer):
    """AI-generated short study guide when student gets a question wrong."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY: return ""
    try:
        global _groq_client
        if not _groq_client:
            _groq_client = Groq(api_key=GROQ_API_KEY)
        r = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":
                f"A student got this question WRONG:\n"
                f"Subject: {topic}\n"
                f"Concept: {concept or topic}\n"
                f"Question: {question}\n"
                f"Correct Answer: {correct_answer}\n\n"
                f"Write a SHORT focused study guide with EXACTLY 4 bullet points to help them understand this specific concept.\n"
                f"Format each bullet as: • [point]\n"
                f"Be specific to the concept. No intro text. Just the 4 bullets."}],
            temperature=0.4, max_tokens=220)
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

# ════════════════════════════════════════════════════════════════
#  AUTH LAYER  — sits in front of all existing routes
#  Uses werkzeug.security for bcrypt + simple HMAC JWT (no PyJWT dep)
# ════════════════════════════════════════════════════════════════

from werkzeug.security import generate_password_hash, check_password_hash

JWT_SECRET  = os.environ.get("JWT_SECRET", "em360_dev_secret_change_in_prod")
JWT_EXPIRY  = 60 * 60 * 24 * 30   # 30 days

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64d(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (pad % 4))

def create_token(user_id: int, username: str) -> str:
    payload = json.dumps({"uid": user_id, "usr": username,
                          "exp": int(time.time()) + JWT_EXPIRY})
    header  = _b64(b'{"alg":"HS256","typ":"JWT"}')
    body    = _b64(payload.encode())
    sig     = _b64(hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(),
                            hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"

def verify_token(token: str):
    """Returns (user_id, username) or (None, None)."""
    try:
        parts = token.split(".")
        if len(parts) != 3: return None, None
        header, body, sig = parts
        expected = _b64(hmac.new(JWT_SECRET.encode(),
                                  f"{header}.{body}".encode(),
                                  hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected): return None, None
        data = json.loads(_b64d(body))
        if data.get("exp", 0) < time.time():      return None, None
        return data["uid"], data["usr"]
    except Exception:
        return None, None

def get_token_from_request():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "): return auth[7:]
    return request.cookies.get("em360_token", "")

def require_auth(f):
    """Decorator: injects current_user_id and current_username into kwargs."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        uid, usr = verify_token(get_token_from_request())
        if uid is None:
            return jsonify({"error": "Unauthorized", "auth": True}), 401
        kwargs["current_user_id"] = uid
        kwargs["current_username"] = usr
        return f(*args, **kwargs)
    return wrapper

def user_sid(user_id: int) -> str:
    """Converts a user_id to a session_id string.
       All existing routes use session_id TEXT — this bridges the gap
       without changing any existing route body."""
    return f"u_{user_id}"

def init_users_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    DEFAULT (datetime('now'))
            )
        """)
        db.commit()

init_users_db()

# ── Auth routes ──────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    d        = request.json or {}
    username = str(d.get("username", "")).strip()[:30]
    email    = str(d.get("email",    "")).strip()[:80].lower()
    password = str(d.get("password", "")).strip()

    if not username or not email or not password:
        return jsonify({"error": "All fields required"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    pw_hash = generate_password_hash(password, method="pbkdf2:sha256")
    try:
        with sqlite3.connect(DB_PATH) as db:
            db.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
                (username, email, pw_hash))
            db.commit()
            user_id = db.execute(
                "SELECT id FROM users WHERE email=?", (email,)).fetchone()[0]
    except Exception:
        return jsonify({"error": "Username or email already exists"}), 409

    token = create_token(user_id, username)
    # Pre-create stats row so dashboard loads immediately
    load_stats(user_sid(user_id))
    return jsonify({"token": token, "username": username,
                    "session_id": user_sid(user_id), "user_id": user_id}), 201

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    d        = request.json or {}
    email    = str(d.get("email",    "")).strip().lower()
    password = str(d.get("password", "")).strip()

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT id, username, password_hash FROM users WHERE email=?",
            (email,)).fetchone()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_token(row["id"], row["username"])
    return jsonify({"token": token, "username": row["username"],
                    "session_id": user_sid(row["id"]), "user_id": row["id"]})

@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    uid, usr = verify_token(get_token_from_request())
    if uid is None:
        return jsonify({"error": "Unauthorized", "auth": True}), 401
    return jsonify({"user_id": uid, "username": usr,
                    "session_id": user_sid(uid)})

@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    # Stateless JWT — logout is handled client-side by deleting the token
    return jsonify({"ok": True})
@app.route("/api/auth/forgot-password", methods=["POST"])
def auth_forgot_password():
    import secrets as _sec
    from datetime import timedelta as _td
    d = request.json or {}
    email = str(d.get("email","")).strip().lower()
    if not email or "@" not in email:
        return jsonify({"error":"Valid email required"}), 400
    with sqlite3.connect(DB_PATH) as db:
        row = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        return jsonify({"error":"No account found with that email"}), 404
    with sqlite3.connect(DB_PATH) as db:
        db.execute("UPDATE password_resets SET used=1 WHERE email=? AND used=0", (email,))
        db.commit()
    token = str(_sec.randbelow(900000) + 100000)
    expires_at = (datetime.now() + _td(minutes=15)).isoformat()
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT INTO password_resets (email,token,expires_at,used) VALUES (?,?,?,0)",
            (email, token, expires_at))
        db.commit()
    return jsonify({"ok":True,"token":token,"message":"Reset code generated — valid for 15 minutes"})


@app.route("/api/auth/reset-password", methods=["POST"])
def auth_reset_password():
    from datetime import datetime as _dt
    d = request.json or {}
    email   = str(d.get("email",           "")).strip().lower()
    token   = str(d.get("token",           "")).strip()
    new_pw  = str(d.get("new_password",    "")).strip()
    confirm = str(d.get("confirm_password","")).strip()
    if not email or not token or not new_pw or not confirm:
        return jsonify({"error":"All fields are required"}), 400
    if new_pw != confirm:
        return jsonify({"error":"Passwords do not match"}), 400
    if len(new_pw) < 6:
        return jsonify({"error":"Password must be at least 6 characters"}), 400
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT id,expires_at FROM password_resets "
            "WHERE email=? AND token=? AND used=0 ORDER BY id DESC LIMIT 1",
            (email, token)).fetchone()
    if not row:
        return jsonify({"error":"Invalid or expired reset code"}), 401
    if _dt.fromisoformat(row["expires_at"]) < _dt.now():
        return jsonify({"error":"Reset code has expired — request a new one"}), 401
    with sqlite3.connect(DB_PATH) as db:
        db.execute("UPDATE password_resets SET used=1 WHERE email=? AND token=?", (email,token))
        db.commit()
    new_hash = generate_password_hash(new_pw, method="pbkdf2:sha256")
    with sqlite3.connect(DB_PATH) as db:
        db.execute("UPDATE users SET password_hash=? WHERE email=?", (new_hash,email))
        db.commit()
    return jsonify({"ok":True,"message":"Password updated — you can now sign in"})


@app.route("/api/auth/change-password", methods=["POST"])
@require_auth
def auth_change_password(**kwargs):
    uid = kwargs["current_user_id"]
    d   = request.json or {}
    current_pw = str(d.get("current_password", "")).strip()
    new_pw     = str(d.get("new_password",     "")).strip()
    confirm_pw = str(d.get("confirm_password", "")).strip()

    if not current_pw or not new_pw or not confirm_pw:
        return jsonify({"error": "All fields are required"}), 400
    if new_pw != confirm_pw:
        return jsonify({"error": "New passwords do not match"}), 400
    if len(new_pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if new_pw == current_pw:
        return jsonify({"error": "New password must differ from current password"}), 400

    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT password_hash FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        return jsonify({"error": "User not found"}), 404
    if not check_password_hash(row["password_hash"], current_pw):
        return jsonify({"error": "Current password is incorrect"}), 401

    new_hash = generate_password_hash(new_pw, method="pbkdf2:sha256")
    with sqlite3.connect(DB_PATH) as db:
        db.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, uid))
        db.commit()
    return jsonify({"ok": True, "message": "Password updated successfully"})




HTML_PAGE = '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8">\n  <meta name="viewport" content="width=device-width,initial-scale=1">\n  <title>EduMate 360 — AI Cognitive Learning</title>\n  \n  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n  <style>\n\n@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Manrope:wght@500;600;700;800;900&display=swap\');\n\n/* ─────────────────────────────────────────────\n   DESIGN TOKENS\n   ───────────────────────────────────────────── */\n*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\n:root {\n  /* Surfaces */\n  --bg:        #050510;\n  --bg-1:      #0a0a1e;\n  --bg-2:      #0f0f28;\n  --glass:     rgba(255,255,255,0.042);\n  --glass-2:   rgba(255,255,255,0.065);\n  --glass-hover: rgba(255,255,255,0.08);\n  /* Borders */\n  --bdr:       rgba(255,255,255,0.08);\n  --bdr-glow:  rgba(99,102,241,0.45);\n  /* Text */\n  --tx-0:      #f0f2ff;\n  --tx-1:      rgba(210,215,255,0.70);\n  --tx-2:      rgba(190,198,255,0.42);\n  /* Brand */\n  --indigo:    #6366f1;\n  --indigo-l:  #818cf8;\n  --violet:    #a78bfa;\n  --mint:      #34d399;\n  --amber:     #fbbf24;\n  --rose:      #fb7185;\n  --sky:       #38bdf8;\n  /* Gradients */\n  --g-brand:   linear-gradient(135deg,#6366f1 0%,#a78bfa 55%,#34d399 100%);\n  --g-indigo:  linear-gradient(135deg,#6366f1,#818cf8);\n  --g-card:    linear-gradient(145deg,rgba(255,255,255,0.052),rgba(255,255,255,0.018));\n  /* Glow */\n  --glow-btn:  0 4px 24px rgba(99,102,241,0.50);\n  --glow-card: 0 0 28px rgba(99,102,241,0.12);\n  --shadow:    0 8px 32px rgba(0,0,0,0.55);\n  /* Radii */\n  --r:    14px;\n  --r-lg: 20px;\n  --r-xl: 26px;\n  --r-p:  999px;\n  /* Layout */\n  --sidebar-w: 64px;\n}\n\n/* ─────────────────────────────────────────────\n   BASE\n   ───────────────────────────────────────────── */\nhtml { scroll-behavior: smooth; }\nbody {\n  font-family: \'Inter\', sans-serif;\n  background: var(--bg);\n  color: var(--tx-0);\n  min-height: 100vh;\n  overflow-x: hidden;\n  -webkit-font-smoothing: antialiased;\n  -moz-osx-font-smoothing: grayscale;\n  font-feature-settings: \'cv11\',\'ss01\';\n  text-rendering: optimizeLegibility;\n  line-height: 1.6;\n  font-size: 15px;\n}\n\n/* Deep gradient background with radial hero highlight */\nbody::before {\n  content: \'\';\n  position: fixed; inset: 0; z-index: 0; pointer-events: none;\n  background:\n    radial-gradient(ellipse 70% 55% at 50% 0%, rgba(99,102,241,0.13) 0%, transparent 65%),\n    radial-gradient(ellipse 50% 40% at 85% 80%, rgba(167,139,250,0.07) 0%, transparent 55%),\n    linear-gradient(180deg, var(--bg-1) 0%, var(--bg) 40%, #030308 100%);\n}\n\n/* Dot grid texture */\nbody::after {\n  content: \'\';\n  position: fixed; inset: 0; z-index: 0; pointer-events: none;\n  background-image: radial-gradient(rgba(99,102,241,0.055) 1px, transparent 1px);\n  background-size: 28px 28px;\n  mask-image: radial-gradient(ellipse 90% 70% at 50% 25%, black 30%, transparent 100%);\n}\n\n#app { display: flex; min-height: 100vh; position: relative; z-index: 1; }\n\n::-webkit-scrollbar { width: 3px; }\n::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.28); border-radius: 99px; }\n::-webkit-scrollbar-track { background: transparent; }\n::selection { background: rgba(99,102,241,0.30); color: #fff; }\n\n/* ─────────────────────────────────────────────\n   TYPOGRAPHY\n   ───────────────────────────────────────────── */\nh1, h2, h3, h4 {\n  font-family: \'Manrope\', sans-serif;\n  letter-spacing: -0.03em;\n  line-height: 1.18;\n  font-weight: 700;\n}\n.grad {\n  background: var(--g-brand);\n  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;\n}\n\n/* ─────────────────────────────────────────────\n   GLASS CARD  (reusable base)\n   ───────────────────────────────────────────── */\n.glass-card {\n  background: var(--glass);\n  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);\n  border: 1px solid var(--bdr);\n  border-radius: var(--r-lg);\n  box-shadow: var(--shadow);\n  transition: border-color .22s, transform .22s, box-shadow .22s;\n}\n.glass-card:hover {\n  border-color: rgba(99,102,241,0.28);\n  box-shadow: var(--glow-card), var(--shadow);\n  transform: translateY(-3px);\n}\n\n/* ─────────────────────────────────────────────\n   SIDEBAR\n   ───────────────────────────────────────────── */\n.sidebar {\n  width: var(--sidebar-w);\n  background: rgba(5,5,16,0.90);\n  backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);\n  border-right: 1px solid var(--bdr);\n  display: flex; flex-direction: column;\n  align-items: center; padding: 14px 0 16px;\n  position: fixed; top: 0; left: 0; bottom: 0;\n  z-index: 200; overflow: hidden;\n  gap: 4px;\n  box-shadow: 2px 0 30px rgba(0,0,0,0.5);\n}\n.logo {\n  width: 36px; height: 36px;\n  background: var(--g-indigo);\n  border-radius: 11px;\n  display: flex; align-items: center; justify-content: center;\n  font-size: 1.08rem; margin-bottom: 14px; flex-shrink: 0;\n  box-shadow: var(--glow-btn); cursor: default;\n  transition: transform .2s;\n}\n.logo:hover { transform: rotate(-8deg) scale(1.1); }\n.nav-btn {\n  width: 40px; height: 40px;\n  background: transparent;\n  border: 1px solid transparent;\n  border-radius: 11px;\n  cursor: pointer; font-size: 1rem;\n  color: var(--tx-2);\n  transition: all .18s cubic-bezier(.4,0,.2,1);\n  flex-shrink: 0; position: relative;\n  display: flex; align-items: center; justify-content: center;\n}\n.nav-btn::after {\n  content: attr(title);\n  position: absolute; left: calc(100% + 10px); top: 50%;\n  transform: translateY(-50%) scale(.85);\n  background: rgba(5,5,16,0.97);\n  border: 1px solid rgba(99,102,241,0.35);\n  color: var(--tx-0); font-size: .67rem; font-weight: 600;\n  padding: 4px 10px; border-radius: 8px;\n  white-space: nowrap; pointer-events: none;\n  opacity: 0; transition: all .14s; z-index: 300;\n  font-family: \'Inter\', sans-serif;\n  box-shadow: var(--shadow);\n}\n.nav-btn:hover::after { opacity: 1; transform: translateY(-50%) scale(1); }\n.nav-btn:hover {\n  background: rgba(99,102,241,0.11);\n  border-color: rgba(99,102,241,0.26);\n  color: var(--indigo-l);\n  transform: scale(1.07);\n}\n.nav-btn.active {\n  background: rgba(99,102,241,0.16);\n  border-color: rgba(99,102,241,0.42);\n  color: var(--indigo-l);\n  box-shadow: 0 0 18px rgba(99,102,241,0.18);\n}\n.level-pill {\n  margin-top: auto; flex-shrink: 0;\n  width: 36px; height: 36px;\n  background: linear-gradient(135deg, var(--amber), #d97706);\n  border-radius: 10px;\n  display: flex; flex-direction: column;\n  align-items: center; justify-content: center;\n  box-shadow: 0 3px 14px rgba(251,191,36,0.30); cursor: default;\n}\n.level-pill .lv { font-size: .44rem; font-weight: 800; color: rgba(0,0,0,.58); letter-spacing: .8px; }\n#lv-num { font-size: .9rem; font-weight: 900; color: #1c1912; line-height: 1; }\n\n/* ─────────────────────────────────────────────\n   MAIN LAYOUT\n   ───────────────────────────────────────────── */\n.main {\n  margin-left: var(--sidebar-w);\n  flex: 1;\n  padding: 32px 28px 80px;\n  max-width: 1040px;\n  width: calc(100% - var(--sidebar-w));\n}\n\n/* ─────────────────────────────────────────────\n   SCREENS\n   ───────────────────────────────────────────── */\n.screen { display: none; }\n.screen.active {\n  display: block;\n  animation: fadeUp .30s cubic-bezier(.4,0,.2,1);\n}\n@keyframes fadeUp {\n  from { opacity: 0; transform: translateY(14px); }\n  to   { opacity: 1; transform: translateY(0); }\n}\n\n/* Page header */\n.page-head { margin-bottom: 28px; }\n.page-head h1 { font-size: 1.85rem; font-weight: 800; margin: 8px 0 6px; letter-spacing: -0.03em; }\n.page-head .sub { font-size: .9rem; color: var(--tx-1); line-height: 1.65; }\n\n/* ─────────────────────────────────────────────\n   TAG / BADGE TOKENS\n   ───────────────────────────────────────────── */\n.tag {\n  display: inline-flex; align-items: center; gap: 4px;\n  font-size: .65rem; font-weight: 700;\n  letter-spacing: .9px; text-transform: uppercase;\n  padding: 3px 10px; border-radius: var(--r-p);\n  background: rgba(99,102,241,0.10);\n  border: 1px solid rgba(99,102,241,0.24);\n  color: var(--indigo-l);\n}\n.tag.green  { background:rgba(52,211,153,0.09); border-color:rgba(52,211,153,0.25); color:var(--mint); }\n.tag.indigo { background:rgba(99,102,241,0.10); border-color:rgba(99,102,241,0.25); color:var(--indigo-l); }\n.tag.violet { background:rgba(167,139,250,0.10); border-color:rgba(167,139,250,0.26); color:var(--violet); }\n.tag.teal   { background:rgba(56,189,248,0.09);  border-color:rgba(56,189,248,0.25); color:var(--sky); }\n.tag.gold,.tag.amber { background:rgba(251,191,36,0.09); border-color:rgba(251,191,36,0.25); color:var(--amber); }\n.tag.orange { background:rgba(251,146,60,0.09); border-color:rgba(251,146,60,0.24); color:#fb923c; }\n.tag.red    { background:rgba(251,113,133,0.09); border-color:rgba(251,113,133,0.22); color:var(--rose); }\n.badge-pill,.badge {\n  display: inline-block;\n  background: rgba(251,191,36,0.09); border: 1px solid rgba(251,191,36,0.25);\n  color: var(--amber); border-radius: var(--r-p);\n  padding: 3px 11px; font-size: .72rem; font-weight: 700; margin: 2px;\n}\n\n/* ─────────────────────────────────────────────\n   BUTTONS\n   ───────────────────────────────────────────── */\n.btn-primary {\n  display: inline-flex; align-items: center; gap: 7px;\n  background: var(--g-indigo);\n  border: none; color: #fff;\n  padding: 11px 24px; border-radius: 12px;\n  font-size: .88rem; font-weight: 700;\n  font-family: \'Inter\', sans-serif;\n  cursor: pointer; letter-spacing: .1px;\n  transition: all .22s cubic-bezier(.4,0,.2,1);\n  box-shadow: var(--glow-btn);\n  position: relative; overflow: hidden;\n}\n.btn-primary::after {\n  content: \'\';\n  position: absolute; inset: 0;\n  background: linear-gradient(135deg,rgba(255,255,255,0.18),transparent);\n  opacity: 0; transition: opacity .2s;\n}\n.btn-primary:hover { transform: translateY(-3px); box-shadow: 0 10px 36px rgba(99,102,241,0.55); }\n.btn-primary:hover::after { opacity: 1; }\n.btn-primary:active { transform: translateY(0); }\n\n.btn-ghost {\n  display: inline-flex; align-items: center; gap: 7px;\n  background: var(--glass);\n  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);\n  border: 1px solid var(--bdr);\n  color: var(--tx-1);\n  padding: 11px 24px; border-radius: 12px;\n  font-size: .88rem; font-weight: 600;\n  font-family: \'Inter\', sans-serif;\n  cursor: pointer;\n  transition: all .2s;\n}\n.btn-ghost:hover {\n  background: rgba(99,102,241,0.10);\n  border-color: rgba(99,102,241,0.32);\n  color: var(--indigo-l);\n  transform: translateY(-2px);\n}\n\n/* Action buttons (Voice, Cam, Back) */\n.abtn {\n  display: inline-flex; align-items: center; gap: 5px;\n  background: var(--glass);\n  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);\n  border: 1px solid var(--bdr);\n  color: rgba(200,215,255,0.75);\n  border-radius: 10px; padding: 7px 15px;\n  font-size: .79rem; font-weight: 600;\n  font-family: \'Inter\', sans-serif;\n  cursor: pointer; transition: all .18s;\n  white-space: nowrap;\n}\n.abtn:hover {\n  background: rgba(99,102,241,0.10);\n  border-color: rgba(99,102,241,0.30);\n  color: var(--indigo-l);\n  box-shadow: 0 0 12px rgba(99,102,241,0.14);\n}\n.abtn.active {\n  background: rgba(52,211,153,0.12);\n  border-color: rgba(52,211,153,0.38);\n  color: var(--mint);\n}\n\n/* Generate / upload buttons */\n.gen-btn, .upload-btn, .submit-btn, .submit-short-btn, .lb-submit {\n  background: var(--g-indigo);\n  border: none; color: #fff;\n  padding: 10px 20px; border-radius: 11px;\n  font-size: .86rem; font-weight: 700;\n  font-family: \'Inter\', sans-serif;\n  cursor: pointer;\n  box-shadow: 0 3px 18px rgba(99,102,241,0.36);\n  transition: all .2s; align-self: flex-end;\n}\n.gen-btn:hover,.upload-btn:hover,.submit-btn:hover,.submit-short-btn:hover,.lb-submit:hover {\n  transform: translateY(-2px); box-shadow: 0 7px 28px rgba(99,102,241,0.50);\n}\n.gen-btn:disabled,.upload-btn:disabled { opacity: .44; cursor: not-allowed; transform: none; box-shadow: none; }\n\n.fb-btn,.next-btn {\n  background: var(--g-indigo); border: none; color: #fff;\n  padding: 9px 20px; border-radius: 10px; font-size: .83rem; font-weight: 700;\n  font-family: \'Inter\', sans-serif; cursor: pointer;\n  box-shadow: 0 3px 14px rgba(99,102,241,0.26); margin-top: 6px;\n  transition: all .2s;\n}\n.fb-btn:hover,.next-btn:hover { transform: translateY(-2px); }\n\n.end-btn,.report-btn {\n  background: var(--glass); backdrop-filter: blur(8px);\n  border: 1px solid var(--bdr); color: var(--tx-1);\n  border-radius: 9px; padding: 6px 13px; font-size: .74rem; font-weight: 600;\n  font-family: \'Inter\', sans-serif; cursor: pointer; transition: all .18s;\n}\n.end-btn:hover    { border-color: rgba(251,113,133,.35); color: var(--rose); }\n.report-btn:hover { border-color: rgba(99,102,241,.38); color: var(--indigo-l); }\n\n.review-btn {\n  background: rgba(99,102,241,0.10); border: 1px solid rgba(99,102,241,0.24);\n  color: var(--indigo-l); border-radius: 8px; padding: 5px 12px;\n  font-size: .73rem; font-weight: 700; font-family: \'Inter\', sans-serif;\n  cursor: pointer; transition: all .18s;\n}\n.review-btn:hover  { background: rgba(99,102,241,0.18); }\n.review-btn.urgent { background: rgba(251,191,36,0.10); border-color: rgba(251,191,36,0.32); color: var(--amber); }\n\n.export-btn {\n  background: var(--glass); border: 1px solid var(--bdr); color: var(--tx-1);\n  border-radius: 9px; padding: 7px 14px; font-size: .75rem; font-weight: 600;\n  cursor: pointer; transition: all .18s; font-family: \'Inter\', sans-serif;\n}\n.export-btn:hover { border-color: rgba(99,102,241,0.32); color: var(--indigo-l); }\n\n.tab-btn,.upload-tab-btn {\n  background: var(--glass); border: 1px solid var(--bdr); color: var(--tx-1);\n  border-radius: 10px; padding: 8px 16px; font-size: .80rem; font-weight: 600;\n  cursor: pointer; transition: all .18s; font-family: \'Inter\', sans-serif;\n}\n.tab-btn.active,.upload-tab-btn.active {\n  background: rgba(99,102,241,0.12); border-color: rgba(99,102,241,0.38); color: var(--indigo-l);\n}\n.tab-btn:hover:not(.active) { border-color: rgba(99,102,241,0.20); }\n\n.due-btn {\n  background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.28);\n  color: var(--amber); border-radius: 8px; padding: 4px 12px;\n  font-size: .76rem; font-weight: 700; cursor: pointer;\n  font-family: \'Inter\', sans-serif; margin-left: 8px; transition: background .18s;\n}\n.due-btn:hover { background: rgba(251,191,36,0.20); }\n\n/* CR close */\n.cr-close-btn {\n  width: 100%; background: var(--g-indigo); border: none; color: #fff;\n  padding: 12px; border-radius: 12px; font-size: .87rem; font-weight: 700;\n  font-family: \'Inter\', sans-serif; cursor: pointer;\n  box-shadow: var(--glow-btn); transition: all .2s;\n}\n.cr-close-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(99,102,241,0.50); }\n\n/* Mistake actions */\n.mn-retry-btn {\n  background: rgba(167,139,250,0.09); border: 1px solid rgba(167,139,250,0.20);\n  color: var(--violet); border-radius: 9px; padding: 5px 12px; font-size: .73rem;\n  font-weight: 700; cursor: pointer; font-family: \'Inter\', sans-serif; transition: all .16s;\n}\n.mn-retry-btn:hover:not(:disabled) { background: rgba(167,139,250,0.16); }\n.mn-retry-btn:disabled { opacity: .42; cursor: not-allowed; }\n.mn-ask-btn {\n  background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.19);\n  color: var(--mint); border-radius: 9px; padding: 5px 12px; font-size: .73rem;\n  font-weight: 700; cursor: pointer; font-family: \'Inter\', sans-serif; transition: all .16s;\n}\n.mn-ask-btn:hover { background: rgba(52,211,153,0.13); }\n.mn-clear-btn {\n  background: rgba(251,113,133,0.08); border: 1px solid rgba(251,113,133,0.18);\n  color: var(--rose); border-radius: 10px; padding: 7px 14px; cursor: pointer;\n  font-size: .78rem; font-weight: 700; font-family: \'Inter\', sans-serif; transition: all .18s;\n}\n.mn-clear-btn:hover { background: rgba(251,113,133,0.15); }\n\n/* Chat buttons */\n.chat-send-btn {\n  background: var(--g-indigo); border: none; color: #fff; border-radius: 12px;\n  padding: 11px 18px; cursor: pointer; font-weight: 700;\n  font-family: \'Inter\', sans-serif; font-size: .85rem;\n  box-shadow: var(--glow-btn); white-space: nowrap; transition: all .2s;\n}\n.chat-send-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(99,102,241,0.50); }\n.chat-send-btn:disabled { opacity: .42; cursor: not-allowed; transform: none; box-shadow: none; }\n.chat-clear-btn {\n  background: var(--glass); border: 1px solid var(--bdr); color: var(--tx-1);\n  border-radius: 10px; padding: 8px 13px; font-size: .74rem; font-weight: 600;\n  cursor: pointer; font-family: \'Inter\', sans-serif; transition: all .16s;\n}\n.chat-clear-btn:hover { border-color: rgba(99,102,241,0.32); color: var(--indigo-l); }\n.chat-sug {\n  background: var(--glass); border: 1px solid var(--bdr);\n  color: rgba(190,198,255,.5); border-radius: var(--r-p); padding: 4px 12px;\n  font-size: .7rem; cursor: pointer; font-family: \'Inter\', sans-serif; transition: all .16s;\n}\n.chat-sug:hover { background: rgba(99,102,241,0.10); border-color: rgba(99,102,241,0.32); color: var(--indigo-l); }\n\n/* Breathing skip */\n.breathing-skip {\n  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.18);\n  color: #fff; border-radius: 10px; padding: 8px 22px;\n  cursor: pointer; font-size: .82rem; font-weight: 600;\n  font-family: \'Inter\', sans-serif; transition: background .18s;\n}\n.breathing-skip:hover { background: rgba(255,255,255,.14); }\n\n/* ─────────────────────────────────────────────\n   FORM INPUTS\n   ───────────────────────────────────────────── */\n.topic-input {\n  flex: 1; min-width: 200px;\n  background: var(--glass);\n  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);\n  border: 1px solid var(--bdr); color: var(--tx-0); border-radius: 12px;\n  padding: 12px 15px; font-size: .9rem; font-family: \'Inter\', sans-serif;\n  outline: none; transition: all .2s;\n}\n.topic-input:focus {\n  border-color: rgba(99,102,241,0.50);\n  background: rgba(99,102,241,0.06);\n  box-shadow: 0 0 0 3px rgba(99,102,241,0.10);\n}\n.topic-input::placeholder { color: var(--tx-2); }\n\n.sel, .lb-input, .short-input, .chat-topic-inp, .chat-input {\n  background: var(--glass);\n  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);\n  border: 1px solid var(--bdr); color: var(--tx-0);\n  font-family: \'Inter\', sans-serif; outline: none; transition: all .2s;\n}\n.sel { border-radius: 10px; padding: 9px 28px 9px 11px; font-size: .83rem; cursor: pointer; appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'10\' height=\'6\' fill=\'none\'%3E%3Cpath d=\'M1 1l4 4 4-4\' stroke=\'rgba(190,198,255,0.4)\' stroke-width=\'1.5\' stroke-linecap=\'round\'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 10px center; }\n.sel option { background: #0a0a1e; color: var(--tx-0); }\n.sel:focus,.lb-input:focus,.short-input:focus,.chat-topic-inp:focus,.chat-input:focus { border-color: rgba(99,102,241,0.50); box-shadow: 0 0 0 3px rgba(99,102,241,0.09); }\n.lb-input { border-radius: 10px; padding: 10px 14px; font-size: .86rem; flex: 1; min-width: 140px; }\n.short-input,.notes-input,.text-area { width: 100%; }\n.short-input { border-radius: 12px; padding: 12px 15px; font-size: .9rem; }\n.notes-input,.text-area { min-height: 125px; background: var(--glass); border: 1px solid var(--bdr); color: var(--tx-0); border-radius: 12px; padding: 12px 15px; font-size: .86rem; font-family: \'Inter\', sans-serif; outline: none; resize: vertical; transition: border-color .2s; line-height: 1.6; }\n.notes-input:focus,.text-area:focus { border-color: rgba(99,102,241,0.50); }\n.notes-input::placeholder { color: var(--tx-2); }\n.chat-topic-inp { flex: 1; min-width: 150px; border-radius: 10px; padding: 8px 13px; font-size: .82rem; }\n.chat-input { flex: 1; border-radius: 12px; padding: 11px 15px; font-size: .87rem; }\n\n/* ─────────────────────────────────────────────\n   HERO\n   ───────────────────────────────────────────── */\n.hero {\n  position: relative;\n  border-radius: var(--r-xl);\n  padding: 52px 46px 46px;\n  margin-bottom: 28px;\n  overflow: hidden;\n  background: linear-gradient(135deg,\n    rgba(99,102,241,0.07) 0%,\n    rgba(167,139,250,0.05) 50%,\n    rgba(52,211,153,0.04) 100%);\n  border: 1px solid rgba(99,102,241,0.14);\n  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);\n}\n.hero::before {\n  content: \'\'; position: absolute; top: 0; left: 0; right: 0; height: 1px;\n  background: linear-gradient(90deg,transparent,var(--indigo),var(--violet),transparent);\n  animation: shimmer 4s ease-in-out infinite;\n}\n@keyframes shimmer { 0%,100%{opacity:.30} 50%{opacity:1} }\n.hero::after {\n  content: \'\'; position: absolute; top: -30%; right: -8%; width: 50%; height: 130%;\n  background: radial-gradient(ellipse,rgba(99,102,241,0.09) 0%,transparent 65%);\n  pointer-events: none;\n}\n.hero-inner {\n  position: relative; z-index: 1;\n  display: flex; justify-content: space-between; align-items: flex-start;\n  gap: 28px; flex-wrap: wrap;\n}\n.hero-text { flex: 1; min-width: 260px; }\n.hero-eyebrow {\n  display: inline-flex; align-items: center; gap: 6px;\n  font-size: .66rem; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase;\n  color: var(--indigo-l); margin-bottom: 16px; padding: 5px 13px;\n  background: rgba(99,102,241,0.09); border: 1px solid rgba(99,102,241,0.24);\n  border-radius: var(--r-p);\n}\n.hero-eyebrow-dot {\n  width: 5px; height: 5px; border-radius: 50%; background: var(--indigo);\n  animation: pulse 2.2s ease infinite;\n}\n@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,0.6)} 50%{box-shadow:0 0 0 5px rgba(99,102,241,0)} }\n.hero-text h1 { font-size: 2.35rem; font-weight: 800; line-height: 1.12; margin-bottom: 16px; letter-spacing: -0.03em; }\n.hero-tagline { font-size: .9rem; color: var(--tx-1); line-height: 1.75; max-width: 480px; margin-bottom: 26px; }\n.hero-actions { display: flex; gap: 10px; flex-wrap: wrap; }\n\n/* XP ring (hero right) */\n.xp-ring {\n  flex-shrink: 0;\n  background: var(--glass);\n  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);\n  border: 1px solid var(--bdr);\n  border-radius: var(--r-lg); padding: 20px 24px;\n  text-align: center; min-width: 130px; position: relative; z-index: 1;\n}\n.xp-num { font-family: \'Manrope\', sans-serif; font-size: 2.15rem; font-weight: 800; background: var(--g-brand); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }\n.xp-lbl { font-size: .57rem; font-weight: 700; color: var(--tx-2); letter-spacing: 1.6px; text-transform: uppercase; margin: 3px 0 11px; }\n.xp-bar-wrap { height: 3px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; margin-bottom: 7px; }\n.xp-bar { height: 100%; width: 0%; background: var(--g-brand); border-radius: 99px; transition: width .75s cubic-bezier(.4,0,.2,1); }\n.xp-to { font-size: .62rem; color: var(--tx-2); }\n\n/* Due alert */\n.due-alert {\n  background: linear-gradient(135deg,rgba(251,191,36,0.09),rgba(251,113,133,0.04));\n  border: 1px solid rgba(251,191,36,0.22); border-radius: var(--r);\n  padding: 12px 16px; margin-bottom: 18px;\n  font-size: .82rem; color: var(--amber); font-weight: 500;\n  display: flex; align-items: center; gap: 8px;\n}\n.due-alert.hidden { display: none!important; }\n\n/* Stats row */\n.stats-row { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin-bottom: 22px; }\n.sc {\n  background: var(--glass);\n  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);\n  border: 1px solid var(--bdr); border-radius: var(--r);\n  padding: 14px 10px; text-align: center; transition: all .22s;\n}\n.sc:hover { border-color: rgba(99,102,241,0.30); transform: translateY(-3px); box-shadow: var(--glow-card); }\n.si { display: block; font-size: 1.15rem; margin-bottom: 4px; }\n.sv { display: block; font-family: \'Manrope\', sans-serif; font-size: 1.28rem; font-weight: 800; background: var(--g-brand); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }\n.sl { display: block; font-size: .57rem; color: var(--tx-2); font-weight: 700; text-transform: uppercase; letter-spacing: .8px; margin-top: 2px; }\n\n/* Badge row */\n.badge-row { margin-bottom: 18px; }\n.badge-list { display: flex; flex-wrap: wrap; gap: 6px; }\n\n/* Home section title */\n.home-section-title {\n  font-size: .64rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.4px;\n  color: var(--tx-2); margin: 28px 0 13px; display: flex; align-items: center; gap: 7px;\n}\n\n/* Gen card */\n.gen-card {\n  background: var(--glass);\n  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);\n  border: 1px solid var(--bdr); border-radius: var(--r-xl);\n  padding: 24px 22px; margin-bottom: 22px; transition: border-color .2s;\n}\n.gen-card:focus-within { border-color: rgba(99,102,241,0.40); box-shadow: 0 0 0 1px rgba(99,102,241,0.12); }\n.gen-title { font-size: .98rem; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; font-family: \'Manrope\', sans-serif; }\n.gen-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 10px; }\n.gen-row:last-of-type { margin-bottom: 0; }\n.gen-controls { flex-wrap: wrap; }\n.control-group { display: flex; flex-direction: column; gap: 4px; min-width: 0; }\n.control-group label { font-size: .62rem; font-weight: 700; color: var(--tx-2); text-transform: uppercase; letter-spacing: .9px; }\n.gen-error { color: var(--rose); font-size: .79rem; padding: 8px 12px; background: rgba(251,113,133,0.07); border: 1px solid rgba(251,113,133,0.15); border-radius: 9px; margin-top: 8px; }\n.gen-error.hidden { display: none!important; }\n\n/* Feature Showcase (6 cards) */\n.feature-showcase { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 26px; }\n.fsc-card {\n  background: var(--glass);\n  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);\n  border: 1px solid var(--bdr); border-radius: var(--r-lg);\n  padding: 16px 15px; cursor: pointer;\n  display: flex; gap: 13px; align-items: flex-start;\n  transition: all .22s cubic-bezier(.4,0,.2,1);\n  position: relative; overflow: hidden;\n}\n.fsc-card::before {\n  content: \'\'; position: absolute; top: 0; left: 0; right: 0; height: 2px;\n  background: var(--g-brand); transform: scaleX(0); transform-origin: left;\n  transition: transform .25s; border-radius: var(--r-lg) var(--r-lg) 0 0;\n}\n.fsc-card:hover { border-color: rgba(99,102,241,0.32); transform: translateY(-5px); box-shadow: var(--glow-card), var(--shadow); }\n.fsc-card:hover::before { transform: scaleX(1); }\n.fsc-icon-wrap { width: 38px; height: 38px; min-width: 38px; border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; }\n.fsc-blue   { background: rgba(99,102,241,0.14); }\n.fsc-violet { background: rgba(167,139,250,0.13); }\n.fsc-teal   { background: rgba(56,189,248,0.11); }\n.fsc-amber  { background: rgba(251,191,36,0.12); }\n.fsc-red    { background: rgba(251,113,133,0.10); }\n.fsc-green  { background: rgba(52,211,153,0.11); }\n.fsc-body   { flex: 1; min-width: 0; }\n.fsc-title  { font-size: .84rem; font-weight: 700; font-family: \'Manrope\', sans-serif; margin-bottom: 4px; letter-spacing: -0.015em; }\n.fsc-desc   { font-size: .72rem; color: var(--tx-1); line-height: 1.55; margin-bottom: 7px; }\n.fsc-badge  { display: inline-block; background: rgba(99,102,241,0.09); border: 1px solid rgba(99,102,241,0.20); color: var(--indigo-l); border-radius: var(--r-p); padding: 2px 9px; font-size: .62rem; font-weight: 700; }\n\n/* Tech stack pills */\n.tech-stack-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 26px; }\n.tech-pill { display: inline-flex; align-items: center; gap: 5px; padding: 6px 13px; border-radius: var(--r-p); font-size: .73rem; font-weight: 700; border: 1px solid transparent; transition: transform .14s, box-shadow .14s; }\n.tech-pill:hover { transform: translateY(-2px); box-shadow: var(--glow-card); }\n.tech-ai    { background:rgba(167,139,250,0.10); border-color:rgba(167,139,250,0.26); color:var(--violet); }\n.tech-cv    { background:rgba(56,189,248,0.09);  border-color:rgba(56,189,248,0.25);  color:var(--sky); }\n.tech-ml    { background:rgba(99,102,241,0.10);  border-color:rgba(99,102,241,0.26);  color:var(--indigo-l); }\n.tech-chart { background:rgba(52,211,153,0.09);  border-color:rgba(52,211,153,0.25);  color:var(--mint); }\n.tech-db    { background:rgba(251,191,36,0.09);  border-color:rgba(251,191,36,0.25);  color:var(--amber); }\n.tech-algo  { background:rgba(56,189,248,0.09);  border-color:rgba(56,189,248,0.24);  color:var(--sky); }\n.tech-pdf   { background:rgba(251,113,133,0.08); border-color:rgba(251,113,133,0.22); color:var(--rose); }\n.tech-voice { background:rgba(167,139,250,0.09); border-color:rgba(167,139,250,0.24); color:var(--violet); }\n\n/* ─────────────────────────────────────────────\n   QUIZ SCREEN\n   ───────────────────────────────────────────── */\n.quiz-top { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }\n.q-topic-badge { background:rgba(99,102,241,0.10); border:1px solid rgba(99,102,241,0.24); color:var(--indigo-l); border-radius:var(--r-p); padding:4px 12px; font-size:.73rem; font-weight:700; }\n.q-diff-badge  { border-radius:var(--r-p); padding:4px 12px; font-size:.70rem; font-weight:700; }\n.diff-beginner,.diff-easy { background:rgba(52,211,153,0.09); border:1px solid rgba(52,211,153,0.24); color:var(--mint); }\n.diff-medium   { background:rgba(99,102,241,0.09); border:1px solid rgba(99,102,241,0.24); color:var(--indigo-l); }\n.diff-hard,.diff-expert { background:rgba(251,113,133,0.09); border:1px solid rgba(251,113,133,0.24); color:var(--rose); }\n.quiz-actions { margin-left: auto; display: flex; gap: 6px; }\n.qprog-wrap { height: 2px; background: rgba(255,255,255,0.05); border-radius: 99px; margin-bottom: 22px; overflow: hidden; }\n.qprog-fill { height: 100%; background: var(--g-brand); border-radius: 99px; transition: width .4s ease; }\n.quiz-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }\n\n/* Live strip */\n.live-strip {\n  display: flex;\n  background: var(--glass); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);\n  border: 1px solid var(--bdr); border-radius: 12px; overflow: hidden; margin-bottom: 18px;\n}\n.live-strip > span { flex:1; padding:7px 10px; text-align:center; border-right:1px solid var(--bdr); font-size:.78rem; color:var(--tx-1); display:inline-flex; align-items:center; justify-content:center; gap:4px; }\n.live-strip > span:last-child { border-right: none; }\n.live-strip b { color: var(--tx-0); font-weight: 800; font-family: \'Manrope\', sans-serif; }\n.ml-tag     { font-size:.73rem; font-weight:700; color:#93c5fd!important; }\n.timer-badge{ font-size:.76rem; font-weight:700; color:var(--amber)!important; font-family:\'Manrope\', sans-serif; }\n.cog-badge  { display:inline-flex; align-items:center; gap:4px; font-size:.71rem; font-weight:700; border:1px solid currentColor; border-radius:var(--r-p); padding:2px 9px; opacity:.9; }\n.cog-badge.hidden { display: none!important; }\n\n/* Camera panel — AI monitoring interface */\n.cam-panel {\n  position: fixed; bottom: 18px; right: 18px;\n  background: rgba(5,5,16,0.94);\n  backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);\n  border: 1px solid rgba(99,102,241,0.18);\n  border-radius: var(--r-lg); overflow: hidden;\n  box-shadow: var(--shadow), 0 0 30px rgba(99,102,241,0.12); z-index: 100;\n}\n.cam-panel.hidden { display: none!important; }\n.cam-panel-topbar { display:flex; align-items:center; justify-content:space-between; padding:7px 11px; background:rgba(99,102,241,0.07); border-bottom:1px solid rgba(99,102,241,0.12); cursor:grab; user-select:none; }\n.cam-panel-topbar:active { cursor: grabbing; }\n.cam-panel-title,.cam-panel-label { font-size:.6rem; font-weight:700; color:var(--indigo-l); letter-spacing:.8px; text-transform:uppercase; }\n.cam-panel-actions { display: flex; gap: 4px; }\n.cam-hide-btn,.cam-close-btn { background:rgba(255,255,255,0.04); border:none; color:var(--tx-1); cursor:pointer; border-radius:6px; padding:3px 7px; font-size:.68rem; font-weight:600; font-family:\'Inter\', sans-serif; transition:all .14s; }\n.cam-hide-btn:hover  { background:rgba(99,102,241,0.14); color:var(--indigo-l); }\n.cam-close-btn:hover { background:rgba(251,113,133,0.14); color:var(--rose); }\n.cam-body { display: flex; }\n.cam-body-hidden { padding: 8px 14px; font-size: .68rem; color: var(--tx-1); }\n.cam-left { position: relative; }\nvideo#cam-feed { display: block; width: 180px; }\n.face-status { position:absolute; bottom:6px; left:6px; right:6px; background:rgba(0,0,0,0.75); border-radius:6px; padding:3px 6px; font-size:.6rem; font-weight:600; text-align:center; backdrop-filter:blur(4px); }\n.cam-right { width:130px; padding:10px 10px 8px; border-left:1px solid rgba(255,255,255,0.05); display:flex; flex-direction:column; gap:5px; }\n.cog-state-label { font-size:.57rem; font-weight:700; color:var(--tx-2); text-transform:uppercase; letter-spacing:.8px; }\n.emo-state { font-size:.92rem; font-weight:800; font-family:\'Manrope\', sans-serif; margin:1px 0; }\n.emo-sub   { font-size:.63rem; color:var(--tx-1); margin-bottom:4px; }\n.emo-bars  { display:flex; flex-direction:column; gap:3px; }\n.emo-bar-row   { display:flex; align-items:center; gap:4px; }\n.emo-bar-label { font-size:.53rem; color:var(--tx-2); width:48px; flex-shrink:0; text-transform:capitalize; }\n.emo-bar-track { flex:1; height:3px; background:rgba(255,255,255,.06); border-radius:99px; overflow:hidden; }\n.emo-bar-fill  { height:100%; border-radius:99px; transition:width .5s ease; }\n.emo-bar-pct   { font-size:.52rem; color:var(--tx-2); width:22px; text-align:right; flex-shrink:0; }\n.emo-model-tag { font-size:.52rem; color:var(--tx-2); text-align:center; margin-top:4px; opacity:.55; }\n.cam-video-wrap { position:relative; width:160px; height:120px; border-radius:8px; overflow:hidden; background:#000; border:1px solid rgba(99,102,241,0.16); }\n.face-canvas    { position:absolute; top:0; left:0; pointer-events:none; }\n\n/* Question box — glass card */\n.q-box {\n  background: var(--glass); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);\n  border: 1px solid var(--bdr); border-radius: var(--r-xl);\n  padding: 26px 24px; margin-bottom: 14px;\n}\n.q-concept { background:rgba(167,139,250,0.09); border:1px solid rgba(167,139,250,0.20); color:var(--violet); border-radius:var(--r-p); display:inline-block; padding:3px 11px; font-size:.68rem; font-weight:700; margin-bottom:12px; }\n.q-num  { font-size:.7rem; font-weight:700; color:var(--tx-2); margin-bottom:4px; }\n.q-text { font-size:1.05rem; font-weight:600; line-height:1.65; margin-bottom:20px; font-family:\'Manrope\', sans-serif; }\n.options-grid { display: flex; flex-direction: column; gap: 9px; }\n.opt-btn {\n  background: var(--glass); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);\n  border: 1px solid var(--bdr); color: var(--tx-0); border-radius: 12px;\n  padding: 12px 16px; text-align: left; font-size: .86rem;\n  font-family: \'Inter\', sans-serif; cursor: pointer;\n  transition: all .17s; display: flex; align-items: center; gap: 10px; font-weight: 500;\n}\n.opt-btn:hover:not(:disabled) { border-color:rgba(99,102,241,0.40); background:rgba(99,102,241,0.07); transform:translateX(4px); }\n.opt-btn.selected   { border-color:rgba(99,102,241,0.45); background:rgba(99,102,241,0.08); }\n.opt-btn.correct-opt{ border-color:rgba(52,211,153,0.55)!important; background:rgba(52,211,153,0.08)!important; color:#6ee7b7!important; }\n.opt-btn.wrong-opt  { border-color:rgba(251,113,133,0.48)!important; background:rgba(251,113,133,0.07)!important; color:var(--rose)!important; }\n.opt-btn:disabled   { cursor: default; }\n.opt-letter { width:24px; height:24px; min-width:24px; background:rgba(255,255,255,0.06); border-radius:7px; display:flex; align-items:center; justify-content:center; font-size:.68rem; font-weight:800; font-family:\'Manrope\', sans-serif; color:var(--tx-2); }\n\n/* Feedback */\n.feedback-wrap { background:var(--glass); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid var(--bdr); border-radius:var(--r-lg); padding:18px 20px; margin-bottom:14px; }\n.feedback-wrap.hidden { display: none!important; }\n.fb-msg { font-size:.97rem; font-weight:700; margin-bottom:8px; font-family:\'Manrope\', sans-serif; }\n.fb-msg.correct-fb { color: var(--mint); }\n.fb-msg.wrong-fb   { color: var(--rose); }\n.fb-exp { font-size:.82rem; color:var(--tx-1); line-height:1.6; margin-bottom:10px; }\n.fb-followup { font-size:.8rem; color:var(--indigo-l); margin-bottom:10px; font-style:italic; }\n.fb-followup.hidden { display: none!important; }\n.ml-explain { font-size:.75rem; color:var(--tx-2); font-style:italic; margin-bottom:8px; }\n.study-guide-wrap { background:rgba(167,139,250,0.06); border:1px solid rgba(167,139,250,0.16); border-radius:var(--r); padding:15px 17px; margin-bottom:12px; }\n.study-guide-wrap.hidden { display: none!important; }\n.sg-header { display:flex; align-items:center; gap:6px; margin-bottom:10px; }\n.sg-header h3 { font-size:.8rem; font-weight:700; font-family:\'Manrope\', sans-serif; }\n.sg-body,.sg-line { font-size:.8rem; color:rgba(200,210,240,.55); line-height:1.65; margin-bottom:5px; }\n\n/* ─────────────────────────────────────────────\n   DASHBOARD — SaaS analytics panel\n   ───────────────────────────────────────────── */\n.big-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:18px; }\n.bs {\n  background:var(--glass); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);\n  border:1px solid var(--bdr); border-radius:var(--r); padding:16px 13px; text-align:center;\n  transition:all .22s;\n}\n.bs:hover { border-color:rgba(99,102,241,0.30); transform:translateY(-3px); box-shadow:var(--glow-card); }\n.bs-icon { font-size:1.2rem; margin-bottom:5px; }\n.bs-val  { font-family:\'Manrope\', sans-serif; font-size:1.5rem; font-weight:800; background:var(--g-brand); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }\n.bs-lbl  { font-size:.6rem; color:var(--tx-2); font-weight:700; text-transform:uppercase; letter-spacing:.8px; margin-top:3px; }\n\n.lp-card { background:var(--glass); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border:1px solid var(--bdr); border-radius:var(--r-lg); padding:16px 18px; margin-bottom:14px; }\n.lp-top  { display:flex; justify-content:space-between; align-items:center; margin-bottom:9px; }\n.lp-top h3 { font-size:.84rem; font-weight:700; font-family:\'Manrope\', sans-serif; }\n.lp-pct  { font-family:\'Manrope\', sans-serif; font-size:.9rem; font-weight:800; background:var(--g-brand); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }\n.lp-bar-wrap { height:4px; background:rgba(255,255,255,0.05); border-radius:99px; overflow:hidden; margin-bottom:7px; }\n.lp-bar { height:100%; background:var(--g-brand); border-radius:99px; transition:width .7s ease; }\n.lp-diff,.lp-xp { font-size:.71rem; color:var(--tx-1); font-style:italic; }\n\n.charts-row { display:grid; grid-template-columns:1fr 1fr; gap:13px; margin-bottom:13px; }\n.chart-card {\n  background:var(--glass); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);\n  border:1px solid var(--bdr); border-radius:var(--r-lg); padding:18px; margin-bottom:13px;\n}\n.chart-card.half { margin-bottom:0; }\n.chart-wrap { position:relative; height:200px; }\n.chart-wrap.tall { height:260px; }\n\n.tt-row { display:grid; grid-template-columns:2fr 1fr 1fr 1fr 1fr 1fr; gap:8px; align-items:center; padding:9px 12px; border-radius:9px; font-size:.78rem; margin-bottom:3px; }\n.tt-row.header { font-size:.62rem; font-weight:700; color:var(--tx-2); text-transform:uppercase; letter-spacing:.7px; background:transparent; padding-bottom:4px; }\n.tt-row:not(.header) { background:var(--glass); border:1px solid var(--bdr); transition:all .14s; }\n.tt-row:not(.header):hover { border-color:rgba(99,102,241,0.28); transform:translateX(2px); }\n.mastery-bar-wrap { height:4px; background:rgba(255,255,255,0.055); border-radius:99px; overflow:hidden; width:100%; }\n.mastery-bar { height:100%; border-radius:99px; background:var(--g-brand); }\n\n.cal-card { background:var(--glass); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border:1px solid var(--bdr); border-radius:var(--r-lg); padding:16px; margin-bottom:14px; }\n.streak-calendar { display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; }\n.cal-day { width:12px; height:12px; border-radius:3px; background:rgba(255,255,255,0.04); cursor:default; transition:transform .14s; }\n.cal-day.studied { background:var(--indigo); box-shadow:0 0 6px rgba(99,102,241,0.30); }\n.cal-day.today   { outline:1.5px solid rgba(99,102,241,0.55); outline-offset:1px; }\n.cal-day:hover   { transform:scale(1.35); }\n\n.topics-list,.weak-list { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }\n.topic-tag  { background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.18); color:var(--indigo-l); border-radius:var(--r-p); padding:3px 10px; font-size:.72rem; font-weight:600; cursor:pointer; transition:all .14s; }\n.topic-tag:hover { background:rgba(99,102,241,0.15); }\n.weak-tag,.weak-tag2 { background:rgba(251,113,133,0.07); border:1px solid rgba(251,113,133,0.17); color:var(--rose); border-radius:var(--r-p); padding:3px 10px; font-size:.72rem; font-weight:600; display:inline-block; margin:2px; cursor:pointer; transition:all .14s; }\n.weak-tag:hover,.weak-tag2:hover { background:rgba(251,113,133,0.14); }\n.wt-pill { background:rgba(251,113,133,0.07); border:1px solid rgba(251,113,133,0.17); color:var(--rose); border-radius:var(--r-p); padding:4px 11px; font-size:.73rem; font-weight:600; margin:3px; cursor:pointer; transition:all .16s; display:inline-block; }\n.wt-pill:hover { background:rgba(251,113,133,0.14); }\n.cw-row { display:flex; gap:10px; margin-bottom:13px; flex-wrap:wrap; }\n.cw { flex:1; min-width:80px; background:var(--glass); border:1px solid var(--bdr); border-radius:12px; padding:12px; text-align:center; }\n.cw.green  { border-color:rgba(52,211,153,0.18); }\n.cw.red    { border-color:rgba(251,113,133,0.18); }\n.cw.purple { border-color:rgba(167,139,250,0.18); }\n.cw-v { font-family:\'Manrope\', sans-serif; font-size:1.4rem; font-weight:800; }\n.cw-l { font-size:.6rem; color:var(--tx-2); font-weight:700; text-transform:uppercase; letter-spacing:.7px; margin-top:3px; }\n.export-row { display:flex; gap:8px; margin-top:13px; flex-wrap:wrap; }\n\n/* ─────────────────────────────────────────────\n   LEADERBOARD\n   ───────────────────────────────────────────── */\n.lb-table   { margin-bottom:18px; }\n.lb-loading { color:var(--tx-1); font-size:.84rem; padding:20px 0; text-align:center; }\n.lb-row-item { display:flex; align-items:center; gap:12px; background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--bdr); border-radius:var(--r); padding:12px 16px; margin-bottom:7px; transition:all .18s; }\n.lb-row-item:hover { border-color:rgba(99,102,241,0.28); transform:translateX(3px); }\n.lb-rank { width:28px; text-align:center; font-family:\'Manrope\', sans-serif; font-size:.97rem; font-weight:900; color:var(--tx-1); }\n.lb-rank.gold   { color:var(--amber); }\n.lb-rank.silver { color:#94a3b8; }\n.lb-rank.bronze { color:#cd7c2a; }\n.lb-name  { flex:1; font-weight:600; font-size:.88rem; }\n.lb-score { font-family:\'Manrope\', sans-serif; font-weight:800; font-size:.92rem; color:var(--indigo-l); }\n.lb-acc   { font-size:.7rem; color:var(--tx-2); margin-top:2px; }\n.lb-submit-card { background:var(--glass); backdrop-filter:blur(14px); border:1px solid var(--bdr); border-radius:var(--r-lg); padding:18px; margin-bottom:14px; }\n.lb-row { display:flex; gap:8px; align-items:flex-end; flex-wrap:wrap; }\n\n/* ─────────────────────────────────────────────\n   REVIEW\n   ───────────────────────────────────────────── */\n.review-section-title { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:var(--tx-2); margin-bottom:10px; margin-top:16px; }\n.review-item { background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--bdr); border-radius:12px; padding:12px 15px; display:flex; align-items:center; gap:12px; margin-bottom:7px; transition:all .18s; }\n.review-item:hover { border-color:rgba(99,102,241,0.28); transform:translateX(3px); }\n.review-item.due-now { border-color:rgba(251,191,36,0.22); }\n.review-topic    { flex:1; font-weight:600; font-size:.86rem; }\n.review-interval { font-size:.7rem; color:var(--amber); font-weight:700; }\n.review-empty { text-align:center; padding:48px 20px; color:var(--tx-1); }\n.re-icon { font-size:2.5rem; margin-bottom:10px; }\n.re-msg  { font-size:.97rem; font-weight:700; margin-bottom:5px; font-family:\'Manrope\', sans-serif; }\n.re-sub  { font-size:.8rem; }\n\n/* ─────────────────────────────────────────────\n   UPLOAD\n   ───────────────────────────────────────────── */\n.upload-tabs   { display:flex; gap:8px; margin-bottom:18px; flex-wrap:wrap; }\n.upload-card   { background:var(--glass); backdrop-filter:blur(16px); border:1px solid var(--bdr); border-radius:var(--r-xl); padding:22px; margin-bottom:14px; }\n.pdf-drop { border:2px dashed rgba(99,102,241,0.22); border-radius:var(--r-lg); padding:40px 20px; text-align:center; cursor:pointer; transition:all .22s; background:rgba(99,102,241,0.02); }\n.pdf-drop:hover { border-color:rgba(99,102,241,0.48); background:rgba(99,102,241,0.05); }\n.pdf-icon  { font-size:2.1rem; margin-bottom:8px; }\n.pdf-label { font-weight:700; font-size:.92rem; margin-bottom:4px; font-family:\'Manrope\', sans-serif; }\n.pdf-sub   { font-size:.78rem; color:var(--tx-1); }\n.pdf-name  { font-size:.78rem; color:var(--mint); margin-top:8px; font-weight:600; }\n.upload-controls { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; margin-top:14px; }\n\n/* ─────────────────────────────────────────────\n   COGNITIVE REPORT MODAL — glass overlay\n   ───────────────────────────────────────────── */\n.cog-modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.82); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); display:flex; align-items:center; justify-content:center; z-index:500; padding:16px; }\n.cog-modal-overlay.hidden { display:none!important; }\n.cog-modal { background:rgba(10,10,30,0.94); backdrop-filter:blur(28px); -webkit-backdrop-filter:blur(28px); border:1px solid rgba(99,102,241,0.16); border-radius:24px; width:100%; max-width:580px; max-height:88vh; overflow-y:auto; padding:28px; box-shadow:var(--shadow),0 0 60px rgba(99,102,241,0.14); }\n.cog-modal-header { text-align:center; margin-bottom:22px; }\n.cog-title    { font-family:\'Manrope\', sans-serif; font-size:1.3rem; font-weight:800; margin-bottom:5px; }\n.cog-subtitle { font-size:.8rem; color:var(--tx-1); }\n.cr-loading   { text-align:center; padding:28px; }\n.cr-spinner   { width:40px; height:40px; margin:0 auto 12px; border:3px solid rgba(99,102,241,0.10); border-top-color:var(--indigo); border-radius:50%; animation:spin .7s linear infinite; }\n@keyframes spin { to { transform:rotate(360deg); } }\n.cr-loading p { font-size:.81rem; color:var(--tx-1); }\n.cr-hero { display:flex; align-items:center; gap:16px; margin-bottom:18px; }\n.cr-ring-wrap { position:relative; flex-shrink:0; }\n.cr-ring-text { text-align:center; }\n.cr-ring-pct  { font-family:\'Manrope\', sans-serif; font-size:1.42rem; font-weight:900; background:var(--g-brand); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }\n.cr-ring-label{ font-size:.6rem; color:var(--tx-1); font-weight:600; text-transform:uppercase; letter-spacing:.7px; }\n.cr-dom-emoji { font-size:1.8rem; }\n.cr-dom-label { font-size:1.02rem; font-weight:700; font-family:\'Manrope\', sans-serif; margin:3px 0; }\n.cr-dom-sub   { font-size:.76rem; color:var(--tx-1); }\n.cr-section-title { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:var(--tx-2); margin-bottom:10px; display:flex; align-items:center; gap:5px; }\n.cr-state-bars { margin-bottom:16px; }\n.cr-bar-row   { display:flex; align-items:center; gap:8px; margin-bottom:6px; }\n.cr-bar-label,.cr-bar-name { font-size:.74rem; font-weight:600; width:72px; flex-shrink:0; text-transform:capitalize; }\n.cr-bar-name  { width:90px; }\n.cr-bar-track { flex:1; height:5px; background:rgba(255,255,255,0.055); border-radius:99px; overflow:hidden; }\n.cr-bar-fill  { height:100%; border-radius:99px; transition:width .6s ease; }\n.cr-bar-pct   { font-size:.7rem; color:var(--tx-2); width:26px; text-align:right; flex-shrink:0; font-weight:600; }\n.cr-study-plan{ margin-bottom:16px; }\n.cr-no-weak   { font-size:.81rem; color:var(--tx-1); text-align:center; padding:14px 0; }\n.cr-topic-card{ background:rgba(251,113,133,0.05); border:1px solid rgba(251,113,133,0.13); border-radius:10px; padding:10px 13px; margin-bottom:6px; }\n.cr-topic-title{ font-size:.8rem; font-weight:700; margin-bottom:4px; font-family:\'Manrope\', sans-serif; }\n.cr-concept-row{ font-size:.75rem; color:var(--rose); }\n.cr-nocam-msg  { font-size:.8rem; color:var(--tx-1); text-align:center; padding:10px 0; }\n.cr-tip-card   { background:rgba(255,255,255,0.025); border-radius:12px; padding:14px; margin-bottom:8px; border:1px solid rgba(255,255,255,0.055); }\n.cr-tip-header { display:flex; align-items:center; gap:8px; margin-bottom:10px; }\n.cr-tip-section-lbl { font-size:.61rem; font-weight:800; text-transform:uppercase; letter-spacing:1px; color:var(--tx-2); margin:8px 0 5px; display:flex; align-items:center; gap:5px; }\n.cr-tip-row    { font-size:.78rem; color:rgba(200,210,240,.55); line-height:1.55; margin-bottom:4px; padding-left:4px; }\n.cr-tip-emoji  { font-size:1.08rem; }\n.cr-tip-state  { font-size:.78rem; font-weight:700; font-family:\'Manrope\', sans-serif; }\n.cr-tip-pct    { font-size:.65rem; font-weight:700; margin-left:auto; padding:2px 8px; border:1px solid currentColor; border-radius:var(--r-p); opacity:.75; }\n.cr-actions    { margin-top:16px; }\n\n/* ─────────────────────────────────────────────\n   MISTAKE NOTEBOOK\n   ───────────────────────────────────────────── */\n.mn-toolbar { display:flex; gap:10px; align-items:center; margin-bottom:18px; flex-wrap:wrap; }\n.mn-stat-row { display:flex; gap:10px; margin-bottom:18px; flex-wrap:wrap; }\n.mn-stat { flex:1; min-width:80px; background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--bdr); border-radius:12px; padding:14px; text-align:center; }\n.mn-stat-v { display:block; font-size:1.42rem; font-weight:900; font-family:\'Manrope\', sans-serif; }\n.mn-stat-l { display:block; font-size:.57rem; color:var(--tx-2); font-weight:700; text-transform:uppercase; letter-spacing:.8px; margin-top:3px; }\n.mn-group  { margin-bottom:22px; }\n.mn-group-title { font-size:.83rem; font-weight:700; color:var(--violet); margin-bottom:9px; display:flex; align-items:center; gap:7px; font-family:\'Manrope\', sans-serif; }\n.mn-group-count { background:rgba(167,139,250,0.09); border-radius:var(--r-p); padding:2px 9px; font-size:.66rem; }\n.mn-card { background:var(--glass); backdrop-filter:blur(12px); border:1px solid rgba(251,113,133,0.10); border-radius:var(--r); padding:15px; margin-bottom:9px; transition:border-color .18s; }\n.mn-card:hover { border-color:rgba(251,113,133,0.26); }\n.mn-card.mn-retried { border-color:rgba(52,211,153,0.16); opacity:.65; }\n.mn-card-top { display:flex; align-items:center; gap:7px; margin-bottom:9px; flex-wrap:wrap; }\n.mn-concept  { background:rgba(167,139,250,0.08); border:1px solid rgba(167,139,250,0.17); color:var(--violet); border-radius:var(--r-p); padding:2px 9px; font-size:.66rem; font-weight:700; }\n.mn-date     { font-size:.66rem; color:var(--tx-2); margin-left:auto; }\n.mn-done-badge{ background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.20); color:var(--mint); border-radius:var(--r-p); padding:2px 8px; font-size:.66rem; font-weight:700; }\n.mn-question { font-size:.84rem; font-weight:600; margin-bottom:9px; line-height:1.55; }\n.mn-answers  { display:flex; flex-direction:column; gap:4px; margin-bottom:9px; }\n.mn-your-ans   { background:rgba(251,113,133,0.05); border-radius:8px; padding:6px 11px; font-size:.77rem; color:var(--rose); }\n.mn-correct-ans{ background:rgba(52,211,153,0.05); border-radius:8px; padding:6px 11px; font-size:.77rem; color:var(--mint); }\n.mn-explanation{ font-size:.75rem; color:rgba(200,210,240,.42); line-height:1.55; margin-bottom:10px; padding-left:2px; }\n.mn-actions  { display:flex; gap:7px; flex-wrap:wrap; }\n.mn-empty    { text-align:center; padding:52px 20px; color:var(--tx-1); }\n\n/* Emotion history */\n.eh-card { background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--bdr); border-radius:12px; padding:11px 15px; display:flex; justify-content:space-between; align-items:center; margin-bottom:7px; transition:all .18s; }\n.eh-card:hover { border-color:rgba(99,102,241,0.28); transform:translateX(3px); }\n\n/* ─────────────────────────────────────────────\n   AI CHAT\n   ───────────────────────────────────────────── */\n.chat-topic-bar { display:flex; align-items:center; gap:9px; margin-bottom:13px; flex-wrap:wrap; }\n.chat-box { background:rgba(5,5,16,0.50); backdrop-filter:blur(16px); border:1px solid var(--bdr); border-radius:var(--r-lg); padding:15px; min-height:270px; max-height:390px; overflow-y:auto; display:flex; flex-direction:column; gap:11px; margin-bottom:11px; }\n.chat-bubble { max-width:82%; padding:10px 14px; border-radius:14px; font-size:.85rem; line-height:1.65; }\n.cb-lbl { display:block; font-size:.55rem; font-weight:800; text-transform:uppercase; letter-spacing:.8px; margin-bottom:4px; opacity:.5; font-family:\'Manrope\', sans-serif; }\n.chat-user { background:rgba(99,102,241,0.11); border:1px solid rgba(99,102,241,0.24); align-self:flex-end; color:#e0eaff; }\n.chat-ai   { background:rgba(52,211,153,0.06); border:1px solid rgba(52,211,153,0.15); align-self:flex-start; color:#e0eaff; }\n.chat-typing { opacity:.45; }\n@keyframes typingPulse { 0%,100%{opacity:.2} 50%{opacity:1} }\n.typing-dots { animation:typingPulse 1.1s infinite; font-size:1.2rem; letter-spacing:3px; }\n.chat-input-wrap { display:flex; gap:7px; margin-bottom:9px; }\n.chat-suggestions { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }\n\n/* ─────────────────────────────────────────────\n   BREATHING EXERCISE\n   ───────────────────────────────────────────── */\n.breathing-overlay { position:fixed; inset:0; background:rgba(0,0,0,.90); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); z-index:9999; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:22px; }\n.breathing-circle  { width:120px; height:120px; border-radius:50%; background:radial-gradient(circle,var(--indigo) 0%,var(--violet) 100%); display:flex; align-items:center; justify-content:center; color:#fff; font-size:.97rem; font-weight:700; animation:breathe 4s ease-in-out infinite alternate; box-shadow:var(--glow-btn); }\n@keyframes breathe { from{transform:scale(.7);opacity:.6} to{transform:scale(1.3);opacity:1} }\n.breathing-text { color:#fff; font-size:1rem; font-weight:600; text-align:center; }\n\n/* ─────────────────────────────────────────────\n   TOAST\n   ───────────────────────────────────────────── */\n.toast { position:fixed; bottom:24px; left:50%; transform:translateX(-50%) translateY(18px); background:rgba(5,5,16,0.97); backdrop-filter:blur(12px); border:1px solid rgba(99,102,241,0.24); color:var(--tx-0); padding:9px 20px; border-radius:12px; font-size:.81rem; font-weight:600; box-shadow:var(--shadow); opacity:0; pointer-events:none; transition:all .25s cubic-bezier(.4,0,.2,1); z-index:999; }\n.toast.show { opacity:1; transform:translateX(-50%) translateY(0); }\n\n/* Loading overlay */\n.loading-overlay { position:fixed; inset:0; background:rgba(5,5,16,0.90); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); display:flex; align-items:center; justify-content:center; z-index:400; }\n.loading-overlay.hidden { display:none!important; }\n.loader-box { text-align:center; }\n.spinner { width:42px; height:42px; margin:0 auto 14px; border:3px solid rgba(99,102,241,0.10); border-top-color:var(--indigo); border-radius:50%; animation:spin .7s linear infinite; }\n.loading-msg { font-size:.84rem; color:var(--tx-1); font-weight:500; }\n\n/* ─────────────────────────────────────────────\n   ANIMATED STAT COUNTERS  (count-up via JS uses same .sv / .bs-val)\n   ───────────────────────────────────────────── */\n@keyframes countUp {\n  from { opacity:0; transform:translateY(8px); }\n  to   { opacity:1; transform:translateY(0); }\n}\n.sv, .bs-val { animation: countUp .45s ease both; }\n\n/* ─────────────────────────────────────────────\n   UTILITY\n   ───────────────────────────────────────────── */\n.hidden    { display:none!important; }\n.empty-tag,.nb { font-size:.8rem; color:var(--tx-1); font-style:italic; }\n\n/* ─────────────────────────────────────────────\n   RESPONSIVE\n   ───────────────────────────────────────────── */\n@media(max-width:900px) {\n  .feature-showcase { grid-template-columns:repeat(2,1fr); }\n  .big-stats         { grid-template-columns:repeat(2,1fr); }\n  .charts-row        { grid-template-columns:1fr; }\n  .stats-row         { grid-template-columns:repeat(3,1fr); }\n  .hero-text h1      { font-size:1.9rem; }\n}\n@media(max-width:640px) {\n  :root { --sidebar-w:50px; }\n  .main       { padding:16px 13px 56px; }\n  .feature-showcase { grid-template-columns:1fr; }\n  .stats-row  { grid-template-columns:repeat(2,1fr); }\n  .hero       { padding:28px 18px 24px; }\n  .hero-text h1 { font-size:1.5rem; }\n  .hero-actions { flex-direction:column; align-items:stretch; }\n  .btn-primary,.btn-ghost { justify-content:center; width:100%; }\n  .xp-ring    { display:none; }\n  .hero-inner { flex-direction:column; }\n  .gen-row,.gen-controls { flex-direction:column; }\n  .gen-btn,.topic-input,.sel { width:100%; }\n  .nav-btn    { width:34px; height:34px; border-radius:9px; font-size:.92rem; }\n  .logo       { width:30px; height:30px; font-size:.9rem; border-radius:9px; }\n  .nav-btn::after { display:none; }\n  .big-stats  { grid-template-columns:repeat(2,1fr); }\n  .cog-modal  { padding:18px 14px; }\n  .live-strip { display:none; }\n  .cam-panel  { bottom:10px; right:10px; }\n}\n\n  \n\n/* ════════════════════════════════════════════════════════════════\n   AUTH LAYER — login / register overlay\n   Sits above everything, shares design tokens with app\n   ════════════════════════════════════════════════════════════════ */\n\n/* Full-screen overlay — covers the entire app until authenticated */\n.auth-overlay {\n  position: fixed; inset: 0; z-index: 9000;\n  background: var(--bg);\n  display: flex; align-items: center; justify-content: center;\n  padding: 20px;\n  animation: screenReveal .35s cubic-bezier(.4,0,.2,1);\n}\n.auth-overlay.hidden { display: none !important; }\n\n/* Ambient glow same as app */\n.auth-overlay::before {\n  content: \'\'; position: absolute; inset: 0; pointer-events: none;\n  background:\n    radial-gradient(ellipse 70% 55% at 50% 0%, rgba(99,102,241,0.13) 0%, transparent 65%),\n    radial-gradient(ellipse 50% 40% at 85% 80%, rgba(167,139,250,0.07) 0%, transparent 55%);\n}\n.auth-overlay::after {\n  content: \'\'; position: absolute; inset: 0; pointer-events: none;\n  background-image: radial-gradient(rgba(99,102,241,0.05) 1px, transparent 1px);\n  background-size: 28px 28px;\n  mask-image: radial-gradient(ellipse 90% 70% at 50% 25%, black 30%, transparent 100%);\n}\n\n/* Central card */\n.auth-card {\n  position: relative; z-index: 1;\n  width: 100%; max-width: 420px;\n  background: var(--glass);\n  backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);\n  border: 1px solid var(--bdr);\n  border-radius: var(--r-xl);\n  padding: 38px 36px 32px;\n  box-shadow: 0 20px 64px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04);\n}\n\n/* Brand header inside card */\n.auth-brand {\n  text-align: center; margin-bottom: 28px;\n}\n.auth-logo {\n  display: inline-flex; align-items: center; justify-content: center;\n  width: 46px; height: 46px;\n  background: var(--g-indigo);\n  border-radius: 14px; font-size: 1.4rem;\n  box-shadow: var(--glow-btn); margin-bottom: 12px;\n}\n.auth-brand-name {\n  font-family: \'Manrope\', sans-serif;\n  font-size: 1.55rem; font-weight: 800;\n  letter-spacing: -0.03em; color: var(--tx-0);\n}\n.auth-brand-sub {\n  font-size: .78rem; color: var(--tx-2); margin-top: 3px;\n  font-weight: 500;\n}\n\n/* Tab toggle (Login / Register) */\n.auth-tabs {\n  display: flex; gap: 4px;\n  background: rgba(255,255,255,0.038);\n  border: 1px solid var(--bdr);\n  border-radius: 12px; padding: 4px;\n  margin-bottom: 24px;\n}\n.auth-tab {\n  flex: 1; padding: 7px 8px;\n  background: transparent; border: none;\n  color: var(--tx-2); border-radius: 9px;\n  font-size: .76rem; font-weight: 600;\n  font-family: \'Inter\', sans-serif;\n  cursor: pointer; transition: all .18s;\n  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;\n}\n.auth-tab.active {\n  background: var(--g-indigo);\n  color: #fff;\n  box-shadow: 0 2px 12px rgba(99,102,241,0.35);\n}\n.auth-tab:hover:not(.active) { color: var(--tx-0); }\n\n/* Form panels */\n.auth-form { display: flex; flex-direction: column; gap: 14px; }\n/* auth-form visibility now controlled via inline styles */\n\n/* Field group */\n.auth-field { display: flex; flex-direction: column; gap: 5px; }\n.auth-label {\n  font-size: .72rem; font-weight: 700;\n  color: var(--tx-2); text-transform: uppercase; letter-spacing: .9px;\n}\n\n/* Input */\n.auth-input {\n  background: rgba(255,255,255,0.038);\n  border: 1px solid var(--bdr);\n  color: var(--tx-0); border-radius: 11px;\n  padding: 12px 15px; font-size: .9rem;\n  font-family: \'Inter\', sans-serif;\n  outline: none; transition: all .2s;\n}\n.auth-input:focus {\n  border-color: rgba(99,102,241,0.55);\n  background: rgba(99,102,241,0.05);\n  box-shadow: 0 0 0 3px rgba(99,102,241,0.10);\n}\n.auth-input::placeholder { color: var(--tx-2); }\n.auth-input.error { border-color: rgba(251,113,133,0.55) !important; }\n\n/* Password wrapper with show/hide toggle */\n.auth-pw-wrap { position: relative; }\n.auth-pw-wrap .auth-input { width: 100%; padding-right: 44px; }\n.auth-pw-toggle {\n  position: absolute; right: 10px; top: 50%;\n  transform: translateY(-50%);\n  background: none; border: none;\n  color: var(--tx-2); cursor: pointer;\n  font-size: .65rem; font-weight: 700;\n  font-family: \'Inter\', sans-serif; letter-spacing: .3px;\n  padding: 3px 6px; border-radius: 5px; transition: all .15s;\n}\n.auth-pw-toggle:hover { color: var(--indigo-l); background: rgba(99,102,241,0.08); }\n\n/* Error message */\n.auth-error {\n  background: rgba(251,113,133,0.08);\n  border: 1px solid rgba(251,113,133,0.22);\n  color: #ff8099; border-radius: 9px;\n  padding: 9px 13px; font-size: .80rem;\n  font-weight: 500; line-height: 1.5;\n  display: none;\n}\n.auth-error.show { display: block; }\n\n/* Submit button */\n.auth-btn {\n  background: var(--g-indigo); border: none; color: #fff;\n  padding: 13px 20px; border-radius: 12px;\n  font-size: .91rem; font-weight: 700;\n  font-family: \'Inter\', sans-serif;\n  cursor: pointer; transition: all .22s;\n  box-shadow: var(--sh-brand); margin-top: 4px;\n  position: relative; overflow: hidden;\n}\n.auth-btn::after {\n  content: \'\'; position: absolute; inset: 0;\n  background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);\n  opacity: 0; transition: opacity .2s;\n}\n.auth-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(99,102,241,0.50); }\n.auth-btn:hover::after { opacity: 1; }\n.auth-btn:active { transform: translateY(0); }\n.auth-btn:disabled { opacity: .5; cursor: not-allowed; transform: none; box-shadow: none; }\n\n/* Spinner inside button */\n.auth-btn-spinner {\n  display: inline-block; width: 14px; height: 14px;\n  border: 2px solid rgba(255,255,255,0.3);\n  border-top-color: #fff; border-radius: 50%;\n  animation: spin .6s linear infinite;\n  vertical-align: middle; margin-right: 6px;\n}\n\n/* Divider */\n.auth-divider {\n  display: flex; align-items: center; gap: 10px;\n  font-size: .72rem; color: var(--tx-2); margin: 4px 0;\n}\n.auth-divider::before, .auth-divider::after {\n  content: \'\'; flex: 1; height: 1px; background: var(--bdr);\n}\n\n/* Footer note */\n.auth-footer {\n  text-align: center; margin-top: 16px;\n  font-size: .75rem; color: var(--tx-2);\n}\n\n/* User menu in sidebar (shown when logged in) */\n.user-menu {\n  margin-top: auto; flex-shrink: 0;\n  display: flex; flex-direction: column;\n  align-items: center; gap: 4px;\n}\n.user-avatar {\n  width: 36px; height: 36px;\n  background: var(--g-indigo);\n  border-radius: 10px;\n  display: flex; align-items: center; justify-content: center;\n  font-size: .88rem; font-weight: 800; color: #fff;\n  font-family: \'Manrope\', sans-serif;\n  cursor: pointer; transition: transform .18s;\n  box-shadow: var(--sh-brand);\n  position: relative;\n}\n.user-avatar:hover { transform: scale(1.08); }\n\n/* Logout button (appears below avatar on hover) */\n.logout-btn {\n  background: rgba(251,113,133,0.08);\n  border: 1px solid rgba(251,113,133,0.20);\n  color: #ff8099; border-radius: 8px;\n  padding: 5px 10px; font-size: .64rem; font-weight: 700;\n  font-family: \'Inter\', sans-serif;\n  cursor: pointer; transition: all .16s; white-space: nowrap;\n  letter-spacing: .4px; text-transform: uppercase;\n}\n.logout-btn:hover { background: rgba(251,113,133,0.15); }\n\n/* Welcome banner on home screen (shown after login) */\n.welcome-banner {\n  background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(167,139,250,0.05));\n  border: 1px solid rgba(99,102,241,0.16);\n  border-radius: var(--r-md);\n  padding: 11px 16px; margin-bottom: 18px;\n  font-size: .84rem; color: var(--tx-1);\n  display: flex; align-items: center; gap: 8px;\n}\n.welcome-banner strong { color: var(--tx-0); font-weight: 700; }\n.welcome-banner.hidden { display: none !important; }\n\n/* Responsive */\n@media (max-width: 480px) {\n  .auth-card { padding: 26px 20px 22px; }\n  .auth-brand-name { font-size: 1.3rem; }\n}\n\n/* Change Password Modal */\n.cpw-overlay {\n  position: fixed; inset: 0; z-index: 9100;\n  background: rgba(0,0,0,0.75);\n  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);\n  display: flex; align-items: center; justify-content: center;\n  padding: 20px;\n}\n.cpw-overlay.hidden { display: none !important; }\n.cpw-card {\n  position: relative; z-index: 1;\n  width: 100%; max-width: 380px;\n  background: rgba(10,10,30,0.97);\n  backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);\n  border: 1px solid var(--bdr);\n  border-radius: var(--r-xl);\n  padding: 30px 28px 26px;\n  box-shadow: 0 20px 64px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04);\n}\n.cpw-title {\n  font-family: \'Manrope\', sans-serif;\n  font-size: 1.1rem; font-weight: 800;\n  letter-spacing: -0.025em; margin-bottom: 6px;\n  color: var(--tx-0);\n}\n.cpw-sub {\n  font-size: .78rem; color: var(--tx-2);\n  margin-bottom: 22px;\n}\n.cpw-close {\n  position: absolute; top: 16px; right: 16px;\n  background: var(--glass); border: 1px solid var(--bdr);\n  color: var(--tx-1); border-radius: 8px;\n  width: 28px; height: 28px;\n  display: flex; align-items: center; justify-content: center;\n  cursor: pointer; font-size: .8rem; transition: all .15s;\n}\n.cpw-close:hover { background: rgba(251,113,133,0.12); color: var(--rose); }\n.cpw-success {\n  background: rgba(52,211,153,0.08);\n  border: 1px solid rgba(52,211,153,0.22);\n  color: var(--mint); border-radius: 9px;\n  padding: 9px 13px; font-size: .80rem;\n  font-weight: 500; display: none; margin-bottom: 12px;\n}\n.cpw-success.show { display: block; }\n\n/* Change-pw trigger button under avatar */\n.change-pw-btn {\n  width: 40px; height: 40px;\n  background: transparent; border: 1px solid transparent;\n  border-radius: 11px; cursor: pointer; font-size: 1rem;\n  color: var(--tx-2);\n  transition: all .18s cubic-bezier(.4,0,.2,1);\n  flex-shrink: 0; position: relative;\n  display: none; align-items: center; justify-content: center;\n}\n.change-pw-btn::after {\n  content: attr(title);\n  position: absolute; left: calc(100% + 10px); top: 50%;\n  transform: translateY(-50%) scale(.85);\n  background: rgba(5,5,16,0.97);\n  border: 1px solid rgba(99,102,241,0.35);\n  color: var(--tx-0); font-size: .67rem; font-weight: 600;\n  padding: 4px 10px; border-radius: 8px;\n  white-space: nowrap; pointer-events: none;\n  opacity: 0; transition: all .14s; z-index: 300;\n  font-family: \'Inter\', sans-serif; box-shadow: var(--shadow);\n}\n.change-pw-btn:hover::after { opacity: 1; transform: translateY(-50%) scale(1); }\n.change-pw-btn:hover {\n  background: rgba(99,102,241,0.11);\n  border-color: rgba(99,102,241,0.26);\n  color: var(--indigo-l); transform: scale(1.07);\n}\n.change-pw-btn:hover { background: rgba(99,102,241,0.16); }\n\n/* Forgot Password */\n.reset-code-card {\n  background: linear-gradient(135deg,rgba(99,102,241,0.10),rgba(167,139,250,0.07));\n  border: 1px solid rgba(99,102,241,0.30);\n  border-radius: var(--r); padding: 16px; text-align: center; margin-bottom: 4px;\n}\n.reset-code-label { font-size:.68rem; font-weight:700; color:var(--tx-2); text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }\n.reset-code-value { font-family:\'Manrope\',sans-serif; font-size:2rem; font-weight:900; letter-spacing:.18em;\n  background:var(--g-brand); -webkit-background-clip:text; -webkit-text-fill-color:transparent;\n  background-clip:text; line-height:1.1; }\n.reset-code-expiry { font-size:.70rem; color:var(--amber); margin-top:6px; font-weight:600; }\n.reset-step { flex-direction:column; gap:14px; }\n.reset-step.active { display:flex; }\n.reset-step:not(.active) { display:none; }\n\n@keyframes screenReveal { from { opacity:0; transform:scale(.97); } to { opacity:1; transform:scale(1); } }\n</style>\n</head>\n<body>\n<div id="app">\n\n<nav class="sidebar">\n    <div class="logo">&#127891;</div>\n    <button class="nav-btn active" data-screen="home" title="Home">&#127968;</button>\n    <button class="nav-btn" data-screen="quiz" title="Quiz">&#9889;</button>\n    <button class="nav-btn" data-screen="dashboard" title="Dashboard">&#128202;</button>\n    <button class="nav-btn" data-screen="upload" title="Upload">&#128196;</button>\n    <button class="nav-btn" data-screen="leaderboard" title="Leaderboard">&#127942;</button>\n    <button class="nav-btn" data-screen="review" title="Review">&#128337;</button>\n    <button class="nav-btn" data-screen="mistakes" title="Mistake Notebook">&#128218;</button>\n    <button class="nav-btn" data-screen="emotion-history" title="Emotion History">&#128200;</button>\n    <button class="nav-btn" data-screen="chat" title="AI Tutor">&#129302;</button>\n    <div class="user-menu">\n    <div class="level-pill"><span class="lv">LV</span><span id="lv-num">1</span></div>\n    <div id="user-avatar" class="user-avatar" style="display:none" title=""></div>\n    <button id="logout-btn" class="logout-btn" style="display:none" onclick="logout()">⎋ Out</button>\n    <button id="change-pw-btn" class="change-pw-btn" style="display:none" onclick="openCPW()" title="Change Password">🔐</button>\n  </div>\n  </nav>\n\n  <main class="main">\n\n<section id="screen-home" class="screen active">\n      <div class="hero">\n        <div class="hero-inner">\n          <div class="hero-text">\n            <div class="hero-eyebrow"><span class="hero-eyebrow-dot"></span>AI-Powered Cognitive Learning Platform</div>\n            <h1>EduMate <span class="grad">360</span></h1>\n            <p class="hero-tagline">Generate quizzes on any topic · Analyse focus with computer vision · Track performance with smart analytics · Improve retention using spaced repetition</p>\n            <div class="hero-actions">\n              <button class="btn-primary" onclick="document.getElementById(\'topic-input\').focus();document.getElementById(\'topic-input\').scrollIntoView({behavior:\'smooth\'})">&#9889; Start Learning</button>\n              <button class="btn-ghost" onclick="showScreen(\'dashboard\')">&#128202; View Analytics</button>\n            </div>\n          </div>\n          <div class="xp-ring">\n            <div class="xp-num" id="xp-num">0</div>\n            <div class="xp-lbl">XP</div>\n            <div class="xp-bar-wrap"><div class="xp-bar" id="xp-bar"></div></div>\n            <div class="xp-to" id="xp-to">200 to Level 2</div>\n          </div>\n        </div>\n      </div>\n      <div id="welcome-banner" class="welcome-banner hidden">\n        👋 Welcome back, <strong id="welcome-name"></strong>! Ready to learn?\n      </div>\n      <div id="due-alert" class="due-alert hidden"></div>\n      <div class="gen-card">\n        <h2 class="gen-title">&#128269; What do you want to study?</h2>\n        <div class="gen-row">\n          <input id="topic-input" class="topic-input" type="text"\n            placeholder="e.g. French Revolution, Machine Learning, Photosynthesis..."\n            autocomplete="off"/>\n        </div>\n        <div class="gen-row gen-controls">\n          <div class="control-group">\n            <label>Difficulty</label>\n            <select id="diff-select" class="sel">\n              <option value="beginner">Beginner</option>\n              <option value="easy">Easy</option>\n              <option value="medium" selected>Medium</option>\n              <option value="hard">Hard</option>\n              <option value="expert">Expert</option>\n            </select>\n          </div>\n          <div class="control-group">\n            <label>Questions</label>\n            <select id="numq-select" class="sel">\n              <option value="3">3</option>\n              <option value="5" selected>5</option>\n              <option value="7">7</option>\n              <option value="10">10</option>\n              <option value="15">15</option>\n            </select>\n          </div>\n          <div class="control-group">\n            <label>Type</label>\n            <select id="type-select" class="sel">\n              <option value="mcq">MCQ</option>\n              <option value="truefalse">True/False</option>\n              <option value="short">Short Answer</option>\n              <option value="mixed">Mixed</option>\n            </select>\n          </div>\n          <button id="generate-btn" class="gen-btn" onclick="generateQuestions()">&#10024; Generate</button>\n        </div>\n        <div id="gen-error" class="gen-error hidden"></div>\n      </div>\n      <div class="stats-row">\n        <div class="sc"><span class="si">&#128293;</span><span class="sv" id="hs-streak">0</span><span class="sl">Streak</span></div>\n        <div class="sc"><span class="si">&#127919;</span><span class="sv" id="hs-acc">0%</span><span class="sl">Accuracy</span></div>\n        <div class="sc"><span class="si">&#11088;</span><span class="sv" id="hs-score">0</span><span class="sl">Score</span></div>\n        <div class="sc"><span class="si">&#128218;</span><span class="sv" id="hs-topics">0</span><span class="sl">Topics</span></div>\n        <div class="sc"><span class="si">&#128197;</span><span class="sv" id="hs-days">0</span><span class="sl">Day Streak</span></div>\n      </div>\n      <div id="home-badges" class="badge-row hidden">\n        <h3 class="section-label">&#127885; Your Badges</h3>\n        <div id="home-badge-list" class="badge-list"></div>\n      </div>\n\n      <!-- ══ FEATURE SHOWCASE ══ -->\n      <div class="home-section-title">&#128161; Core Features</div>\n      <div class="feature-showcase">\n        <div class="fsc-card" onclick="showScreen(\'quiz\')">\n          <div class="fsc-icon-wrap fsc-blue"><span>&#129302;</span></div>\n          <div class="fsc-body">\n            <div class="fsc-title">AI Quiz Generator</div>\n            <div class="fsc-desc">Generate MCQ, True/False &amp; short-answer questions instantly using LLaMA 3.3-70B. Supports topic input, PDF upload, or paste text.</div>\n            <div class="fsc-badge">Groq · LLaMA 3.3-70B</div>\n          </div>\n        </div>\n        <div class="fsc-card" onclick="showScreen(\'quiz\')">\n          <div class="fsc-icon-wrap fsc-violet"><span>&#129504;</span></div>\n          <div class="fsc-body">\n            <div class="fsc-title">Cognitive Camera</div>\n            <div class="fsc-desc">Uses MediaPipe FaceLandmarker (478 facial landmarks) to detect focus, fatigue, anxiety &amp; overload in real time during quizzes.</div>\n            <div class="fsc-badge">MediaPipe · 478 Landmarks</div>\n          </div>\n        </div>\n        <div class="fsc-card" onclick="showScreen(\'dashboard\')">\n          <div class="fsc-icon-wrap fsc-teal"><span>&#128200;</span></div>\n          <div class="fsc-body">\n            <div class="fsc-title">Learning Analytics</div>\n            <div class="fsc-desc">Track accuracy trends, topic mastery, study streaks &amp; cognitive states across sessions with live Chart.js visualisations.</div>\n            <div class="fsc-badge">Chart.js · SQLite</div>\n          </div>\n        </div>\n        <div class="fsc-card" onclick="showScreen(\'review\')">\n          <div class="fsc-icon-wrap fsc-amber"><span>&#128337;</span></div>\n          <div class="fsc-body">\n            <div class="fsc-title">Spaced Repetition</div>\n            <div class="fsc-desc">SM-2 algorithm schedules topic reviews at optimal intervals — maximises long-term memory retention automatically.</div>\n            <div class="fsc-badge">SM-2 Algorithm</div>\n          </div>\n        </div>\n        <div class="fsc-card" onclick="showScreen(\'mistakes\')">\n          <div class="fsc-icon-wrap fsc-red"><span>&#128218;</span></div>\n          <div class="fsc-body">\n            <div class="fsc-title">Mistake Notebook</div>\n            <div class="fsc-desc">Every wrong answer is auto-saved with AI-generated study guides. Filter by topic, mark retried, or ask the AI Tutor to explain.</div>\n            <div class="fsc-badge">Auto-saved · AI Study Guides</div>\n          </div>\n        </div>\n        <div class="fsc-card" onclick="showScreen(\'chat\')">\n          <div class="fsc-icon-wrap fsc-green"><span>&#128172;</span></div>\n          <div class="fsc-body">\n            <div class="fsc-title">AI Tutor Chat</div>\n            <div class="fsc-desc">Multi-turn conversational AI tutor. Context-aware, remembers your quiz topic, and generates practice questions on demand.</div>\n            <div class="fsc-badge">Multi-turn · Context-aware</div>\n          </div>\n        </div>\n      </div>\n\n      <!-- ══ TECH STACK SECTION ══ -->\n      <div class="home-section-title">&#9881; Technology Stack</div>\n      <div class="tech-stack-row">\n        <div class="tech-pill tech-ai">&#129302; LLaMA 3.3-70B via Groq</div>\n        <div class="tech-pill tech-cv">&#128247; MediaPipe FaceLandmarker</div>\n        <div class="tech-pill tech-ml">&#129302; GradientBoosting ML</div>\n        <div class="tech-pill tech-chart">&#128200; Chart.js Analytics</div>\n        <div class="tech-pill tech-db">&#128451; SQLite + Flask</div>\n        <div class="tech-pill tech-algo">&#128337; SM-2 Spaced Repetition</div>\n        <div class="tech-pill tech-pdf">&#128196; PyPDF2 Extraction</div>\n        <div class="tech-pill tech-voice">&#128266; Web Speech API</div>\n      </div>\n\n    </section>\n\n    <!-- QUIZ -->\n<section id="screen-quiz" class="screen">\n      <div class="quiz-top">\n        <div class="quiz-meta">\n          <div class="q-topic-badge" id="q-topic-badge">Topic</div>\n          <div class="q-diff-badge diff-medium" id="q-diff-badge">Medium</div>\n          <div class="q-num" id="q-num">Q 1/5</div>\n        </div>\n        <div class="quiz-actions">\n          <button class="abtn" id="voice-btn" onclick="toggleVoice()">&#128266; Voice</button>\n          <button class="abtn" id="cam-btn" onclick="toggleCam()">📷 Cam</button>\n          <button class="abtn" onclick="showScreen(\'home\')">&#8592; Back</button>\n        </div>\n      </div>\n      <div class="qprog-wrap"><div class="qprog-fill" id="qprog"></div></div>\n      <div class="live-strip">\n        <span>&#11088; <b id="ls-score">0</b></span>\n        <span>&#128293; <b id="ls-streak">0</b></span>\n        <span>&#127919; <b id="ls-acc">0%</b></span>\n        <span class="ml-tag" id="ml-tag">ML: medium</span>\n        <span id="timer-badge" class="timer-badge">&#9201; 0s</span>\n        <span id="cog-badge" class="cog-badge hidden"></span>\n      </div>\n      <!-- CAMERA / EMOTION PANEL -->\n      <div id="cam-panel" class="cam-panel hidden">\n        <div class="cam-panel-topbar">\n          <span class="cam-panel-title">🧠 Cognitive Cam</span>\n          <div class="cam-panel-actions">\n            <button id="cam-hide-btn" class="cam-hide-btn" onclick="toggleCamView()" title="Hide video feed">Hide</button>\n            <button class="cam-close-btn" onclick="toggleCam()" title="Turn off camera">✕ Off</button>\n          </div>\n        </div>\n        <div id="cam-body" class="cam-body">\n        <div class="cam-left">\n          <div class="cam-video-wrap">\n            <video id="webcam" autoplay muted playsinline width="160" height="120"></video>\n            <canvas id="face-canvas" width="160" height="120" class="face-canvas"></canvas>\n            <div id="face-status" class="face-status">Starting...</div>\n          </div>\n        </div>\n        <div class="cam-right">\n          <div class="cog-state-label">COGNITIVE STATE</div>\n          <div id="emo-state" class="emo-state">&#127919; Focused</div>\n          <div id="emo-sub" class="emo-sub">Camera loading...</div>\n          <div id="emo-bars" class="emo-bars"></div>\n          <div class="emo-model-tag">MediaPipe &#8226; FaceLandmarker &#8226; 478 Landmarks</div>\n        </div>\n        </div><!-- /cam-body -->\n      </div>\n      <div class="q-box" id="q-box">\n        <div class="q-concept" id="q-concept"></div>\n        <div class="q-text" id="q-text">Loading...</div>\n      </div>\n      <div id="options-wrap" class="options-grid"></div>\n      <div id="short-wrap" class="short-wrap hidden">\n        <textarea id="short-input" class="short-input" placeholder="Type your answer here..." rows="3"></textarea>\n        <button class="submit-short-btn" onclick="submitShort()">Submit &#8594;</button>\n      </div>\n      <div id="feedback-wrap" class="feedback-wrap hidden">\n        <div id="fb-msg" class="fb-msg"></div>\n        <div id="fb-exp" class="fb-exp"></div>\n        <div id="fb-followup" class="fb-followup hidden">\n          <span class="fq-label">&#128172; Follow-up:</span>\n          <span id="fq-text"></span>\n        </div>\n        \n        <div id="study-guide-wrap" class="study-guide-wrap hidden">\n          <div class="sg-header">📖 Study This Concept</div>\n          <div id="study-guide-text" class="sg-body"></div>\n        </div>\n        <div id="ml-explain" class="ml-explain"></div>\n        <button class="fb-btn next-btn" onclick="nextQuestion()">Next &#8594;</button>\n      </div>\n    </section>\n\n    <!-- DASHBOARD -->\n<section id="screen-dashboard" class="screen">\n      <div class="page-head">\n        <div class="tag green">Analytics</div>\n        <h1>Performance Dashboard</h1>\n        <p class="sub">Your progress, charts and topic mastery</p>\n      </div>\n      <div id="big-stats" class="big-stats"></div>\n      <div class="lp-card">\n        <div class="lp-top"><span id="lp-lbl">Level 1</span><span id="lp-xp" class="lp-xp">200 XP to Level 2</span></div>\n        <div class="lp-bar-wrap"><div id="lp-bar" class="lp-bar"></div></div>\n      </div>\n      <div class="charts-row">\n        <div class="chart-card half">\n          <h3 class="section-label">&#128200; Accuracy Over Time</h3>\n          <div class="chart-wrap"><canvas id="progress-chart"></canvas></div>\n        </div>\n        <div class="chart-card half">\n          <h3 class="section-label">&#127914; Correct vs Wrong</h3>\n          <div class="chart-wrap"><canvas id="pie-chart"></canvas></div>\n        </div>\n      </div>\n      <div class="chart-card">\n        <h3 class="section-label">&#127891; Topic Mastery %</h3>\n        <div class="chart-wrap tall"><canvas id="mastery-chart"></canvas></div>\n      </div>\n      <div class="chart-card">\n        <h3 class="section-label">&#9200; Topic Stats Table</h3>\n        <div id="topic-table"></div>\n      </div>\n      <div class="cal-card">\n        <h3 class="section-label">&#128197; Study Streak — Last 90 Days</h3>\n        <div id="streak-calendar" class="streak-calendar"></div>\n      </div>\n      <h3 class="section-label">Topics Studied</h3>\n      <div id="topics-list" class="topics-list"></div>\n      <h3 class="section-label">Weak Topics (below 50% accuracy)</h3>\n      <div id="weak-list" class="weak-list"></div>\n      <div class="cw-row">\n        <div class="cw green"><div id="d-correct" class="cw-v">0</div><div class="cw-l">&#9989; Correct</div></div>\n        <div class="cw red"><div id="d-wrong" class="cw-v">0</div><div class="cw-l">&#10060; Wrong</div></div>\n        <div class="cw purple"><div id="d-total" class="cw-v">0</div><div class="cw-l">&#128221; Total</div></div>\n      </div>\n      <h3 class="section-label">Badges Earned</h3>\n      <div id="d-badges" class="badge-list"><span class="nb">Complete quizzes to earn badges!</span></div>\n      <div class="export-row">\n        <button class="export-btn" onclick="exportReport()">&#128190; Export Report</button>\n        <button class="export-btn secondary" onclick="submitToLeaderboard()">&#127942; Submit to Leaderboard</button>\n      </div>\n    </section>\n\n    <!-- UPLOAD -->\n<section id="screen-upload" class="screen">\n      <div class="page-head">\n        <div class="tag teal">Study Material</div>\n        <h1>Learn From Your Material</h1>\n        <p class="sub">Upload a PDF or paste text — AI generates questions from your content only.</p>\n      </div>\n      <div class="upload-tabs">\n        <button class="tab-btn active" onclick="switchTab(\'text-tab\',\'pdf-tab\',this)">&#128203; Paste Text</button>\n        <button class="tab-btn" onclick="switchTab(\'pdf-tab\',\'text-tab\',this)">&#128196; Upload PDF</button>\n      </div>\n      <div id="text-tab" class="upload-card">\n        <textarea id="notes-input" class="notes-input" placeholder="Paste your notes, lecture content or textbook text here..."></textarea>\n        <div class="upload-controls">\n          <div class="control-group"><label>Difficulty</label>\n            <select id="upload-diff" class="sel">\n              <option value="easy">Easy</option><option value="medium" selected>Medium</option>\n              <option value="hard">Hard</option>\n            </select>\n          </div>\n          <div class="control-group"><label>Questions</label>\n            <select id="upload-numq" class="sel">\n              <option value="3">3</option><option value="5" selected>5</option><option value="10">10</option>\n            </select>\n          </div>\n          <button class="gen-btn" onclick="generateFromNotes()">&#9889; Generate</button>\n        </div>\n        <div id="upload-error" class="gen-error hidden"></div>\n      </div>\n      <div id="pdf-tab" class="upload-card hidden">\n        <div class="pdf-drop" id="pdf-drop" onclick="document.getElementById(\'pdf-file\').click()">\n          <div class="pdf-icon">&#128196;</div>\n          <div class="pdf-label">Click to upload PDF</div>\n          <div class="pdf-sub">Any PDF — textbooks, notes, research papers</div>\n          <div id="pdf-name" class="pdf-name"></div>\n        </div>\n        <input type="file" id="pdf-file" accept=".pdf" style="display:none" onchange="handlePDFSelect(this)"/>\n        <div class="upload-controls" style="margin-top:16px">\n          <div class="control-group"><label>Difficulty</label>\n            <select id="pdf-diff" class="sel">\n              <option value="easy">Easy</option><option value="medium" selected>Medium</option><option value="hard">Hard</option>\n            </select>\n          </div>\n          <div class="control-group"><label>Questions</label>\n            <select id="pdf-numq" class="sel">\n              <option value="3">3</option><option value="5" selected>5</option><option value="10">10</option>\n            </select>\n          </div>\n          <button class="gen-btn" id="pdf-btn" onclick="generateFromPDF()" disabled>&#128196; Generate</button>\n        </div>\n        <div id="pdf-error" class="gen-error hidden"></div>\n      </div>\n    </section>\n\n    <!-- LEADERBOARD -->\n<section id="screen-leaderboard" class="screen">\n      <div class="page-head">\n        <div class="tag gold">Rankings</div>\n        <h1>&#127942; Leaderboard</h1>\n        <p class="sub">Top students globally.</p>\n      </div>\n      <div id="lb-table" class="lb-table"><div class="lb-loading">Loading...</div></div>\n      <div class="lb-submit-card">\n        <h3>Submit Your Score</h3>\n        <div class="lb-row">\n          <input id="lb-name" class="topic-input" placeholder="Enter your name..." style="flex:1"/>\n          <button class="gen-btn" onclick="submitToLeaderboard()">&#127942; Submit</button>\n        </div>\n        <div id="lb-error" class="gen-error hidden"></div>\n      </div>\n    </section>\n\n    <!-- REVIEW -->\n<section id="screen-review" class="screen">\n      <div class="page-head">\n        <div class="tag orange">Spaced Repetition</div>\n        <h1>&#128337; Topics for Review</h1>\n        <p class="sub">SM-2 algorithm schedules review at the optimal time for maximum retention.</p>\n      </div>\n      <div id="review-list" class="review-list"></div>\n      <div id="review-empty" class="review-empty hidden">\n        <div class="re-icon">&#127881;</div>\n        <div class="re-msg">No topics yet!</div>\n        <div class="re-sub">Complete a quiz first, then come back here to review.</div>\n        <button class="gen-btn" style="margin-top:16px" onclick="showScreen(\'home\')">Start Studying &#8594;</button>\n      </div>\n    </section>\n\n\n    <!-- ══ MISTAKE NOTEBOOK ══ -->\n<section id="screen-mistakes" class="screen">\n      <div class="page-head">\n        <div class="tag" style="color:#ef4444">Mistakes</div>\n        <h1>&#128218; Mistake Notebook</h1>\n        <p class="sub">Every wrong answer saved — review, understand, retry.</p>\n      </div>\n      <div class="mn-toolbar">\n        <select id="mn-filter" class="sel" onchange="filterMistakes(this.value)">\n          <option value="">&#128218; All Topics</option>\n        </select>\n        <button class="mn-clear-btn" onclick="clearMistakes()">&#128465; Clear All</button>\n      </div>\n      <div id="mn-stats"></div>\n      <div id="mn-list"></div>\n    </section>\n\n    <!-- ══ EMOTION HISTORY ══ -->\n<section id="screen-emotion-history" class="screen">\n      <div class="page-head">\n        <div class="tag green">Cognitive Trends</div>\n        <h1>&#128200; Emotion History</h1>\n        <p class="sub">Your mental state across all quiz sessions.</p>\n      </div>\n      <div class="chart-card">\n        <h3 class="section-label">&#129504; Cognitive State — Last 20 Sessions</h3>\n        <div class="chart-wrap" style="height:260px"><canvas id="emotion-history-chart"></canvas></div>\n      </div>\n      <div id="eh-sessions" style="margin-top:14px"></div>\n    </section>\n\n    <!-- ══ AI TUTOR CHAT ══ -->\n<section id="screen-chat" class="screen">\n      <div class="page-head">\n        <div class="tag" style="color:#a78bfa">AI Tutor</div>\n        <h1>&#129302; Ask Your AI Tutor</h1>\n        <p class="sub">Ask anything — concepts, examples, practice questions.</p>\n      </div>\n      <div class="chat-topic-bar">\n        <span style="font-size:.8rem;font-weight:700;color:var(--sub)">&#128218; Topic:</span>\n        <input id="chat-topic-input" class="chat-topic-inp" placeholder="e.g. Machine Learning..."/>\n        <button class="chat-clear-btn" onclick="clearChat()">&#128465; New Chat</button>\n      </div>\n      <div id="chat-box" class="chat-box"></div>\n      <div class="chat-input-wrap">\n        <input id="chat-input" class="chat-input"\n               placeholder="Ask anything... e.g. explain K-Clustering simply"\n               autocomplete="off"/>\n        <button id="chat-send-btn" class="chat-send-btn" onclick="sendChatMessage()">Send &#9658;</button>\n      </div>\n      <div class="chat-suggestions">\n        <span style="font-size:.72rem;color:var(--sub);font-weight:600">Try:</span>\n        <button class="chat-sug" onclick="quickAsk(this)">&#128161; Explain simply</button>\n        <button class="chat-sug" onclick="quickAsk(this)">&#128221; Practice questions</button>\n        <button class="chat-sug" onclick="quickAsk(this)">&#10067; Why was I wrong?</button>\n        <button class="chat-sug" onclick="quickAsk(this)">&#127757; Real world example</button>\n      </div>\n    </section>\n\n  </main>\n</div>\n\n<div id="toast" class="toast hidden"></div>\n<div id="loading" class="loading-overlay hidden">\n  <div class="loader-box">\n    <div class="spinner"></div>\n    <div id="loading-msg" class="loading-msg">Generating questions with AI...</div>\n  </div>\n</div>\n<div id="cog-report-modal" class="cog-modal-overlay hidden">\n  <div class="cog-modal">\n    <div class="cog-modal-header">\n      <div class="cog-title">🧠 Cognitive Report</div>\n      <div class="cog-subtitle">Your mental state during this quiz</div>\n    </div>\n\n    <!-- Loading -->\n    <div id="cr-loading" class="cr-loading">\n      <div class="cr-spinner"></div>\n      <div>Analysing your session...</div>\n    </div>\n\n    <!-- Content -->\n    <div id="cr-content" class="hidden">\n\n      <!-- Dominant state + focus ring -->\n      <div class="cr-hero">\n        <div class="cr-ring-wrap">\n          <svg width="130" height="130" viewBox="0 0 130 130">\n            <circle cx="65" cy="65" r="54" fill="none" stroke="#1e1e2e" stroke-width="12"/>\n            <circle id="cr-ring-circle" cx="65" cy="65" r="54" fill="none"\n              stroke="#10b981" stroke-width="12"\n              stroke-linecap="round"\n              style="transform:rotate(-90deg);transform-origin:65px 65px;\n                     stroke-dasharray:339;stroke-dashoffset:339;\n                     transition:stroke-dashoffset 1s ease"/>\n          </svg>\n          <div class="cr-ring-text">\n            <div id="cr-ring-pct" class="cr-ring-pct">0%</div>\n            <div id="cr-ring-label" class="cr-ring-label">Focused</div>\n          </div>\n        </div>\n        <div class="cr-dom-info">\n          <div id="cr-dominant-emoji" class="cr-dom-emoji">🎯</div>\n          <div id="cr-dominant-label" class="cr-dom-label">Focused</div>\n          <div id="cr-total-q" class="cr-dom-sub">0 questions</div>\n        </div>\n      </div>\n\n      <!-- State bars -->\n      <div class="cr-section-title">Mental State Breakdown</div>\n      <div id="cr-state-bars" class="cr-state-bars"></div>\n\n      <!-- Weak topics study plan -->\n      <div class="cr-section-title">📚 Study Plan for Wrong Answers</div>\n      <div id="cr-study-plan" class="cr-study-plan"></div>\n\n      <!-- Tips + Recommendations -->\n      <div class="cr-section-title">⚡ Real-Time Tips for Your Session</div>\n      <div id="cr-tips-wrap" class="cr-tips-wrap"></div>\n\n      <!-- Actions -->\n      <div class="cr-actions">\n        <button class="cr-close-btn" onclick="closeCogReport()">Go to Dashboard</button>\n      </div>\n\n    </div><!-- /cr-content -->\n  </div>\n</div>\n\n<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>\n<script>\n  // Load MediaPipe Tasks Vision for face detection\n  (function(){\n    var s = document.createElement(\'script\');\n    s.src = \'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.js\';\n    s.crossOrigin = \'anonymous\';\n    s.onload = function(){ window._mpLoaded = true; console.log(\'MediaPipe loaded\'); };\n    s.onerror = function(){ console.warn(\'MediaPipe CDN failed\'); };\n    document.head.appendChild(s);\n  })();\n</script>\n<script>\n\n\n\n\n\n\n\'use strict\';\n\n// ════════════════════════════════════════════════════════════════\n//  AUTH LAYER  — runs before app initialisation\n//  Pure addition: no existing function is modified\n// ════════════════════════════════════════════════════════════════\n\nconst AUTH_TOKEN_KEY = \'em360_auth_token\';\nconst AUTH_USER_KEY  = \'em360_auth_user\';\n\nlet currentUser = null;  // { user_id, username, session_id }\n\n// ── Token helpers ────────────────────────────────────────────────\nfunction getToken()  { return localStorage.getItem(AUTH_TOKEN_KEY); }\nfunction saveToken(token, user) {\n  localStorage.setItem(AUTH_TOKEN_KEY, token);\n  localStorage.setItem(AUTH_USER_KEY,  JSON.stringify(user));\n  currentUser = user;\n}\nfunction clearToken() {\n  localStorage.removeItem(AUTH_TOKEN_KEY);\n  localStorage.removeItem(AUTH_USER_KEY);\n  currentUser = null;\n}\n\n// Returns Authorization header object — used by all fetch calls\nfunction authHeaders() {\n  const t = getToken();\n  return t ? { \'Authorization\': \'Bearer \' + t } : {};\n}\n\n// ── Patch the global fetch so all existing calls get the auth header\n//    without changing any existing function body\nconst _origFetch = window.fetch;\nwindow.fetch = function(url, opts) {\n  if (typeof url === \'string\' && url.startsWith(API)) {\n    opts = opts || {};\n    opts.headers = Object.assign({}, opts.headers || {}, authHeaders());\n  }\n  return _origFetch.call(this, url, opts);\n};\n\n// ── SESSION_ID override: logged-in users use their server session_id\n//    Guests keep the random s_xxxx id\nfunction getEffectiveSID() {\n  return currentUser ? currentUser.session_id : SESSION_ID;\n}\n\n// ── Show / hide layers ───────────────────────────────────────────\nfunction showApp() {\n  const overlay = document.getElementById(\'auth-overlay\');\n  if (overlay) overlay.classList.add(\'hidden\');\n  document.getElementById(\'app\').style.opacity = \'1\';\n  document.getElementById(\'app\').style.pointerEvents = \'\';\n  updateUserMenu();\n  updateWelcomeBanner();\n}\n\nfunction showAuthOverlay() {\n  const overlay = document.getElementById(\'auth-overlay\');\n  if (overlay) overlay.classList.remove(\'hidden\');\n  document.getElementById(\'app\').style.opacity = \'0\';\n  document.getElementById(\'app\').style.pointerEvents = \'none\';\n}\n\nfunction updateUserMenu() {\n  const avatarEl  = document.getElementById(\'user-avatar\');\n  const logoutEl  = document.getElementById(\'logout-btn\');\n  const levelPill = document.querySelector(\'.level-pill\');\n  if (!avatarEl) return;\n  if (currentUser) {\n    avatarEl.textContent  = currentUser.username[0].toUpperCase();\n    avatarEl.title        = currentUser.username;\n    avatarEl.style.display = \'flex\';\n    if (logoutEl)  logoutEl.style.display = \'flex\';\n    if (levelPill) levelPill.style.display = \'none\';\n    const cpwBtn = document.getElementById(\'change-pw-btn\');\n    if (cpwBtn) cpwBtn.style.display = \'flex\';\n  } else {\n    avatarEl.style.display  = \'none\';\n    if (logoutEl) logoutEl.style.display = \'none\';\n    if (levelPill) levelPill.style.display = \'flex\';\n    const cpwBtn2 = document.getElementById(\'change-pw-btn\');\n    if (cpwBtn2) cpwBtn2.style.display = \'none\';\n  }\n}\n\nfunction updateWelcomeBanner() {\n  const banner = document.getElementById(\'welcome-banner\');\n  const nameEl = document.getElementById(\'welcome-name\');\n  if (!banner || !nameEl) return;\n  if (currentUser) {\n    nameEl.textContent = currentUser.username;\n    banner.classList.remove(\'hidden\');\n  } else {\n    banner.classList.add(\'hidden\');\n  }\n}\n\n// ── Auth check on load ───────────────────────────────────────────\nasync function authCheck() {\n  const token = getToken();\n  if (!token) { showAuthOverlay(); return; }\n\n  try {\n    const res  = await _origFetch(API + \'/api/auth/me\', {\n      headers: { \'Authorization\': \'Bearer \' + token }\n    });\n    if (!res.ok) throw new Error(\'invalid\');\n    const data = await res.json();\n    currentUser = { user_id: data.user_id, username: data.username,\n                    session_id: data.session_id };\n    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(currentUser));\n    showApp();\n  } catch(e) {\n    clearToken();\n    showAuthOverlay();\n  }\n}\n\n// ── Auth tab switching ───────────────────────────────────────────\nfunction switchAuthTab(tab) {\n  // Hide all three forms using only inline styles — avoids !important class conflicts\n  [\'login\',\'register\',\'forgot\'].forEach(function(t) {\n    const f = document.getElementById(\'auth-form-\' + t);\n    if (f) f.style.display = \'none\';\n    const btn = document.querySelector(\'[data-auth-tab="\' + t + \'"]\');\n    if (btn) btn.classList.remove(\'active\');\n  });\n\n  // Show the target form\n  const targetForm = document.getElementById(\'auth-form-\' + tab);\n  if (targetForm) {\n    targetForm.style.display = \'flex\';\n    targetForm.style.flexDirection = \'column\';\n    targetForm.style.gap = \'14px\';\n    targetForm.classList.remove(\'hidden\');\n  }\n\n  // Activate the tab button\n  const activeBtn = document.querySelector(\'[data-auth-tab="\' + tab + \'"]\');\n  if (activeBtn) activeBtn.classList.add(\'active\');\n\n  // If switching to forgot: ensure step-1 is shown, step-2 hidden\n  if (tab === \'forgot\') {\n    const s1 = document.getElementById(\'forgot-step-1\');\n    const s2 = document.getElementById(\'forgot-step-2\');\n    if (s1) { s1.style.display = \'flex\'; s1.classList.add(\'active\'); }\n    if (s2) { s2.style.display = \'none\'; s2.classList.remove(\'active\'); }\n    // Also clear forgot email field and reset to step 1\n    const emailEl = document.getElementById(\'forgot-email\');\n    if (emailEl) emailEl.value = \'\';\n  }\n\n  // Clear error\n  const errEl = document.getElementById(\'auth-error\');\n  if (errEl) { errEl.classList.remove(\'show\'); errEl.textContent = \'\'; }\n}\n\n// ── Password show/hide ───────────────────────────────────────────\nfunction togglePwVisibility(inputId, btn) {\n  const inp = document.getElementById(inputId);\n  if (!inp) return;\n  if (inp.type === \'password\') {\n    inp.type = \'text\';     btn.textContent = \'Hide\';\n  } else {\n    inp.type = \'password\'; btn.textContent = \'Show\';\n  }\n}\n\n// ── Login ────────────────────────────────────────────────────────\nasync function authLogin() {\n  const email    = document.getElementById(\'login-email\').value.trim();\n  const password = document.getElementById(\'login-password\').value;\n  const errEl    = document.getElementById(\'auth-error\');\n  const btn      = document.getElementById(\'auth-login-btn\');\n\n  errEl.classList.remove(\'show\');\n  if (!email || !password) {\n    errEl.textContent = \'Please enter your email and password.\';\n    errEl.classList.add(\'show\'); return;\n  }\n\n  btn.disabled = true;\n  btn.innerHTML = \'<span class="auth-btn-spinner"></span>Signing in...\';\n\n  try {\n    const res  = await _origFetch(API + \'/api/auth/login\', {\n      method:  \'POST\',\n      headers: { \'Content-Type\': \'application/json\' },\n      body:    JSON.stringify({ email, password })\n    });\n    const data = await res.json();\n    if (!res.ok) throw new Error(data.error || \'Login failed\');\n    saveToken(data.token, { user_id: data.user_id, username: data.username,\n                             session_id: data.session_id });\n    showApp();\n    // Load stats for this user\n    if (typeof updateHomeStats === \'function\') {\n      const sr = await _origFetch(API + \'/api/stats/\' + data.session_id, {\n        headers: { \'Authorization\': \'Bearer \' + data.token }\n      });\n      if (sr.ok) {\n        const sd = await sr.json();\n        if (typeof state !== \'undefined\') {\n          state.score = sd.score || 0; state.xp = sd.xp || 0;\n          state.level = sd.level || 1; state.streak = sd.streak || 0;\n          state.totalQ = sd.total_q || 0; state.correctQ = sd.correct_q || 0;\n          state.topics_studied = sd.topics_studied || [];\n          state.study_dates    = sd.study_dates    || [];\n          state.badges         = sd.badges         || [];\n        }\n        updateHomeStats();\n      }\n    }\n  } catch(e) {\n    errEl.textContent = e.message;\n    errEl.classList.add(\'show\');\n  }\n\n  btn.disabled = false;\n  btn.textContent = \'Sign In\';\n}\n\n// ── Register ─────────────────────────────────────────────────────\nasync function authRegister() {\n  const username  = document.getElementById(\'reg-username\').value.trim();\n  const email     = document.getElementById(\'reg-email\').value.trim();\n  const password  = document.getElementById(\'reg-password\').value;\n  const confirm   = document.getElementById(\'reg-confirm\').value;\n  const errEl     = document.getElementById(\'auth-error\');\n  const btn       = document.getElementById(\'auth-register-btn\');\n\n  errEl.classList.remove(\'show\');\n\n  if (!username || !email || !password || !confirm) {\n    errEl.textContent = \'All fields are required.\';\n    errEl.classList.add(\'show\'); return;\n  }\n  if (password !== confirm) {\n    errEl.textContent = \'Passwords do not match.\';\n    errEl.classList.add(\'show\'); return;\n  }\n  if (password.length < 6) {\n    errEl.textContent = \'Password must be at least 6 characters.\';\n    errEl.classList.add(\'show\'); return;\n  }\n\n  btn.disabled = true;\n  btn.innerHTML = \'<span class="auth-btn-spinner"></span>Creating account...\';\n\n  try {\n    const res  = await _origFetch(API + \'/api/auth/register\', {\n      method:  \'POST\',\n      headers: { \'Content-Type\': \'application/json\' },\n      body:    JSON.stringify({ username, email, password })\n    });\n    const data = await res.json();\n    if (!res.ok) throw new Error(data.error || \'Registration failed\');\n    saveToken(data.token, { user_id: data.user_id, username: data.username,\n                             session_id: data.session_id });\n    showApp();\n    if (typeof updateHomeStats === \'function\') updateHomeStats();\n  } catch(e) {\n    errEl.textContent = e.message;\n    errEl.classList.add(\'show\');\n  }\n\n  btn.disabled = false;\n  btn.textContent = \'Create Account\';\n}\n\n// ── Logout ───────────────────────────────────────────────────────\nfunction logout() {\n  clearToken();\n  // Clear charts\n  if (typeof charts !== \'undefined\') {\n    Object.keys(charts).forEach(k => { if (charts[k]) { charts[k].destroy(); charts[k] = null; } });\n  }\n  // Reset state\n  if (typeof state !== \'undefined\') {\n    state.questions=[]; state.qIdx=0; state.score=0; state.xp=0;\n    state.level=1; state.streak=0; state.totalQ=0; state.correctQ=0;\n    state.topics_studied=[]; state.study_dates=[]; state.badges=[];\n  }\n  if (typeof updateHomeStats === \'function\') updateHomeStats();\n  showAuthOverlay();\n  switchAuthTab(\'login\');\n}\n\n// ── Enter key on auth forms ──────────────────────────────────────\ndocument.addEventListener(\'keydown\', function(e) {\n  if (e.key !== \'Enter\') return;\n  const overlay = document.getElementById(\'auth-overlay\');\n  if (!overlay || overlay.classList.contains(\'hidden\')) return;\n  const loginForm = document.getElementById(\'auth-form-login\');\n  if (loginForm && !loginForm.classList.contains(\'hidden\')) authLogin();\n  else authRegister();\n});\n\n\n// ═══════════════════════════════════════════════════════\n//  EduMate 360 v5.3 — Clean rewrite, zero crashes\n//  Features: AI Quiz, ML Adaptive, Charts, Spaced Rep,\n//            Leaderboard, Badges, PDF, Voice\n// ═══════════════════════════════════════════════════════\n\nconst API = window.location.origin;\n// Unique session per browser tab — stays consistent across quiz\nconst SESSION_ID = localStorage.getItem(\'em360_sid\') || (function(){\n  const id = \'s_\' + Math.random().toString(36).slice(2,10);\n  try { localStorage.setItem(\'em360_sid\', id); } catch(e){}\n  return id;\n})();\n\n// ── CHART INSTANCES (kept so we can destroy before redraw)\nconst charts = { progress: null, pie: null, mastery: null };\n\n// ── APP STATE\nconst state = {\n  questions: [], qIdx: 0,\n  topic: \'\', difficulty: \'medium\',\n  score: 0, xp: 0, level: 1,\n  streak: 0, totalQ: 0, correctQ: 0,\n  timerStart: 0, timerInterval: null,\n  voiceEnabled: false,\n  cogStateRaw: \'focused\',\n  topics_studied: [], study_dates: [], badges: []\n};\n\n// ── HELPERS\nfunction $(id){ return document.getElementById(id); }\n\nfunction showToast(msg, dur) {\n  const t = $(\'toast\');\n  if (!t) return;\n  t.textContent = msg;\n  t.classList.remove(\'hidden\');\n  clearTimeout(t._timer);\n  t._timer = setTimeout(() => t.classList.add(\'hidden\'), dur || 3500);\n}\n\nfunction setLoading(on, msg) {\n  const el = $(\'loading\');\n  if (!el) return;\n  if (msg) $(\'loading-msg\').textContent = msg;\n  el.classList.toggle(\'hidden\', !on);\n}\n\nfunction showScreen(name) {\n  document.querySelectorAll(\'.screen\').forEach(s => s.classList.remove(\'active\'));\n  document.querySelectorAll(\'.nav-btn\').forEach(b => b.classList.remove(\'active\'));\n  const screen = $(\'screen-\' + name);\n  if (screen) screen.classList.add(\'active\');\n  const btn = document.querySelector(\'[data-screen="\' + name + \'"]\');\n  if (btn) btn.classList.add(\'active\');\n  if (name === \'dashboard\') renderDashboard();\n  if (name === \'leaderboard\') loadLeaderboard();\n  if (name === \'review\') loadReview();\n  if (name === \'mistakes\') loadMistakes(\'\');\n  if (name === \'emotion-history\') loadEmotionHistory();\n  if (name === \'chat\') initChat();\n}\n\nfunction switchTab(showId, hideId, btn) {\n  $(showId).classList.remove(\'hidden\');\n  $(hideId).classList.add(\'hidden\');\n  document.querySelectorAll(\'.tab-btn\').forEach(b => b.classList.remove(\'active\'));\n  btn.classList.add(\'active\');\n}\n\nfunction calcDayStreak(dates) {\n  if (!dates || !dates.length) return 0;\n  const sorted = [...new Set(dates)].sort().reverse();\n  let streak = 0;\n  let cur = new Date(); cur.setHours(0,0,0,0);\n  for (const d of sorted) {\n    const dd = new Date(d); dd.setHours(0,0,0,0);\n    const diff = (cur - dd) / 86400000;\n    if (diff <= 1) { streak++; cur = dd; }\n    else break;\n  }\n  return streak;\n}\n\n// ── UPDATE HOME STATS BAR\nfunction updateHomeStats() {\n  $(\'hs-streak\').textContent = state.streak;\n  $(\'hs-score\').textContent = state.score;\n  $(\'hs-acc\').textContent = state.totalQ > 0\n    ? Math.round(state.correctQ / state.totalQ * 100) + \'%\' : \'0%\';\n  $(\'hs-topics\').textContent = state.topics_studied.length;\n  $(\'hs-days\').textContent = calcDayStreak(state.study_dates);\n  $(\'xp-num\').textContent = state.xp;\n  $(\'lv-num\').textContent = state.level;\n  const xpInLevel = state.xp % 200;\n  $(\'xp-bar\').style.width = (xpInLevel / 200 * 100) + \'%\';\n  $(\'xp-to\').textContent = (200 - xpInLevel) + \' XP to Level \' + (state.level + 1);\n  if (state.badges && state.badges.length > 0) {\n    $(\'home-badges\').classList.remove(\'hidden\');\n    $(\'home-badge-list\').innerHTML = state.badges\n      .map(b => \'<span class="badge">\' + b + \'</span>\').join(\'\');\n  }\n}\n\n// ── CHECK DUE TOPICS ON HOME LOAD\nasync function checkDueTopics() {\n  try {\n    const res = await fetch(API + \'/api/due-topics/\' + getEffectiveSID());\n    if (!res.ok) return;\n    const data = await res.json();\n    if (data.due && data.due.length > 0) {\n      const a = $(\'due-alert\');\n      a.innerHTML = \'&#9201; <strong>\' + data.due.length + \' topic(s) due for review:</strong> \'\n        + data.due.map(d => d.topic).join(\', \')\n        + \' <button onclick="showScreen(\\\'review\\\')" class="due-btn">Review Now</button>\';\n      a.classList.remove(\'hidden\');\n    }\n  } catch(e) { /* silent */ }\n}\n\n// ── INIT\ndocument.addEventListener(\'DOMContentLoaded\', function() {\n  authCheck();\n  // Wire nav buttons\n  document.querySelectorAll(\'.nav-btn\').forEach(function(btn) {\n    btn.addEventListener(\'click\', function() {\n      showScreen(btn.getAttribute(\'data-screen\'));\n    });\n  });\n  // Enter key on topic input\n  var topicInput = $(\'topic-input\');\n  if (topicInput) {\n    topicInput.addEventListener(\'keydown\', function(e) {\n      if (e.key === \'Enter\') generateQuestions();\n    });\n  }\n  updateHomeStats();\n  checkDueTopics();\n});\n\n\n// ═══════════════════════════════════════════════════════\n//  QUESTION GENERATION\n// ═══════════════════════════════════════════════════════\n\nasync function generateQuestions() {\n  const topicEl = $(\'topic-input\');\n  const topic = topicEl ? topicEl.value.trim() : \'\';\n  const diff = $(\'diff-select\').value;\n  const numQ = parseInt($(\'numq-select\').value) || 5;\n  const qType = $(\'type-select\').value;\n\n  if (!topic) { showToast(\'Please enter a topic first!\'); topicEl && topicEl.focus(); return; }\n\n  const errEl = $(\'gen-error\');\n  errEl.classList.add(\'hidden\');\n  $(\'generate-btn\').disabled = true;\n  setLoading(true, \'AI generating \' + numQ + \' \' + qType + \' questions on "\' + topic + \'"...\');\n\n  try {\n    const res = await fetch(API + \'/api/generate\', {\n      method: \'POST\',\n      headers: { \'Content-Type\': \'application/json\' },\n      body: JSON.stringify({\n        topic, difficulty: diff, num_questions: numQ,\n        type: qType, session_id: getEffectiveSID()\n      })\n    });\n    const data = await res.json();\n    if (!res.ok || !data.questions || !data.questions.length) {\n      throw new Error(data.error || \'No questions returned. Check your Groq API key on Render.\');\n    }\n    state.questions = data.questions;\n    state.topic = topic;\n    state.difficulty = diff;\n    state.qIdx = 0;\n  allQuizEmotions = [];  // reset for new quiz\n    if (!state.topics_studied.includes(topic)) state.topics_studied.push(topic);\n    setLoading(false);\n    $(\'generate-btn\').disabled = false;\n    showScreen(\'quiz\');\n    renderQuestion();\n  } catch(err) {\n    setLoading(false);\n    $(\'generate-btn\').disabled = false;\n    errEl.textContent = \'❌ \' + err.message;\n    errEl.classList.remove(\'hidden\');\n  }\n}\n\nasync function generateFromNotes() {\n  const text = $(\'notes-input\').value.trim();\n  const diff = $(\'upload-diff\').value;\n  const numQ = parseInt($(\'upload-numq\').value) || 5;\n  const errEl = $(\'upload-error\');\n  errEl.classList.add(\'hidden\');\n  if (text.length < 50) { showToast(\'Please paste at least 50 characters of text\'); return; }\n  setLoading(true, \'Reading your notes and generating questions...\');\n  try {\n    const res = await fetch(API + \'/api/generate-from-text\', {\n      method: \'POST\',\n      headers: { \'Content-Type\': \'application/json\' },\n      body: JSON.stringify({ text, difficulty: diff, num_questions: numQ, session_id: getEffectiveSID() })\n    });\n    const data = await res.json();\n    if (!res.ok || !data.questions || !data.questions.length) throw new Error(data.error || \'Failed\');\n    state.questions = data.questions;\n    state.topic = data.topic || \'Notes\';\n    state.difficulty = diff;\n    state.qIdx = 0;\n  allQuizEmotions = [];  // reset for new quiz\n    setLoading(false);\n    showScreen(\'quiz\');\n    renderQuestion();\n  } catch(err) {\n    setLoading(false);\n    errEl.textContent = \'❌ \' + err.message;\n    errEl.classList.remove(\'hidden\');\n  }\n}\n\nfunction handlePDFSelect(input) {\n  const file = input.files[0];\n  if (file) {\n    $(\'pdf-name\').textContent = \'📄 \' + file.name;\n    $(\'pdf-btn\').disabled = false;\n  }\n}\n\nasync function generateFromPDF() {\n  const file = $(\'pdf-file\').files[0];\n  if (!file) { showToast(\'Please select a PDF file\'); return; }\n  const diff = $(\'pdf-diff\').value;\n  const numQ = parseInt($(\'pdf-numq\').value) || 5;\n  const errEl = $(\'pdf-error\');\n  errEl.classList.add(\'hidden\');\n  setLoading(true, \'Reading PDF and generating questions...\');\n  try {\n    const fd = new FormData();\n    fd.append(\'pdf\', file);\n    fd.append(\'difficulty\', diff);\n    fd.append(\'num_questions\', numQ);\n    fd.append(\'session_id\', SESSION_ID);\n    const res = await fetch(API + \'/api/upload-pdf\', { method: \'POST\', body: fd });\n    const data = await res.json();\n    if (!res.ok || !data.questions || !data.questions.length) throw new Error(data.error || \'Failed\');\n    showToast(\'Read \' + (data.pages_read || \'?\') + \' pages → \' + data.total + \' questions!\');\n    state.questions = data.questions;\n    state.topic = data.topic || \'PDF\';\n    state.difficulty = diff;\n    state.qIdx = 0;\n  allQuizEmotions = [];  // reset for new quiz\n    setLoading(false);\n    showScreen(\'quiz\');\n    renderQuestion();\n  } catch(err) {\n    setLoading(false);\n    errEl.textContent = \'❌ \' + err.message;\n    errEl.classList.remove(\'hidden\');\n  }\n}\n\n\n// ═══════════════════════════════════════════════════════\n//  QUIZ RENDERING\n// ═══════════════════════════════════════════════════════\n\nfunction renderQuestion() {\n  const q = state.questions[state.qIdx];\n  if (!q) return;\n  const tot = state.questions.length;\n\n  // Header\n  $(\'q-topic-badge\').textContent = state.topic;\n  $(\'q-diff-badge\').textContent = state.difficulty;\n  $(\'q-diff-badge\').className = \'q-diff-badge diff-\' + state.difficulty;\n  $(\'q-num\').textContent = \'Q \' + (state.qIdx + 1) + \'/\' + tot;\n  $(\'qprog\').style.width = ((state.qIdx + 1) / tot * 100) + \'%\';\n\n  // Question\n  $(\'q-concept\').textContent = q.concept || \'\';\n  $(\'q-text\').textContent = q.question;\n\n  // Bg colour by difficulty\n  const bg = {\n    beginner:\'rgba(14,165,233,0.08)\', easy:\'rgba(16,185,129,0.06)\',\n    medium:\'rgba(245,158,11,0.07)\',   hard:\'rgba(239,68,68,0.07)\',\n    expert:\'rgba(139,92,246,0.09)\'\n  };\n  $(\'q-box\').style.background = bg[state.difficulty] || bg.medium;\n\n  // Live strip\n  $(\'ls-score\').textContent = state.score;\n  $(\'ls-streak\').textContent = state.streak;\n  $(\'ls-acc\').textContent = state.totalQ > 0\n    ? Math.round(state.correctQ / state.totalQ * 100) + \'%\' : \'0%\';\n  $(\'ml-tag\').textContent = \'ML: \' + state.difficulty;\n\n  // Hide feedback, reset timer\n  $(\'feedback-wrap\').classList.add(\'hidden\');\n  $(\'options-wrap\').classList.add(\'hidden\');\n  $(\'short-wrap\').classList.add(\'hidden\');\n  startTimer();\n\n  // Voice\n  if (state.voiceEnabled) speak(q.question);\n\n  // Render inputs\n  const type = q.type || \'mcq\';\n  if (type === \'short\') {\n    $(\'short-input\').value = \'\';\n    $(\'short-wrap\').classList.remove(\'hidden\');\n  } else {\n    const opts = q.options || [];\n    const letters = [\'A\',\'B\',\'C\',\'D\',\'E\',\'F\'];\n    $(\'options-wrap\').innerHTML = opts.map(function(opt, i) {\n      return \'<button class="opt-btn" onclick="handleMCQ(this,\' + i + \')">\'\n        + \'<span class="opt-letter">\' + (letters[i]||String(i+1)) + \'</span>\'\n        + escHtml(opt) + \'</button>\';\n    }).join(\'\');\n    $(\'options-wrap\').classList.remove(\'hidden\');\n  }\n}\n\nfunction escHtml(s) {\n  return String(s).replace(/&/g,\'&amp;\').replace(/</g,\'&lt;\').replace(/>/g,\'&gt;\').replace(/"/g,\'&quot;\');\n}\n\nfunction startTimer() {\n  clearInterval(state.timerInterval);\n  state.timerStart = Date.now();\n  state.timerInterval = setInterval(function() {\n    const secs = Math.round((Date.now() - state.timerStart) / 1000);\n    $(\'timer-badge\').textContent = \'⏱ \' + secs + \'s\';\n  }, 1000);\n}\n\nfunction toggleVoice() {\n  state.voiceEnabled = !state.voiceEnabled;\n  $(\'voice-btn\').textContent = state.voiceEnabled ? \'🔊 On\' : \'🔇 Voice\';\n  $(\'voice-btn\').classList.toggle(\'active\', state.voiceEnabled);\n  if (state.voiceEnabled && state.questions[state.qIdx]) {\n    speak(state.questions[state.qIdx].question);\n  }\n}\n\nfunction speak(text) {\n  if (!window.speechSynthesis) return;\n  window.speechSynthesis.cancel();\n  const u = new SpeechSynthesisUtterance(text);\n  u.rate = 0.9;\n  window.speechSynthesis.speak(u);\n}\n\nasync function handleMCQ(btn, optIdx) {\n  clearInterval(state.timerInterval);\n  const timeSec = (Date.now() - state.timerStart) / 1000;\n  const q = state.questions[state.qIdx];\n  const opts = q.options || [];\n  const selected = opts[optIdx];\n  const correct = selected === q.answer;\n\n  // Disable all options, highlight correct/wrong\n  document.querySelectorAll(\'.opt-btn\').forEach(function(b, i) {\n    b.disabled = true;\n    if (opts[i] === q.answer) b.classList.add(\'correct-opt\');\n    else if (b === btn && !correct) b.classList.add(\'wrong-opt\');\n  });\n\n  await submitAnswer(correct, selected, q, timeSec);\n}\n\nasync function submitShort() {\n  clearInterval(state.timerInterval);\n  const timeSec = (Date.now() - state.timerStart) / 1000;\n  const q = state.questions[state.qIdx];\n  const given = ($(\'short-input\').value || \'\').trim().toLowerCase();\n  const correct_lower = (q.answer || \'\').toLowerCase();\n  // Match if answer contains key words from correct answer\n  const keywords = correct_lower.split(/\\s+/).filter(w => w.length > 4);\n  // Require majority of keywords to match (not just any one) — prevents easy guessing\n  const matchCount = keywords.filter(k => given.includes(k)).length;\n  const correct = keywords.length > 0\n    ? (keywords.length <= 2 ? matchCount >= 1 : matchCount >= Math.ceil(keywords.length * 0.6))\n    : given.includes(correct_lower.slice(0, 8));\n  await submitAnswer(correct, given, q, timeSec);\n}\n\nasync function submitAnswer(correct, studentAns, q, timeSec) {\n  // Update local state immediately (optimistic)\n  state.totalQ++;\n  state.timerStart = state.timerStart; // keep reference\n  if (correct) {\n    state.correctQ++;\n    state.streak++;\n  } else {\n    state.streak = 0;\n  }\n\n  // Show feedback right away\n  const fm = $(\'fb-msg\');\n  fm.textContent = correct ? \'✅ Correct!\' : \'❌ Wrong — Answer: \' + q.answer;\n  fm.className = \'fb-msg \' + (correct ? \'correct-fb\' : \'wrong-fb\');\n  $(\'fb-exp\').innerHTML = \'<strong style="color:#a78bfa">Explanation:</strong> \'\n    + escHtml(q.explanation || \'\');\n  $(\'fb-followup\').classList.add(\'hidden\');\n  $(\'ml-explain\').textContent = \'\';\n  $(\'feedback-wrap\').classList.remove(\'hidden\');\n\n  // Update live strip immediately\n  $(\'ls-score\').textContent = state.score;\n  $(\'ls-streak\').textContent = state.streak;\n  $(\'ls-acc\').textContent = Math.round(state.correctQ / state.totalQ * 100) + \'%\';\n\n  // Submit to backend\n  try {\n    const res = await fetch(API + \'/api/submit\', {\n      method: \'POST\',\n      headers: { \'Content-Type\': \'application/json\' },\n      body: JSON.stringify({\n        session_id: getEffectiveSID(),\n        correct: correct,\n        time_sec: timeSec,\n        topic: state.topic,\n        concept: q.concept || \'\',\n        q_idx: state.qIdx,\n        cog_state: camStream ? (state.cogStateRaw || state.emotion || \'focused\') : \'nocam\',\n        question: q.question,\n        student_answer: studentAns,\n        correct_answer: q.answer,\n        explanation: q.explanation || \'\'\n      })\n    });\n    if (!res.ok) return;\n    const data = await res.json();\n\n    // Sync state from server response\n    state.score    = data.stats.score;\n    state.xp       = data.stats.xp;\n    state.level    = data.stats.level;\n    state.streak   = data.stats.streak;\n    state.difficulty = data.new_difficulty;\n    if (data.stats.badges) state.badges = data.stats.badges;\n\n    // Update live strip with server values\n    $(\'ls-score\').textContent = state.score;\n    $(\'ls-streak\').textContent = state.streak;\n    $(\'ls-acc\').textContent = data.stats.accuracy + \'%\';\n    $(\'ml-tag\').textContent = \'ML: \' + data.new_difficulty;\n\n    // Follow-up question\n    if (data.followup) {\n      $(\'fq-text\').textContent = data.followup;\n      $(\'fb-followup\').classList.remove(\'hidden\');\n    }\n\n    // ML explanation\n    $(\'ml-explain\').textContent = \'🤖 \' + data.ml_reason\n      + \' (confidence: \' + Math.round((data.ml_confidence || 0) * 100) + \'%)\';\n\n    // Badges\n    if (data.new_badges && data.new_badges.length > 0) {\n      data.new_badges.forEach(function(b) { showToast(\'🏅 New badge: \' + b, 4000); });\n    }\n\n    // Difficulty change toast\n    if (data.ml_action === \'harder\') showToast(\'📈 Difficulty increased → \' + data.new_difficulty);\n    else if (data.ml_action === \'easier\') showToast(\'📉 Difficulty reduced → \' + data.new_difficulty);\n\n    // Study guide — shown only when answer was wrong\n    if (!correct && data.topic_guide) {\n      const sg = $(\'study-guide-wrap\');\n      const st = $(\'study-guide-text\');\n      if (sg && st) {\n        // Format bullet points\n        const lines = data.topic_guide.split(\'\\n\').filter(l => l.trim());\n        st.innerHTML = lines.map(l => \'<div class="sg-line">\' + escHtml(l) + \'</div>\').join(\'\');\n        sg.classList.remove(\'hidden\');\n      }\n    } else {\n      const sg = $(\'study-guide-wrap\');\n      if (sg) sg.classList.add(\'hidden\');\n    }\n\n    // Accumulate emotion for cognitive report\n    allQuizEmotions.push({ emotion: state.cogStateRaw || state.emotion, correct: correct,\n      topic: state.topic, concept: q.concept || \'\', qIdx: state.qIdx });\n\n    updateHomeStats();\n  } catch(e) {\n    // Backend error — local state is still correct, just no server sync\n    console.warn(\'Submit error:\', e.message);\n  }\n}\n\nfunction nextQuestion() {\n  state.qIdx++;\n  if (state.qIdx >= state.questions.length) {\n    // Quiz complete — show cognitive report first\n    showToast(\'🎉 Quiz complete! Great work!\', 3000);\n    updateHomeStats();\n    setTimeout(function() { showCognitiveReport(); }, 800);\n    return;\n  }\n  renderQuestion();\n}\n\n\n\n// ═══════════════════════════════════════════════════════\n//  COGNITIVE REPORT MODAL\n// ═══════════════════════════════════════════════════════\n// ── shuffle helper for tips randomisation\nfunction shuffle(arr) {\n  const a = [...arr];\n  for (let i = a.length - 1; i > 0; i--) {\n    const j = Math.floor(Math.random() * (i + 1));\n    [a[i], a[j]] = [a[j], a[i]];\n  }\n  return a;\n}\n\nasync function showCognitiveReport() {\n  const modal = $(\'cog-report-modal\');\n  if (!modal) { showScreen(\'dashboard\'); return; }\n  modal.classList.remove(\'hidden\');\n\n  // Show loading state\n  $(\'cr-loading\').classList.remove(\'hidden\');\n  $(\'cr-content\').classList.add(\'hidden\');\n\n  try {\n    const res = await fetch(API + \'/api/cognitive-report/\' + getEffectiveSID());\n    const d   = await res.json();\n\n    // ── Header ───────────────────────────────────────────\n    const meta = {\n      focused:    { emoji:\'🎯\', label:\'Focused\',    color:\'#10b981\' },\n      loading:    { emoji:\'🤔\', label:\'Processing\', color:\'#f59e0b\' },\n      overloaded: { emoji:\'🧩\', label:\'Overloaded\', color:\'#f97316\' },\n      fatigued:   { emoji:\'😴\', label:\'Fatigued\',   color:\'#94a3b8\' },\n      anxious:    { emoji:\'🧘\', label:\'Anxious\',    color:\'#ef4444\' }\n    };\n    const dom = meta[d.dominant] || meta.focused;\n    $(\'cr-dominant-emoji\').textContent = dom.emoji;\n    $(\'cr-dominant-label\').textContent = dom.label;\n    $(\'cr-dominant-label\').style.color = dom.color;\n    $(\'cr-total-q\').textContent = d.total_questions + \' questions answered\';\n\n    // ── Focus ring ───────────────────────────────────────\n    const camUsed = d.cam_used === true;\n    const pct = camUsed ? (d.focus_pct || 0) : 0;\n    const circ = 2 * Math.PI * 54; // r=54\n    $(\'cr-ring-circle\').style.strokeDasharray  = circ;\n    $(\'cr-ring-circle\').style.strokeDashoffset = camUsed ? circ - (pct / 100 * circ) : circ;\n    $(\'cr-ring-circle\').style.stroke = camUsed ? \'#10b981\' : \'#374151\';\n    $(\'cr-ring-pct\').textContent  = camUsed ? pct + \'%\' : \'—\';\n    $(\'cr-ring-label\').textContent = camUsed ? \'Focused\' : \'No Cam\';\n    $(\'cr-ring-pct\').style.color  = camUsed ? \'#10b981\' : \'#6b7280\';\n\n    // Dominant state — camera off override\n    if (!camUsed) {\n      $(\'cr-dominant-emoji\').textContent = \'📷\';\n      $(\'cr-dominant-label\').textContent = \'Camera Off\';\n      $(\'cr-dominant-label\').style.color = \'#6b7280\';\n    }\n\n    // ── State breakdown bars ─────────────────────────────\n    const states = d.states || {};\n    const barColors = { focused:\'#10b981\',loading:\'#f59e0b\',overloaded:\'#f97316\',fatigued:\'#94a3b8\',anxious:\'#ef4444\' };\n    if (!camUsed) {\n      $(\'cr-state-bars\').innerHTML = \'<div class="cr-nocam-msg">📷 Camera was off — no cognitive data.<br>Enable camera next quiz for full mental state analysis!</div>\';\n    } else {\n      $(\'cr-state-bars\').innerHTML = Object.entries(states).map(([k,v]) =>\n        \'<div class="cr-bar-row">\'\n        + \'<span class="cr-bar-name">\' + (meta[k]||{emoji:\'\',label:k}).emoji + \' \' + (meta[k]||{label:k}).label + \'</span>\'\n        + \'<div class="cr-bar-track"><div class="cr-bar-fill" style="width:\'+v+\'%;background:\'+barColors[k]+\'"></div></div>\'\n        + \'<span class="cr-bar-pct">\'+v+\'%</span>\'\n        + \'</div>\'\n      ).join(\'\');\n    }\n\n    // ── Weak topics study plan ───────────────────────────\n    const pt = d.per_topic || {};\n    const topicKeys = Object.keys(pt);\n    const studyEl = $(\'cr-study-plan\');\n    if (topicKeys.length === 0) {\n      studyEl.innerHTML = \'<div class="cr-no-weak">🏆 No weak areas — perfect score!</div>\';\n    } else {\n      studyEl.innerHTML = topicKeys.map(topic => {\n        const concepts = pt[topic];\n        return \'<div class="cr-topic-card">\'\n          + \'<div class="cr-topic-title">📚 \' + escHtml(topic) + \'</div>\'\n          + concepts.map(c => \'<div class="cr-concept-row">⚠️ Focus on: <strong>\' + escHtml(c) + \'</strong></div>\').join(\'\')\n          + \'</div>\';\n      }).join(\'\');\n    }\n\n    // ── Tips (mixed-state aware) ──────────────────────────\n    const tips = d.tips || [];\n    const tipsWrap = $(\'cr-tips-wrap\');\n    if (tipsWrap) {\n      const stateColors = {\n        focused:    { color:\'#10b981\', bg:\'rgba(16,185,129,0.08)\',  border:\'rgba(16,185,129,0.25)\'  },\n        fatigued:   { color:\'#94a3b8\', bg:\'rgba(148,163,184,0.08)\', border:\'rgba(148,163,184,0.25)\' },\n        anxious:    { color:\'#ef4444\', bg:\'rgba(239,68,68,0.08)\',   border:\'rgba(239,68,68,0.25)\'   },\n        overloaded: { color:\'#f97316\', bg:\'rgba(249,115,22,0.08)\',  border:\'rgba(249,115,22,0.25)\'  },\n        loading:    { color:\'#f59e0b\', bg:\'rgba(245,158,11,0.08)\',  border:\'rgba(245,158,11,0.25)\'  }\n      };\n      const stateEmoji = { focused:\'🎯\', fatigued:\'😴\', anxious:\'🧘\', overloaded:\'🧩\', loading:\'🤔\' };\n      const stateLabel = { focused:\'Focused\', fatigued:\'Fatigued\', anxious:\'Anxious\', overloaded:\'Overloaded\', loading:\'Processing\' };\n\n      if (!tips.length) {\n        tipsWrap.innerHTML = \'<div class="cr-nocam-msg">📷 Enable camera for personalised tips next time.</div>\';\n      } else {\n        tipsWrap.innerHTML = tips.map(t => {\n          const sc = stateColors[t.state] || stateColors.focused;\n          const pctBadge = t.pct < 100\n            ? \'<span class="cr-tip-pct" style="color:\' + sc.color + \'">\' + t.pct + \'% of quiz</span>\' : \'\';\n          return \'<div class="cr-tip-card" style="border-color:\' + sc.border + \';background:\' + sc.bg + \'">\'\n            + \'<div class="cr-tip-header">\'\n              + \'<span class="cr-tip-emoji">\' + (stateEmoji[t.state]||\'💡\') + \'</span>\'\n              + \'<span class="cr-tip-state" style="color:\' + sc.color + \'">\' + (stateLabel[t.state]||t.state) + \'</span>\'\n              + pctBadge\n            + \'</div>\'\n            // Pick 3 random tips from each category so it\'s fresh each time\n            + \'<div class="cr-tip-section-lbl">⚡ Instant Tricks</div>\'\n            + shuffle(t.instant).slice(0,3).map(x => \'<div class="cr-tip-row">\' + escHtml(x) + \'</div>\').join(\'\')\n            + \'<div class="cr-tip-section-lbl">📚 Study Smarter</div>\'\n            + shuffle(t.study).slice(0,3).map(x => \'<div class="cr-tip-row">\' + escHtml(x) + \'</div>\').join(\'\')\n            + \'<div class="cr-tip-section-lbl">🏆 Score Higher</div>\'\n            + shuffle(t.score).slice(0,3).map(x => \'<div class="cr-tip-row">\' + escHtml(x) + \'</div>\').join(\'\')\n            + \'</div>\';\n        }).join(\'\');\n      }\n    }\n\n    $(\'cr-loading\').classList.add(\'hidden\');\n    $(\'cr-content\').classList.remove(\'hidden\');\n\n  } catch(e) {\n    console.warn(\'Cog report error:\', e);\n    showScreen(\'dashboard\');\n  }\n}\n\nfunction closeCogReport() {\n  const modal = $(\'cog-report-modal\');\n  if (modal) modal.classList.add(\'hidden\');\n  showScreen(\'dashboard\');\n}\n\n// ═══════════════════════════════════════════════════════\n//  DASHBOARD — Charts + Stats\n// ═══════════════════════════════════════════════════════\n\nfunction destroyChart(key) {\n  if (charts[key]) { charts[key].destroy(); charts[key] = null; }\n}\n\nasync function renderDashboard() {\n  try {\n    const res = await fetch(API + \'/api/stats/\' + getEffectiveSID());\n    if (!res.ok) return;\n    const data = await res.json();\n\n    // Sync state with server\n    state.score  = data.score  || 0;\n    state.xp     = data.xp    || 0;\n    state.level  = data.level || 1;\n    state.streak = data.streak || 0;\n    state.totalQ   = data.total_q   || 0;\n    state.correctQ = data.correct_q || 0;\n    state.topics_studied = data.topics_studied || [];\n    state.study_dates    = data.study_dates    || [];\n    state.badges         = data.badges         || [];\n    updateHomeStats();\n\n    const acc = data.total_q > 0\n      ? Math.round(data.correct_q / data.total_q * 100) : 0;\n\n    // Big stats\n    $(\'big-stats\').innerHTML = [\n      {i:\'⚡\',l:\'XP\',        v:data.xp,      c:\'#a78bfa\'},\n      {i:\'🏆\',l:\'Level\',     v:data.level,   c:\'#f59e0b\'},\n      {i:\'🔥\',l:\'Streak\',    v:data.streak,  c:\'#f97316\'},\n      {i:\'⭐\',l:\'Score\',     v:data.score,   c:\'#6366f1\'},\n      {i:\'🎯\',l:\'Accuracy\',  v:acc+\'%\',      c:\'#10b981\'},\n      {i:\'📝\',l:\'Answered\',  v:data.total_q, c:\'#06b6d4\'}\n    ].map(function(s) {\n      return \'<div class="bs"><div class="bs-icon">\' + s.i + \'</div>\'\n        + \'<div class="bs-val" style="color:\' + s.c + \'">\' + s.v + \'</div>\'\n        + \'<div class="bs-lbl">\' + s.l + \'</div></div>\';\n    }).join(\'\');\n\n    // Level bar\n    const xpInLevel = data.xp % 200;\n    $(\'lp-bar\').style.width = (xpInLevel / 200 * 100) + \'%\';\n    $(\'lp-lbl\').textContent = \'Level \' + data.level;\n    $(\'lp-xp\').textContent  = (200 - xpInLevel) + \' XP to Level \' + (data.level + 1);\n\n    // Charts\n    renderLineChart(data.progress_data || []);\n    renderPieChart(data.correct_q || 0, (data.total_q || 0) - (data.correct_q || 0));\n    renderMasteryChart(data.topic_stats || []);\n    renderTopicTable(data.topic_stats || []);\n    renderCalendar(data.study_dates || []);\n\n    // Topics\n    $(\'topics-list\').innerHTML = (data.topics_studied || []).length\n      ? data.topics_studied.map(t => \'<span class="topic-tag">\' + escHtml(t) + \'</span>\').join(\'\')\n      : \'<span class="empty-tag">No topics yet — start a quiz!</span>\';\n\n    $(\'weak-list\').innerHTML = (data.weak_topics || []).length\n      ? data.weak_topics.map(t => \'<span class="weak-tag2">⚠️ \' + escHtml(t) + \'</span>\').join(\'\')\n      : \'<span class="empty-tag">No weak topics — great job! ✅</span>\';\n\n    // Correct/wrong/total\n    $(\'d-correct\').textContent = data.correct_q || 0;\n    $(\'d-wrong\').textContent   = (data.total_q || 0) - (data.correct_q || 0);\n    $(\'d-total\').textContent   = data.total_q  || 0;\n\n    // Badges\n    $(\'d-badges\').innerHTML = (data.badges || []).length\n      ? data.badges.map(b => \'<span class="badge">\' + b + \'</span>\').join(\'\')\n      : \'<span class="nb">Complete quizzes to earn badges!</span>\';\n\n  } catch(e) {\n    console.warn(\'Dashboard error:\', e.message);\n  }\n}\n\nconst CHART_OPTS = {\n  responsive: true, maintainAspectRatio: false,\n  plugins: { legend: { display: false } },\n  scales: {\n    x: { ticks:{ color:\'rgba(255,255,255,0.4)\' }, grid:{ color:\'rgba(255,255,255,0.05)\' } },\n    y: { ticks:{ color:\'rgba(255,255,255,0.4)\' }, grid:{ color:\'rgba(255,255,255,0.05)\' } }\n  }\n};\n\nfunction renderLineChart(data) {\n  destroyChart(\'progress\');\n  const ctx = $(\'progress-chart\');\n  if (!ctx || !data.length) return;\n  charts.progress = new Chart(ctx, {\n    type: \'line\',\n    data: {\n      labels: data.map(d => \'Q\' + d.n),\n      datasets: [{\n        label: \'Accuracy %\', data: data.map(d => d.accuracy),\n        borderColor: \'#6366f1\', backgroundColor: \'rgba(99,102,241,0.1)\',\n        borderWidth: 2.5, pointRadius: 3, fill: true, tension: 0.4\n      }]\n    },\n    options: {\n      ...CHART_OPTS,\n      scales: {\n        x: CHART_OPTS.scales.x,\n        y: { ...CHART_OPTS.scales.y, beginAtZero: true, max: 100,\n          ticks: { color: \'rgba(255,255,255,0.4)\', callback: v => v + \'%\' } }\n      }\n    }\n  });\n}\n\nfunction renderPieChart(correct, wrong) {\n  destroyChart(\'pie\');\n  const ctx = $(\'pie-chart\');\n  if (!ctx || (correct === 0 && wrong === 0)) return;\n  charts.pie = new Chart(ctx, {\n    type: \'doughnut\',\n    data: {\n      labels: [\'Correct\', \'Wrong\'],\n      datasets: [{\n        data: [correct, wrong],\n        backgroundColor: [\'rgba(16,185,129,0.8)\', \'rgba(239,68,68,0.7)\'],\n        borderColor: [\'#10b981\', \'#ef4444\'], borderWidth: 2\n      }]\n    },\n    options: {\n      responsive: true, maintainAspectRatio: false,\n      plugins: {\n        legend: {\n          display: true, position: \'bottom\',\n          labels: { color: \'rgba(255,255,255,0.6)\', padding: 15, font: { size: 12 } }\n        }\n      }\n    }\n  });\n}\n\nfunction renderMasteryChart(topicStats) {\n  destroyChart(\'mastery\');\n  const ctx = $(\'mastery-chart\');\n  if (!ctx || !topicStats.length) return;\n  const labels  = topicStats.map(t => t.topic.length > 16 ? t.topic.slice(0,16)+\'…\' : t.topic);\n  const values  = topicStats.map(t => t.mastery);\n  const colors  = values.map(m => m>=80 ? \'rgba(16,185,129,0.8)\'\n                                : m>=50 ? \'rgba(245,158,11,0.8)\' : \'rgba(239,68,68,0.7)\');\n  charts.mastery = new Chart(ctx, {\n    type: \'bar\',\n    data: { labels, datasets: [{ label: \'Mastery %\', data: values, backgroundColor: colors, borderRadius: 6 }] },\n    options: {\n      ...CHART_OPTS,\n      scales: {\n        x: { ticks: { color: \'rgba(255,255,255,0.5)\', font: { size: 11 } }, grid: { color: \'rgba(255,255,255,0.05)\' } },\n        y: { beginAtZero: true, max: 100,\n          ticks: { color: \'rgba(255,255,255,0.4)\', callback: v => v + \'%\' },\n          grid: { color: \'rgba(255,255,255,0.05)\' } }\n      }\n    }\n  });\n}\n\nfunction renderTopicTable(topicStats) {\n  const wrap = $(\'topic-table\');\n  if (!wrap) return;\n  if (!topicStats.length) {\n    wrap.innerHTML = \'<p style="color:var(--sub);font-style:italic;padding:10px">Answer questions to see stats per topic</p>\';\n    return;\n  }\n  wrap.innerHTML = \'<div class="tt-row header">\'\n    + \'<div>Topic</div><div>Total</div><div>Correct</div><div>Mastery</div><div>Avg Time</div></div>\'\n    + topicStats.map(function(t) {\n      return \'<div class="tt-row">\'\n        + \'<div style="font-weight:600">\' + escHtml(t.topic) + \'</div>\'\n        + \'<div>\' + t.total + \'</div>\'\n        + \'<div style="color:#10b981">\' + t.correct + \'</div>\'\n        + \'<div>\'\n          + \'<div style="font-size:.8rem;margin-bottom:3px">\' + t.mastery + \'%</div>\'\n          + \'<div class="mastery-bar-wrap"><div class="mastery-bar" style="width:\' + t.mastery + \'%"></div></div>\'\n        + \'</div>\'\n        + \'<div style="color:#67e8f9">\' + t.avg_time + \'s</div>\'\n        + \'</div>\';\n    }).join(\'\');\n}\n\nfunction renderCalendar(studyDates) {\n  const cal = $(\'streak-calendar\');\n  if (!cal) return;\n  const studied = new Set(studyDates);\n  const today = new Date();\n  let html = \'\';\n  for (let i = 89; i >= 0; i--) {\n    const d = new Date(today);\n    d.setDate(d.getDate() - i);\n    const iso = d.toISOString().split(\'T\')[0];\n    html += \'<div class="cal-day\'\n      + (studied.has(iso) ? \' studied\' : \'\')\n      + (i === 0 ? \' today\' : \'\')\n      + \'" title="\' + iso + \'"></div>\';\n  }\n  cal.innerHTML = html;\n}\n\n\n// ═══════════════════════════════════════════════════════\n//  LEADERBOARD\n// ═══════════════════════════════════════════════════════\n\nasync function loadLeaderboard() {\n  const lb = $(\'lb-table\');\n  lb.innerHTML = \'<div class="lb-loading">Loading...</div>\';\n  try {\n    const res = await fetch(API + \'/api/leaderboard\');\n    if (!res.ok) throw new Error(\'Failed to load\');\n    const data = await res.json();\n    const rows = data.leaderboard || [];\n    if (!rows.length) {\n      lb.innerHTML = \'<div class="lb-loading">No scores yet — be the first!</div>\';\n      return;\n    }\n    const medals = [\'🥇\',\'🥈\',\'🥉\'];\n    lb.innerHTML = rows.map(function(r) {\n      return \'<div class="lb-row-item">\'\n        + \'<div class="lb-rank \' + ([\'gold\',\'silver\',\'bronze\'][r.rank-1]||\'\') + \'">\'\n          + (r.rank <= 3 ? medals[r.rank-1] : r.rank) + \'</div>\'\n        + \'<div class="lb-name">\' + escHtml(r.name) + \'</div>\'\n        + \'<div><div class="lb-score">\' + r.score + \' pts</div>\'\n          + \'<div class="lb-acc">\' + (r.accuracy||0) + \'% · \' + (r.xp||0) + \' XP</div></div>\'\n        + \'</div>\';\n    }).join(\'\');\n  } catch(e) {\n    lb.innerHTML = \'<div class="lb-loading">Failed to load leaderboard</div>\';\n  }\n}\n\nasync function submitToLeaderboard() {\n  const nameEl = $(\'lb-name\');\n  const name = (nameEl ? nameEl.value : \'\').trim();\n  if (!name) { showToast(\'Enter your name first!\'); if (nameEl) nameEl.focus(); return; }\n\n  const errEl = $(\'lb-error\');\n  if (errEl) errEl.classList.add(\'hidden\');\n\n  try {\n    // Get fresh server stats for accurate score\n    const statsRes = await fetch(API + \'/api/stats/\' + getEffectiveSID());\n    const fresh = statsRes.ok ? await statsRes.json() : {};\n    const finalScore = Math.max(state.score, fresh.score || 0);\n    const finalXp    = Math.max(state.xp,    fresh.xp    || 0);\n    const finalAcc   = fresh.accuracy || (state.totalQ > 0\n      ? Math.round(state.correctQ / state.totalQ * 100) : 0);\n    const topics = (fresh.topics_studied || state.topics_studied || []).length;\n\n    const res = await fetch(API + \'/api/leaderboard\', {\n      method: \'POST\',\n      headers: { \'Content-Type\': \'application/json\' },\n      body: JSON.stringify({ name, score: finalScore, xp: finalXp, accuracy: finalAcc, topics, session_id: getEffectiveSID() })\n    });\n    if (!res.ok) throw new Error(\'Server error\');\n    const data = await res.json();\n    showToast(\'🏆 Submitted! Your rank: #\' + data.rank, 4000);\n    showScreen(\'leaderboard\');\n  } catch(e) {\n    showToast(\'Failed to submit score\');\n    if (errEl) { errEl.textContent = \'❌ \' + e.message; errEl.classList.remove(\'hidden\'); }\n  }\n}\n\n\n// ═══════════════════════════════════════════════════════\n//  REVIEW — Spaced Repetition\n// ═══════════════════════════════════════════════════════\n\nasync function loadReview() {\n  const list  = $(\'review-list\');\n  const empty = $(\'review-empty\');\n  list.innerHTML = \'\';\n\n  try {\n    const [dueRes, statsRes] = await Promise.all([\n      fetch(API + \'/api/due-topics/\' + getEffectiveSID()),\n      fetch(API + \'/api/stats/\' + getEffectiveSID())\n    ]);\n    const dueData   = dueRes.ok   ? await dueRes.json()   : { due: [] };\n    const statsData = statsRes.ok ? await statsRes.json() : { topics_studied: [] };\n\n    const due       = dueData.due || [];\n    const allTopics = statsData.topics_studied || [];\n    const dueSet    = new Set(due.map(d => d.topic));\n\n    if (!allTopics.length && !due.length) {\n      list.classList.add(\'hidden\');\n      empty.classList.remove(\'hidden\');\n      return;\n    }\n\n    empty.classList.add(\'hidden\');\n    list.classList.remove(\'hidden\');\n    let html = \'\';\n\n    if (due.length) {\n      html += \'<div class="review-section-title">⏰ Due Now (\' + due.length + \')</div>\';\n      html += due.map(function(d) {\n        return \'<div class="review-item due-now">\'\n          + \'<div><div class="review-topic">\' + escHtml(d.topic) + \'</div>\'\n          + \'<div class="review-interval">Due for review</div></div>\'\n          + \'<button class="review-btn urgent" onclick="quickReview(\\\'\'\n          + d.topic.replace(/\'/g,"\\\\\'") + \'\\\')">Review Now →</button>\'\n          + \'</div>\';\n      }).join(\'\');\n    }\n\n    const upcoming = allTopics.filter(t => !dueSet.has(t));\n    if (upcoming.length) {\n      html += \'<div class="review-section-title" style="margin-top:20px">📚 All Topics (\' + upcoming.length + \')</div>\';\n      html += upcoming.map(function(t) {\n        return \'<div class="review-item">\'\n          + \'<div><div class="review-topic">\' + escHtml(t) + \'</div>\'\n          + \'<div class="review-interval">Not due yet</div></div>\'\n          + \'<button class="review-btn" onclick="quickReview(\\\'\' + t.replace(/\'/g,"\\\\\'") + \'\\\')">\'\n          + \'Practice Again →</button>\'\n          + \'</div>\';\n      }).join(\'\');\n    }\n\n    list.innerHTML = html;\n  } catch(e) {\n    list.innerHTML = \'<p style="color:var(--sub);padding:20px">Error loading topics: \' + e.message + \'</p>\';\n  }\n}\n\nfunction quickReview(topic) {\n  const inp = $(\'topic-input\');\n  if (inp) inp.value = topic;\n  showScreen(\'home\');\n  setTimeout(generateQuestions, 200);\n}\n\n\n// ═══════════════════════════════════════════════════════\n//  EXPORT REPORT\n// ═══════════════════════════════════════════════════════\n\nasync function exportReport() {\n  try {\n    const res = await fetch(API + \'/api/export-report/\' + getEffectiveSID());\n    if (!res.ok) throw new Error(\'Server error\');\n    const data = await res.json();\n    const blob = new Blob([JSON.stringify(data, null, 2)], { type: \'application/json\' });\n    const url  = URL.createObjectURL(blob);\n    const a    = document.createElement(\'a\');\n    a.href = url;\n    a.download = \'EduMate360_Report_\' + new Date().toISOString().split(\'T\')[0] + \'.json\';\n    document.body.appendChild(a);\n    a.click();\n    document.body.removeChild(a);\n    URL.revokeObjectURL(url);\n    showToast(\'📥 Report downloaded!\');\n  } catch(e) {\n    showToast(\'Export failed: \' + e.message);\n  }\n}\n\n\n// ═══════════════════════════════════════════════════════════════════\n//  CAMERA + EMOTION ENGINE\n//  Uses MediaPipe Tasks Vision FaceLandmarker\n//  — No separate model files, no CDN failure\n//  — 478 facial landmarks → cognitive state\n// ═══════════════════════════════════════════════════════════════════\n\nlet faceLandmarker   = null;\nlet mpLoading        = false;\nlet camStream        = null;\nlet emotionInterval  = null;\nlet lastVideoTime    = -1;\nlet consecutiveAnxious = 0;\nlet breathingActive  = false;\nlet currentQEmotions = [];\nlet camViewHidden    = false;  // video hidden but cam still running  // emotion log for current question\nlet allQuizEmotions  = [];  // accumulates all emotions across full quiz\n\nconst COG_META = {\n  focused:    { emoji:\'🎯\', label:\'Focused\',    color:\'#10b981\', tip:\'Performance looks good — keep it up!\' },\n  loading:    { emoji:\'🤔\', label:\'Processing\', color:\'#f59e0b\', tip:\'Take your time, think it through.\' },\n  overloaded: { emoji:\'🧩\', label:\'Overloaded\', color:\'#f97316\', tip:\'Slow down — break the problem into parts.\' },\n  fatigued:   { emoji:\'😴\', label:\'Fatigued\',   color:\'#94a3b8\', tip:\'Consider a short break soon.\' },\n  anxious:    { emoji:\'🧘\', label:\'Anxious\',    color:\'#ef4444\', tip:\'Breathe — you know this material.\' },\n};\n\nconst BAR_COLORS = {\n  happy:\'#10b981\', focused:\'#6366f1\', confused:\'#f59e0b\', tired:\'#94a3b8\', anxious:\'#ef4444\'\n};\n\n// ── Load MediaPipe FaceLandmarker ─────────────────────────────────\nasync function preloadFaceApi() {\n  if (mpLoading || faceLandmarker) return;\n  mpLoading = true;\n  try {\n    // Access the global Vision object from the loaded bundle\n    const Vision = window.vision || window.mpVision;\n    if (!Vision) {\n      // Try dynamic import fallback\n      const mod = await import(\'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs\');\n      window.vision = mod;\n    }\n    const V = window.vision || window.mpVision;\n    if (!V || !V.FaceLandmarker) {\n      console.warn(\'MediaPipe Vision bundle not available yet\');\n      mpLoading = false;\n      return;\n    }\n    const { FaceLandmarker, FilesetResolver } = V;\n    const filesetResolver = await FilesetResolver.forVisionTasks(\n      \'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm\'\n    );\n    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {\n      baseOptions: {\n        modelAssetPath: \'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task\',\n        delegate: \'CPU\'\n      },\n      outputFaceBlendshapes: true,\n      runningMode: \'VIDEO\',\n      numFaces: 1\n    });\n    console.log(\'✅ MediaPipe FaceLandmarker ready\');\n    const statusEl = $(\'face-status\');\n    if (statusEl) { statusEl.textContent = \'AI ready ✓\'; statusEl.style.background = \'rgba(16,185,129,.8)\'; }\n  } catch(e) {\n    console.warn(\'MediaPipe load error:\', e.message);\n    const statusEl = $(\'face-status\');\n    if (statusEl) { statusEl.textContent = \'AI unavailable\'; statusEl.style.background = \'rgba(239,68,68,.7)\'; }\n  }\n  mpLoading = false;\n}\n\n// ── Blendshapes → cognitive state ────────────────────────────────\nfunction blendsToCog(blendshapes) {\n  if (!blendshapes || !blendshapes.categories) return \'focused\';\n  const bs = {};\n  blendshapes.categories.forEach(c => { bs[c.categoryName] = c.score; });\n  const blink   = ((bs.eyeBlinkLeft||0) + (bs.eyeBlinkRight||0)) / 2;\n  const browDn  = ((bs.browDownLeft||0) + (bs.browDownRight||0)) / 2;\n  const browUp  = bs.browInnerUp || 0;\n  const smile   = ((bs.mouthSmileLeft||0) + (bs.mouthSmileRight||0)) / 2;\n  const frown   = ((bs.mouthFrownLeft||0) + (bs.mouthFrownRight||0)) / 2;\n  const jawOpen = bs.jawOpen || 0;\n  const squint  = ((bs.eyeSquintLeft||0) + (bs.eyeSquintRight||0)) / 2;\n  if (browDn > 0.35 && smile < 0.1 && frown > 0.15) return \'anxious\';\n  if (squint > 0.3  && browDn > 0.25)                return \'overloaded\';\n  if (blink  > 0.4)                                   return \'fatigued\';\n  if (browUp > 0.3  && jawOpen > 0.1)                 return \'loading\';\n  return \'focused\';\n}\n\nfunction blendsToBarData(blendshapes) {\n  if (!blendshapes || !blendshapes.categories) return [];\n  const bs = {};\n  blendshapes.categories.forEach(c => { bs[c.categoryName] = c.score; });\n  return [\n    { name:\'happy\',   val: Math.round(((bs.mouthSmileLeft||0)+(bs.mouthSmileRight||0))/2*100) },\n    { name:\'focused\', val: Math.round(Math.max(0, 1-(bs.browDownLeft||0)-(bs.eyeSquintLeft||0))*60) },\n    { name:\'confused\',val: Math.round(((bs.browInnerUp||0)*0.5+(bs.jawOpen||0)*0.5)*100) },\n    { name:\'tired\',   val: Math.round(((bs.eyeBlinkLeft||0)+(bs.eyeBlinkRight||0))/2*100) },\n    { name:\'anxious\', val: Math.round(((bs.browDownLeft||0)+(bs.mouthFrownLeft||0))/2*100) },\n  ];\n}\n\n// ── Main detection loop ───────────────────────────────────────────\nasync function runEmotionDetection() {\n  const video    = $(\'webcam\');\n  const canvas   = $(\'face-canvas\');\n  const statusEl = $(\'face-status\');\n  if (!video || !canvas || video.paused || video.ended) return;\n\n  if (!faceLandmarker) {\n    if (statusEl) { statusEl.textContent = \'Loading AI...\'; statusEl.style.background = \'rgba(245,158,11,.7)\'; }\n    if (!mpLoading) preloadFaceApi();\n    return;\n  }\n  try {\n    const nowMs = performance.now();\n    if (video.currentTime === lastVideoTime) return;\n    lastVideoTime = video.currentTime;\n    const results = faceLandmarker.detectForVideo(video, nowMs);\n    const ctx = canvas.getContext(\'2d\');\n    ctx.clearRect(0,0,160,120);\n\n    if (!results.faceLandmarks || !results.faceLandmarks.length) {\n      if (statusEl) { statusEl.textContent = \'No face detected\'; statusEl.style.background = \'rgba(239,68,68,.7)\'; }\n      return;\n    }\n    // Draw face box\n    const lm = results.faceLandmarks[0];\n    const xs = lm.map(p=>p.x*160), ys = lm.map(p=>p.y*120);\n    const x1=Math.min(...xs), x2=Math.max(...xs), y1=Math.min(...ys), y2=Math.max(...ys);\n    ctx.strokeStyle=\'#6366f1\'; ctx.lineWidth=2;\n    ctx.strokeRect(x1-4, y1-4, x2-x1+8, y2-y1+8);\n    if (statusEl) { statusEl.textContent=\'Face detected ✓\'; statusEl.style.background=\'rgba(16,185,129,.7)\'; }\n\n    const bs = results.faceBlendshapes?.[0];\n    const cogState = bs ? blendsToCog(bs) : \'focused\';\n    state.cogStateRaw = cogState;  // store fine-grained for submit\n    const meta = COG_META[cogState] || COG_META.focused;\n\n    // Update cognitive state display\n    const stateEl = $(\'emo-state\');\n    if (stateEl) { stateEl.textContent = meta.emoji+\' \'+meta.label; stateEl.style.color = meta.color; }\n    const subEl = $(\'emo-sub\');\n    if (subEl) subEl.textContent = meta.tip;\n\n    // Update cog badge in live strip\n    const badge = $(\'cog-badge\');\n    if (badge) { badge.textContent = meta.emoji+\' \'+meta.label; badge.style.borderColor=meta.color; badge.style.color=meta.color; badge.classList.remove(\'hidden\'); }\n\n    // Emotion bars\n    const bars = bs ? blendsToBarData(bs) : [];\n    const barsEl = $(\'emo-bars\');\n    if (barsEl) barsEl.innerHTML = bars.map(b =>\n      \'<div class="emo-bar-row">\'\n      +\'<div class="emo-bar-label">\'+b.name+\'</div>\'\n      +\'<div class="emo-bar-track"><div class="emo-bar-fill" style="width:\'+b.val+\'%;background:\'+(BAR_COLORS[b.name]||\'#6366f1\')+\'"></div></div>\'\n      +\'<div class="emo-bar-pct">\'+b.val+\'%</div>\'\n      +\'</div>\').join(\'\');\n\n    // Log emotion for this question\n    currentQEmotions.push({ cogState, ts: Date.now() });\n\n    // Anxiety check — trigger breathing exercise after 3 consecutive anxious detections\n    if (cogState === \'anxious\') {\n      consecutiveAnxious++;\n      if (consecutiveAnxious >= 3 && !breathingActive) triggerBreathingExercise();\n    } else {\n      consecutiveAnxious = 0;\n    }\n\n    // Send to backend (throttled — only on state change)\n    const backendEmo = (cogState===\'anxious\'||cogState===\'overloaded\') ? \'stressed\'\n                     : (cogState===\'fatigued\'||cogState===\'loading\')   ? \'confused\' : \'focused\';\n    if (backendEmo !== state.emotion) {\n      state.emotion = backendEmo;\n      fetch(API+\'/api/emotion\', {\n        method:\'POST\', headers:{\'Content-Type\':\'application/json\'},\n        body: JSON.stringify({session_id:getEffectiveSID(), emotion:backendEmo})\n      }).catch(()=>{});\n    }\n  } catch(e) { console.warn(\'Detection error:\', e.message); }\n}\n\n// ── Toggle camera on/off ──────────────────────────────────────────\nasync function toggleCam() {\n  const btn   = $(\'cam-btn\');\n  const panel = $(\'cam-panel\');\n  if (!btn || !panel) return;\n\n  if (camStream) {\n    // Turn OFF\n    clearInterval(emotionInterval);\n    emotionInterval = null;\n    camStream.getTracks().forEach(t => t.stop());\n    camStream = null;\n    lastVideoTime = -1;\n    const video = $(\'webcam\');\n    if (video) video.srcObject = null;\n    panel.classList.add(\'hidden\');\n    btn.textContent = \'📷 Cam\';\n    btn.classList.remove(\'active\');\n    camViewHidden = false;\n    const camBody2 = $(\'cam-body\'); if (camBody2) camBody2.classList.remove(\'cam-body-hidden\');\n    const hideBtn2 = $(\'cam-hide-btn\'); if (hideBtn2) hideBtn2.textContent = \'Hide\';\n    const badge = $(\'cog-badge\');\n    if (badge) badge.classList.add(\'hidden\');\n    return;\n  }\n\n  // Turn ON\n  try {\n    camStream = await navigator.mediaDevices.getUserMedia({ video: { width:160, height:120, facingMode:\'user\' }, audio:false });\n    const video = $(\'webcam\');\n    if (video) { video.srcObject = camStream; await video.play().catch(()=>{}); }\n    panel.classList.remove(\'hidden\');\n    btn.textContent = \'📷 On\';\n    btn.classList.add(\'active\');\n    showToast(\'📷 Camera on — loading AI models...\');\n    // Start detection loop\n    emotionInterval = setInterval(runEmotionDetection, 1200);\n    // Load MediaPipe in background\n    if (!faceLandmarker && !mpLoading) preloadFaceApi();\n  } catch(e) {\n    showToast(\'Camera access denied or unavailable\');\n    camStream = null;\n  }\n}\n\n\n// ── Show / Hide camera video feed (camera stays ON) ──────────────\nfunction toggleCamView() {\n  if (!camStream) return;  // camera not running\n  const body   = $(\'cam-body\');\n  const hideBtn = $(\'cam-hide-btn\');\n  if (!body || !hideBtn) return;\n\n  camViewHidden = !camViewHidden;\n\n  if (camViewHidden) {\n    // Hide the video feed but keep emotion detection running\n    body.classList.add(\'cam-body-hidden\');\n    hideBtn.textContent = \'Show\';\n    hideBtn.title = \'Show video feed\';\n    showToast(\'📷 Camera running in background — focus on your quiz!\', 2500);\n  } else {\n    body.classList.remove(\'cam-body-hidden\');\n    hideBtn.textContent = \'Hide\';\n    hideBtn.title = \'Hide video feed\';\n  }\n}\n\n// ── Breathing exercise (triggered on 3x anxious) ──────────────────\nfunction triggerBreathingExercise() {\n  breathingActive = true;\n  const overlay = document.createElement(\'div\');\n  overlay.className = \'breathing-overlay\';\n  overlay.id = \'breathing-overlay\';\n  const phases = [\'Breathe In\',\'Hold\',\'Breathe Out\',\'Hold\'];\n  let phase = 0, cycle = 0;\n  function nextPhase() {\n    if (cycle >= 2) { overlay.remove(); breathingActive=false; consecutiveAnxious=0; return; }\n    overlay.innerHTML =\n      \'<div class="breathing-circle">\'+phases[phase]+\'</div>\'\n      +\'<div class="breathing-text">Box Breathing — \'+(cycle===0?\'Cycle 1\':\'Cycle 2\')+\' of 2</div>\'\n      +\'<button class="breathing-skip" onclick="var o=document.getElementById(&quot;breathing-overlay&quot;);if(o)o.remove();window.breathingActive=false;">Skip</button>\';\n    phase++;\n    if (phase >= 4) { phase=0; cycle++; }\n    setTimeout(nextPhase, 4000);\n  }\n  document.body.appendChild(overlay);\n  nextPhase();\n}\nwindow.breathingActive = false;\n\n\n\n\n\n\n// ══════════════════════════════════════════════════════\n//  MISTAKE NOTEBOOK\n// ══════════════════════════════════════════════════════\nasync function loadMistakes(topicFilter) {\n  var url = topicFilter\n    ? (API + \'/api/mistakes/\' + getEffectiveSID() + \'?topic=\' + encodeURIComponent(topicFilter))\n    : (API + \'/api/mistakes/\' + getEffectiveSID());\n  try {\n    var res  = await fetch(url);\n    var data = await res.json();\n    renderMistakes(data.mistakes || [], data.topics || []);\n  } catch(e) {\n    var el = $(\'mn-list\');\n    if (el) el.innerHTML = \'<p style="color:var(--sub);padding:20px">Error loading mistakes.</p>\';\n  }\n}\n\nfunction renderMistakes(mistakes, topics) {\n  var sel = $(\'mn-filter\');\n  if (sel) {\n    var cur = sel.value;\n    sel.innerHTML = \'<option value="">All Topics</option>\'\n      + topics.map(function(t) {\n          return \'<option value="\' + escHtml(t) + \'"\'\n            + (t === cur ? \' selected\' : \'\') + \'>\' + escHtml(t) + \'</option>\';\n        }).join(\'\');\n  }\n  var statsEl = $(\'mn-stats\');\n  var listEl  = $(\'mn-list\');\n  if (!listEl) return;\n\n  if (!mistakes.length) {\n    if (statsEl) statsEl.innerHTML = \'\';\n    listEl.innerHTML =\n      \'<div class="mn-empty">\'\n      + \'<div style="font-size:3rem">&#127881;</div>\'\n      + \'<div style="font-size:1.1rem;font-weight:700;margin:10px 0">No mistakes yet!</div>\'\n      + \'<div style="color:var(--sub);font-size:.88rem">Answer some questions to see your mistake notebook here.</div>\'\n      + \'</div>\';\n    return;\n  }\n\n  var total   = mistakes.length;\n  var retried = mistakes.filter(function(m){ return m.retried; }).length;\n  var pending = total - retried;\n  if (statsEl) {\n    statsEl.innerHTML =\n      \'<div class="mn-stat-row">\'\n      + \'<div class="mn-stat"><span class="mn-stat-v" style="color:#ef4444">\' + total + \'</span>\'\n      + \'<span class="mn-stat-l">Total Mistakes</span></div>\'\n      + \'<div class="mn-stat"><span class="mn-stat-v" style="color:#f59e0b">\' + pending + \'</span>\'\n      + \'<span class="mn-stat-l">Pending</span></div>\'\n      + \'<div class="mn-stat"><span class="mn-stat-v" style="color:#10b981">\' + retried + \'</span>\'\n      + \'<span class="mn-stat-l">Retried</span></div>\'\n      + \'</div>\';\n  }\n\n  var grouped = {};\n  mistakes.forEach(function(m) {\n    if (!grouped[m.topic]) grouped[m.topic] = [];\n    grouped[m.topic].push(m);\n  });\n\n  listEl.innerHTML = Object.keys(grouped).map(function(topic) {\n    return \'<div class="mn-group">\'\n      + \'<div class="mn-group-title">&#128218; \' + escHtml(topic)\n      + \' <span class="mn-group-count">\' + grouped[topic].length + \'</span></div>\'\n      + grouped[topic].map(function(m) {\n          return \'<div class="mn-card\' + (m.retried ? \' mn-retried\' : \'\') + \'">\'\n            + \'<div class="mn-card-top">\'\n            + \'<span class="mn-concept">&#127991; \' + escHtml(m.concept || \'General\') + \'</span>\'\n            + \'<span class="mn-date">\' + escHtml(m.date || \'\') + \'</span>\'\n            + (m.retried ? \'<span class="mn-done-badge">&#10003; Retried</span>\' : \'\')\n            + \'</div>\'\n            + \'<div class="mn-question">&#10067; \' + escHtml(m.question) + \'</div>\'\n            + \'<div class="mn-answers">\'\n            + \'<div class="mn-your-ans">&#10060; You said: <strong>\' + escHtml(m.your_answer) + \'</strong></div>\'\n            + \'<div class="mn-correct-ans">&#10003; Correct: <strong>\' + escHtml(m.correct_answer) + \'</strong></div>\'\n            + \'</div>\'\n            + \'<div class="mn-explanation">&#128161; \' + escHtml(m.explanation) + \'</div>\'\n            + \'<div class="mn-actions">\'\n            + \'<button class="mn-retry-btn" onclick="retryMistake(\' + m.id + \')"\'\n            + (m.retried ? \' disabled\' : \'\') + \'>\'\n            + (m.retried ? \'&#10003; Done\' : \'&#8635; Mark Retried\') + \'</button>\'\n            + \'<button class="mn-ask-btn" onclick="askTutorAbout(\\\'\'\n            + escHtml(m.concept||m.topic).replace(/\'/g,"\\\\\'") + \'\\\',\\\'\'\n            + escHtml(m.topic).replace(/\'/g,"\\\\\'") + \'\\\')">&#129302; Ask Tutor</button>\'\n            + \'</div></div>\';\n        }).join(\'\')\n      + \'</div>\';\n  }).join(\'\');\n}\n\nfunction filterMistakes(topic) { loadMistakes(topic); }\n\nasync function retryMistake(id) {\n  try {\n    var res  = await fetch(API + \'/api/mistakes/\' + getEffectiveSID() + \'/retry/\' + id, {method:\'POST\'});\n    var data = await res.json();\n    if (data.ok) {\n      showToast(\'Marked as retried!\');\n      loadMistakes($(\'mn-filter\') ? $(\'mn-filter\').value : \'\');\n    }\n  } catch(e) { showToast(\'Error\'); }\n}\n\nasync function clearMistakes() {\n  if (!confirm(\'Clear ALL mistakes? Cannot be undone.\')) return;\n  await fetch(API + \'/api/mistakes/\' + getEffectiveSID() + \'/clear\', {method:\'POST\'});\n  showToast(\'Mistakes cleared\');\n  loadMistakes(\'\');\n}\n\n// ══════════════════════════════════════════════════════\n//  EMOTION HISTORY GRAPH\n// ══════════════════════════════════════════════════════\nvar _ehChart = null;\n\nasync function loadEmotionHistory() {\n  try {\n    var res  = await fetch(API + \'/api/emotion-history/\' + getEffectiveSID());\n    var data = await res.json();\n    renderEmotionHistory(data.history || []);\n  } catch(e) { console.warn(\'emotion history error\', e); }\n}\n\nfunction renderEmotionHistory(history) {\n  var el = $(\'eh-sessions\');\n  if (!history.length) {\n    if (el) el.innerHTML =\n      \'<div class="mn-empty">\'\n      + \'<div style="font-size:2.5rem">&#128247;</div>\'\n      + \'<div style="font-size:1rem;font-weight:700;margin:8px 0">No emotion data yet</div>\'\n      + \'<div style="color:var(--sub);font-size:.85rem">Complete quizzes with camera ON to see cognitive trends.</div>\'\n      + \'</div>\';\n    var cv = $(\'emotion-history-chart\');\n    if (cv) cv.parentElement.innerHTML =\n      \'<div style="color:var(--sub);text-align:center;padding:40px">Enable camera during quizzes to track cognitive history</div>\';\n    return;\n  }\n  var h = history.slice().reverse();\n  var labels = h.map(function(s,i){ return \'S\'+(i+1); });\n  if (_ehChart) { _ehChart.destroy(); _ehChart = null; }\n  var cv = $(\'emotion-history-chart\');\n  if (!cv) return;\n  _ehChart = new Chart(cv, {\n    type: \'line\',\n    data: {\n      labels: labels,\n      datasets: [\n        {label:\'Focused\',    data:h.map(function(s){return s.focused_pct;}),    borderColor:\'#10b981\',tension:.4,fill:false,pointRadius:4,borderWidth:2.5},\n        {label:\'Fatigued\',   data:h.map(function(s){return s.fatigued_pct;}),   borderColor:\'#94a3b8\',tension:.4,fill:false,pointRadius:4,borderWidth:2},\n        {label:\'Anxious\',    data:h.map(function(s){return s.anxious_pct;}),    borderColor:\'#ef4444\',tension:.4,fill:false,pointRadius:4,borderWidth:2},\n        {label:\'Overloaded\', data:h.map(function(s){return s.overloaded_pct;}), borderColor:\'#f97316\',tension:.4,fill:false,pointRadius:4,borderWidth:2},\n        {label:\'Processing\', data:h.map(function(s){return s.loading_pct;}),    borderColor:\'#f59e0b\',tension:.4,fill:false,pointRadius:4,borderWidth:2}\n      ]\n    },\n    options:{\n      responsive:true,maintainAspectRatio:false,\n      plugins:{\n        legend:{display:true,position:\'bottom\',labels:{color:\'rgba(255,255,255,.6)\',padding:14,font:{size:12}}},\n        tooltip:{callbacks:{label:function(c){return c.dataset.label+\': \'+c.raw+\'%\';}}}\n      },\n      scales:{\n        x:{ticks:{color:\'rgba(255,255,255,.4)\'},grid:{color:\'rgba(255,255,255,.05)\'}},\n        y:{beginAtZero:true,max:100,\n           ticks:{color:\'rgba(255,255,255,.4)\',callback:function(v){return v+\'%\';}},\n           grid:{color:\'rgba(255,255,255,.05)\'}}\n      }\n    }\n  });\n  if (el) {\n    var emojis = {focused:\'&#127919;\',fatigued:\'&#128564;\',anxious:\'&#129496;\',overloaded:\'&#129529;\',loading:\'&#129300;\'};\n    var colors = {focused:\'#10b981\',fatigued:\'#94a3b8\',anxious:\'#ef4444\',overloaded:\'#f97316\',loading:\'#f59e0b\'};\n    el.innerHTML = \'<h3 class="section-label" style="margin-top:14px">&#128197; Session Log</h3>\'\n      + history.map(function(s){\n          var emo = s.dominant || \'focused\';\n          return \'<div class="eh-card">\'\n            + \'<div style="display:flex;align-items:center;gap:12px">\'\n            + \'<div style="font-size:1.6rem">\' + (emojis[emo]||\'&#127919;\') + \'</div>\'\n            + \'<div><div style="font-weight:700;color:\' + (colors[emo]||\'#10b981\') + \'">\'\n            + emo.charAt(0).toUpperCase()+emo.slice(1) + \'</div>\'\n            + \'<div style="font-size:.75rem;color:var(--sub)">\' + escHtml(s.topic||\'Quiz\') + \'</div></div></div>\'\n            + \'<div style="text-align:right">\'\n            + \'<div style="font-size:.72rem;color:var(--sub)">\' + escHtml(s.quiz_date||\'\') + \'</div>\'\n            + \'<div style="font-size:.78rem;font-weight:600;color:#a78bfa">\' + s.total_q + \' questions</div>\'\n            + \'</div></div>\';\n        }).join(\'\');\n  }\n}\n\n// ══════════════════════════════════════════════════════\n//  AI TUTOR CHAT\n// ══════════════════════════════════════════════════════\nvar _chatLoading = false;\nvar _chatConcept = \'\';\nvar _chatEnterBound = false;\n\nfunction initChat() {\n  var ti = $(\'chat-topic-input\');\n  if (ti && !ti.value && state && state.topic) ti.value = state.topic;\n  var box = $(\'chat-box\');\n  if (box && box.children.length === 0) {\n    appendChatBubble(\'ai\',\n      \'\\u270b Hi! I am your AI Tutor. Ask me anything about your topic \\u2014 \'\n      + \'I can explain concepts, give examples, or generate practice questions!\');\n  }\n  if (!_chatEnterBound) {\n    _chatEnterBound = true;\n    var inp = $(\'chat-input\');\n    if (inp) {\n      inp.addEventListener(\'keydown\', function(e) {\n        if (e.key === \'Enter\' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }\n      });\n    }\n  }\n  var inp2 = $(\'chat-input\');\n  if (inp2) setTimeout(function(){ inp2.focus(); }, 80);\n}\n\nfunction askTutorAbout(concept, topic) {\n  _chatConcept = concept;\n  var ti = $(\'chat-topic-input\');\n  if (ti) ti.value = topic;\n  showScreen(\'chat\');\n  var inp = $(\'chat-input\');\n  if (inp) {\n    inp.value = \'Explain "\' + concept + \'" \\u2014 I got a question about it wrong\';\n    inp.focus();\n  }\n}\n\nfunction quickAsk(btn) {\n  var inp = $(\'chat-input\');\n  if (inp) { inp.value = btn.textContent.replace(/^[^\\w\\u{1F000}-\\u{1FFFF}]+/u,\'\').trim(); inp.focus(); }\n}\n\nasync function sendChatMessage() {\n  if (_chatLoading) return;\n  var inp  = $(\'chat-input\');\n  var msg  = inp ? inp.value.trim() : \'\';\n  if (!msg) return;\n  var ti   = $(\'chat-topic-input\');\n  var topic = (ti ? ti.value.trim() : \'\') || (state && state.topic ? state.topic : \'General\');\n  appendChatBubble(\'user\', msg);\n  if (inp) inp.value = \'\';\n  _chatLoading = true;\n  var btn = $(\'chat-send-btn\');\n  if (btn) { btn.disabled = true; btn.textContent = \'...\'; }\n  var typingId = \'typing-\' + Date.now();\n  appendChatBubble(\'typing\', \'\', typingId);\n  try {\n    var res  = await fetch(API + \'/api/chat\', {\n      method: \'POST\',\n      headers: {\'Content-Type\':\'application/json\'},\n      body: JSON.stringify({\n        session_id: getEffectiveSID(),\n        message:    msg,\n        topic:      topic,\n        concept:    _chatConcept\n      })\n    });\n    var data = await res.json();\n    var tel  = document.getElementById(typingId);\n    if (tel) tel.remove();\n    appendChatBubble(\'ai\', data.reply || (\'Error: \' + (data.error || \'Unknown\')));\n  } catch(e) {\n    var tel2 = document.getElementById(typingId);\n    if (tel2) tel2.remove();\n    appendChatBubble(\'ai\', \'Connection error \\u2014 please try again.\');\n  }\n  _chatLoading = false;\n  if (btn) { btn.disabled = false; btn.textContent = \'Send \\u25BA\'; }\n}\n\nfunction appendChatBubble(role, text, id) {\n  var box = $(\'chat-box\');\n  if (!box) return;\n  var div = document.createElement(\'div\');\n  if (id) div.id = id;\n  if (role === \'user\') {\n    div.className = \'chat-bubble chat-user\';\n    div.innerHTML = \'<span class="cb-lbl">You</span>\' + escHtml(text);\n  } else if (role === \'typing\') {\n    div.className = \'chat-bubble chat-ai chat-typing\';\n    div.innerHTML = \'<span class="cb-lbl">Tutor</span><span class="typing-dots">&#8226;&#8226;&#8226;</span>\';\n  } else {\n    div.className = \'chat-bubble chat-ai\';\n    var fmt = escHtml(text)\n      .replace(/\\*\\*(.*?)\\*\\*/g, \'<strong>$1</strong>\')\n      .replace(/\\n/g, \'<br>\');\n    div.innerHTML = \'<span class="cb-lbl">Tutor</span>\' + fmt;\n  }\n  box.appendChild(div);\n  box.scrollTop = box.scrollHeight;\n}\n\nasync function clearChat() {\n  await fetch(API + \'/api/chat/clear/\' + getEffectiveSID(), {method:\'POST\'});\n  var box = $(\'chat-box\');\n  if (box) box.innerHTML = \'\';\n  _chatConcept = \'\';\n  showToast(\'New chat started\');\n  initChat();\n}\n\n\n\n// ════════════════════════════════════════════════════════\n//  DRAGGABLE CAM PANEL  (mouse + touch)\n// ════════════════════════════════════════════════════════\n(function () {\n  \'use strict\';\n  var el = null, dragging = false;\n  var sx = 0, sy = 0, ox = 0, oy = 0;\n\n  function getPanel() { return document.getElementById(\'cam-panel\'); }\n\n  function snapToExplicit() {\n    var p = getPanel();\n    if (!p || p.style.left) return;\n    var r = p.getBoundingClientRect();\n    p.style.right  = \'auto\';\n    p.style.bottom = \'auto\';\n    p.style.left   = r.left + \'px\';\n    p.style.top    = r.top  + \'px\';\n  }\n\n  function onDown(cx, cy) {\n    el = getPanel();\n    if (!el) return;\n    snapToExplicit();\n    dragging = true;\n    var r = el.getBoundingClientRect();\n    sx = cx; sy = cy; ox = r.left; oy = r.top;\n    el.style.transition = \'none\';\n    el.style.right = \'auto\';\n    el.style.bottom = \'auto\';\n  }\n\n  function onMove(cx, cy) {\n    if (!dragging || !el) return;\n    var nx = Math.max(0, Math.min(window.innerWidth  - el.offsetWidth,  ox + (cx - sx)));\n    var ny = Math.max(0, Math.min(window.innerHeight - el.offsetHeight, oy + (cy - sy)));\n    el.style.left = nx + \'px\';\n    el.style.top  = ny + \'px\';\n  }\n\n  function onUp() { dragging = false; if (el) el.style.transition = \'\'; el = null; }\n\n  function bindTopbar() {\n    var p = getPanel();\n    if (!p || p._dragBound) return;\n    p._dragBound = true;\n    var tb = p.querySelector(\'.cam-panel-topbar\') || p;\n    tb.addEventListener(\'mousedown\', function(e) {\n      if (e.target.closest(\'button\')) return;\n      e.preventDefault();\n      onDown(e.clientX, e.clientY);\n    });\n    tb.addEventListener(\'touchstart\', function(e) {\n      if (e.target.closest(\'button\')) return;\n      onDown(e.touches[0].clientX, e.touches[0].clientY);\n    }, { passive: true });\n  }\n\n  document.addEventListener(\'mousemove\', function(e) { onMove(e.clientX, e.clientY); });\n  document.addEventListener(\'mouseup\',   onUp);\n  document.addEventListener(\'touchmove\', function(e) {\n    if (dragging) { e.preventDefault(); onMove(e.touches[0].clientX, e.touches[0].clientY); }\n  }, { passive: false });\n  document.addEventListener(\'touchend\', onUp);\n  document.addEventListener(\'DOMContentLoaded\', bindTopbar);\n\n  // Patch toggleCam to re-bind after camera turns on\n  document.addEventListener(\'DOMContentLoaded\', function() {\n    var _orig = window.toggleCam;\n    if (typeof _orig === \'function\') {\n      window.toggleCam = async function() {\n        await _orig.apply(this, arguments);\n        setTimeout(function() {\n          var p = getPanel();\n          if (p && !p.classList.contains(\'hidden\')) { bindTopbar(); snapToExplicit(); }\n        }, 200);\n      };\n    }\n  });\n})();\n\n\n\n/* ── Count-up animation for stat cells ── */\nfunction animateCountUp(el, target, duration=700) {\n  const isFloat = String(target).includes(\'.\');\n  const start = 0; const step = 16;\n  const steps = Math.round(duration / step);\n  let cur = 0;\n  const timer = setInterval(() => {\n    cur++;\n    const val = isFloat\n      ? (target * cur / steps).toFixed(1)\n      : Math.round(target * cur / steps);\n    el.textContent = val;\n    if (cur >= steps) { el.textContent = target; clearInterval(timer); }\n  }, step);\n}\nfunction runCountUps() {\n  document.querySelectorAll(\'.sv,.bs-val\').forEach(el => {\n    const raw = parseFloat(el.textContent.replace(/[^0-9.]/g,\'\'));\n    if (!isNaN(raw) && raw > 0) animateCountUp(el, raw);\n  });\n}\n// Run once on page load and whenever home/dashboard screen becomes active\nconst _origShowScreen = window.showScreen;\nif (typeof showScreen === \'function\') {\n  window._origShowScreen = showScreen;\n  window.showScreen = function(name) {\n    _origShowScreen(name);\n    if (name === \'home\' || name === \'dashboard\') setTimeout(runCountUps, 100);\n  };\n}\ndocument.addEventListener(\'DOMContentLoaded\', () => setTimeout(runCountUps, 300));\n\n\n// ── Change Password ──────────────────────────────────────────────\nfunction openCPW() {\n  const overlay = document.getElementById(\'cpw-overlay\');\n  if (!overlay) return;\n  // Reset form\n  [\'cpw-current\',\'cpw-new\',\'cpw-confirm\'].forEach(id => {\n    const el = document.getElementById(id);\n    if (el) { el.value = \'\'; el.type = \'password\'; }\n  });\n  const errEl = document.getElementById(\'cpw-error\');\n  const sucEl = document.getElementById(\'cpw-success\');\n  if (errEl) { errEl.textContent = \'\'; errEl.classList.remove(\'show\'); }\n  if (sucEl) { sucEl.textContent = \'\'; sucEl.classList.remove(\'show\'); }\n  const btn = document.getElementById(\'cpw-submit-btn\');\n  if (btn) { btn.disabled = false; btn.textContent = \'Update Password\'; }\n  overlay.classList.remove(\'hidden\');\n}\n\nfunction closeCPW() {\n  const overlay = document.getElementById(\'cpw-overlay\');\n  if (overlay) overlay.classList.add(\'hidden\');\n}\n\nasync function submitChangePassword() {\n  const current = document.getElementById(\'cpw-current\').value.trim();\n  const newPw   = document.getElementById(\'cpw-new\').value;\n  const confirm = document.getElementById(\'cpw-confirm\').value;\n  const errEl   = document.getElementById(\'cpw-error\');\n  const sucEl   = document.getElementById(\'cpw-success\');\n  const btn     = document.getElementById(\'cpw-submit-btn\');\n\n  errEl.classList.remove(\'show\'); errEl.textContent = \'\';\n  sucEl.classList.remove(\'show\'); sucEl.textContent = \'\';\n\n  if (!current || !newPw || !confirm) {\n    errEl.textContent = \'All fields are required.\';\n    errEl.classList.add(\'show\'); return;\n  }\n  if (newPw !== confirm) {\n    errEl.textContent = \'New passwords do not match.\';\n    errEl.classList.add(\'show\'); return;\n  }\n  if (newPw.length < 6) {\n    errEl.textContent = \'Password must be at least 6 characters.\';\n    errEl.classList.add(\'show\'); return;\n  }\n\n  btn.disabled = true;\n  btn.innerHTML = \'<span class="auth-btn-spinner"></span>Updating...\';\n\n  try {\n    const res  = await _origFetch(API + \'/api/auth/change-password\', {\n      method:  \'POST\',\n      headers: { \'Content-Type\': \'application/json\', ...authHeaders() },\n      body:    JSON.stringify({\n        current_password: current,\n        new_password:     newPw,\n        confirm_password: confirm\n      })\n    });\n    const data = await res.json();\n    if (!res.ok) throw new Error(data.error || \'Update failed\');\n    sucEl.textContent = \'✅ \' + data.message;\n    sucEl.classList.add(\'show\');\n    // Clear fields on success\n    [\'cpw-current\',\'cpw-new\',\'cpw-confirm\'].forEach(id => {\n      const el = document.getElementById(id);\n      if (el) el.value = \'\';\n    });\n    setTimeout(closeCPW, 2000);\n  } catch(e) {\n    errEl.textContent = e.message;\n    errEl.classList.add(\'show\');\n  }\n  btn.disabled = false;\n  btn.textContent = \'Update Password\';\n}\n\n// Close CPW on Escape key\ndocument.addEventListener(\'keydown\', function(e) {\n  if (e.key === \'Escape\') {\n    const cpw = document.getElementById(\'cpw-overlay\');\n    if (cpw && !cpw.classList.contains(\'hidden\')) closeCPW();\n  }\n});\n\n\n// ── Forgot Password\nlet _forgotEmail=\'\';\nfunction forgotGoBack(){\n  document.getElementById(\'forgot-step-1\').style.display=\'flex\';\n  document.getElementById(\'forgot-step-1\').classList.add(\'active\');\n  document.getElementById(\'forgot-step-2\').style.display=\'none\';\n  document.getElementById(\'forgot-step-2\').classList.remove(\'active\');\n  const e=document.getElementById(\'auth-error\');\n  if(e){e.textContent=\'\';e.classList.remove(\'show\');}\n}\nasync function forgotSendCode(){\n  const email=document.getElementById(\'forgot-email\').value.trim();\n  const errEl=document.getElementById(\'auth-error\');\n  const btn=document.getElementById(\'forgot-send-btn\');\n  errEl.classList.remove(\'show\');errEl.textContent=\'\';\n  if(!email||!email.includes(\'@\')){\n    errEl.textContent=\'Please enter a valid email.\';errEl.classList.add(\'show\');return;\n  }\n  btn.disabled=true;\n  btn.innerHTML=\'<span class="auth-btn-spinner"></span>Generating code...\';\n  try{\n    const res=await _origFetch(API+\'/api/auth/forgot-password\',{\n      method:\'POST\',headers:{\'Content-Type\':\'application/json\'},\n      body:JSON.stringify({email})\n    });\n    const data=await res.json();\n    if(!res.ok)throw new Error(data.error||\'Request failed\');\n    _forgotEmail=email;\n    document.getElementById(\'forgot-code-display\').textContent=data.token;\n    document.getElementById(\'forgot-token\').value=data.token;\n    document.getElementById(\'forgot-step-1\').style.display=\'none\';\n    document.getElementById(\'forgot-step-1\').classList.remove(\'active\');\n    document.getElementById(\'forgot-step-2\').style.display=\'flex\';\n    document.getElementById(\'forgot-step-2\').classList.add(\'active\');\n  }catch(e){\n    errEl.textContent=e.message;errEl.classList.add(\'show\');\n  }\n  btn.disabled=false;btn.textContent=\'Get Reset Code\';\n}\nasync function forgotResetPassword(){\n  const token=document.getElementById(\'forgot-token\').value.trim();\n  const newPw=document.getElementById(\'forgot-new-pw\').value;\n  const confirm=document.getElementById(\'forgot-confirm-pw\').value;\n  const errEl=document.getElementById(\'auth-error\');\n  const btn=document.getElementById(\'forgot-reset-btn\');\n  errEl.classList.remove(\'show\');errEl.textContent=\'\';\n  if(!token||!newPw||!confirm){\n    errEl.textContent=\'All fields are required.\';errEl.classList.add(\'show\');return;\n  }\n  if(newPw!==confirm){\n    errEl.textContent=\'Passwords do not match.\';errEl.classList.add(\'show\');return;\n  }\n  if(newPw.length<6){\n    errEl.textContent=\'Password must be at least 6 characters.\';errEl.classList.add(\'show\');return;\n  }\n  btn.disabled=true;\n  btn.innerHTML=\'<span class="auth-btn-spinner"></span>Updating password...\';\n  try{\n    const res=await _origFetch(API+\'/api/auth/reset-password\',{\n      method:\'POST\',headers:{\'Content-Type\':\'application/json\'},\n      body:JSON.stringify({email:_forgotEmail,token,new_password:newPw,confirm_password:confirm})\n    });\n    const data=await res.json();\n    if(!res.ok)throw new Error(data.error||\'Reset failed\');\n    switchAuthTab(\'login\');\n    const msg=document.createElement(\'div\');\n    msg.className=\'cpw-success show\';\n    msg.style.cssText=\'margin-bottom:12px;display:block\';\n    msg.textContent=\'Password updated successfully — please sign in\';\n    const lf=document.getElementById(\'auth-form-login\');\n    if(lf)lf.prepend(msg);\n    setTimeout(()=>msg.remove(),5000);\n  }catch(e){\n    errEl.textContent=e.message;errEl.classList.add(\'show\');\n  }\n  btn.disabled=false;btn.textContent=\'Set New Password\';\n}\n\n</script>\n\n<!-- ═══ AUTH OVERLAY ═══ -->\n<div id="auth-overlay" class="auth-overlay">\n  <div class="auth-card">\n\n    <!-- Brand -->\n    <div class="auth-brand">\n      <div class="auth-logo">🎓</div>\n      <div class="auth-brand-name">EduMate <span style="background:var(--g-brand);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">360</span></div>\n      <div class="auth-brand-sub">AI-Powered Cognitive Learning Platform</div>\n    </div>\n\n    <!-- Tab toggle -->\n    <div class="auth-tabs">\n      <button class="auth-tab active" data-auth-tab="login"\n              onclick="switchAuthTab(\'login\')">Sign In</button>\n      <button class="auth-tab" data-auth-tab="register"\n              onclick="switchAuthTab(\'register\')">Create Account</button>\n      <button class="auth-tab" data-auth-tab="forgot"\n              onclick="switchAuthTab(\'forgot\')">Forgot?</button>\n    </div>\n\n    <!-- Error banner (shared) -->\n    <div id="auth-error" class="auth-error"></div>\n\n    <!-- LOGIN FORM -->\n    <div id="auth-form-login" class="auth-form" style="display:flex;flex-direction:column;gap:14px">\n      <div class="auth-field">\n        <label class="auth-label">Email</label>\n        <input id="login-email" class="auth-input" type="email"\n               placeholder="you@example.com" autocomplete="email"/>\n      </div>\n      <div class="auth-field">\n        <label class="auth-label">Password</label>\n        <div class="auth-pw-wrap">\n          <input id="login-password" class="auth-input" type="password"\n                 placeholder="Enter your password" autocomplete="current-password"/>\n          <button class="auth-pw-toggle" onclick="togglePwVisibility(\'login-password\',this)"\n                  type="button">Show</button>\n        </div>\n      </div>\n      <button id="auth-login-btn" class="auth-btn" onclick="authLogin()">\n        Sign In\n      </button>\n      <div class="auth-footer">\n        Don\'t have an account?\n        <a href="#" onclick="switchAuthTab(\'register\');return false"\n           style="color:var(--indigo-l);font-weight:600;text-decoration:none">\n          Create one free\n        </a>\n      </div>\n    </div>\n\n    <!-- REGISTER FORM -->\n    <div id="auth-form-register" class="auth-form" style="display:none;flex-direction:column;gap:14px">\n      <div class="auth-field">\n        <label class="auth-label">Username</label>\n        <input id="reg-username" class="auth-input" type="text"\n               placeholder="Choose a username" autocomplete="username"/>\n      </div>\n      <div class="auth-field">\n        <label class="auth-label">Email</label>\n        <input id="reg-email" class="auth-input" type="email"\n               placeholder="you@example.com" autocomplete="email"/>\n      </div>\n      <div class="auth-field">\n        <label class="auth-label">Password</label>\n        <div class="auth-pw-wrap">\n          <input id="reg-password" class="auth-input" type="password"\n                 placeholder="At least 6 characters" autocomplete="new-password"/>\n          <button class="auth-pw-toggle" onclick="togglePwVisibility(\'reg-password\',this)"\n                  type="button">Show</button>\n        </div>\n      </div>\n      <div class="auth-field">\n        <label class="auth-label">Confirm Password</label>\n        <div class="auth-pw-wrap">\n          <input id="reg-confirm" class="auth-input" type="password"\n                 placeholder="Repeat your password" autocomplete="new-password"/>\n          <button class="auth-pw-toggle" onclick="togglePwVisibility(\'reg-confirm\',this)"\n                  type="button">Show</button>\n        </div>\n      </div>\n      <button id="auth-register-btn" class="auth-btn" onclick="authRegister()">\n        Create Account\n      </button>\n      <div class="auth-footer">\n        Already have an account?\n        <a href="#" onclick="switchAuthTab(\'login\');return false"\n           style="color:var(--indigo-l);font-weight:600;text-decoration:none">\n          Sign in\n        </a>\n      </div>\n    </div>\n\n    <!-- FORGOT PASSWORD FORM -->\n    <div id="auth-form-forgot" class="auth-form" style="display:none;flex-direction:column;gap:14px">\n      <div id="forgot-step-1" class="reset-step active" style="display:flex">\n        <div class="auth-field">\n          <label class="auth-label">Registered Email</label>\n          <input id="forgot-email" class="auth-input" type="email"\n                 placeholder="you@example.com" autocomplete="email"/>\n        </div>\n        <button id="forgot-send-btn" class="auth-btn" onclick="forgotSendCode()">Get Reset Code</button>\n        <div class="auth-footer">Remembered it?\n          <a href="#" onclick="switchAuthTab(\'login\');return false"\n             style="color:var(--indigo-l);font-weight:600;text-decoration:none">Sign in</a>\n        </div>\n      </div>\n      <div id="forgot-step-2" class="reset-step" style="display:none">\n        <div class="reset-code-card">\n          <div class="reset-code-label">Your Reset Code</div>\n          <div id="forgot-code-display" class="reset-code-value">- -</div>\n          <div class="reset-code-expiry">Valid for 15 minutes</div>\n        </div>\n        <div class="auth-field">\n          <label class="auth-label">Reset Code</label>\n          <input id="forgot-token" class="auth-input" type="text"\n                 placeholder="6-digit code" maxlength="6"\n                 style="letter-spacing:.2em;font-size:1.1rem;font-weight:700;text-align:center"/>\n        </div>\n        <div class="auth-field">\n          <label class="auth-label">New Password</label>\n          <div class="auth-pw-wrap">\n            <input id="forgot-new-pw" class="auth-input" type="password"\n                   placeholder="At least 6 characters"/>\n            <button class="auth-pw-toggle" onclick="togglePwVisibility(\'forgot-new-pw\',this)"\n                    type="button">Show</button>\n          </div>\n        </div>\n        <div class="auth-field">\n          <label class="auth-label">Confirm New Password</label>\n          <div class="auth-pw-wrap">\n            <input id="forgot-confirm-pw" class="auth-input" type="password"\n                   placeholder="Repeat new password"/>\n            <button class="auth-pw-toggle" onclick="togglePwVisibility(\'forgot-confirm-pw\',this)"\n                    type="button">Show</button>\n          </div>\n        </div>\n        <button id="forgot-reset-btn" class="auth-btn" onclick="forgotResetPassword()">Set New Password</button>\n        <div class="auth-footer">\n          <a href="#" onclick="forgotGoBack();return false"\n             style="color:var(--tx-2);text-decoration:none">Back to email</a>\n        </div>\n      </div>\n    </div>\n\n  </div>\n</div>\n\n\n<!-- ═══ CHANGE PASSWORD MODAL ═══ -->\n<div id="cpw-overlay" class="cpw-overlay hidden">\n  <div class="cpw-card">\n    <button class="cpw-close" onclick="closeCPW()">✕</button>\n    <div class="cpw-title">🔐 Change Password</div>\n    <div class="cpw-sub">Enter your current password and choose a new one.</div>\n    <div id="cpw-success" class="cpw-success"></div>\n    <div id="cpw-error"   class="auth-error"></div>\n    <div class="auth-form" style="gap:12px">\n      <div class="auth-field">\n        <label class="auth-label">Current Password</label>\n        <div class="auth-pw-wrap">\n          <input id="cpw-current" class="auth-input" type="password"\n                 placeholder="Your current password"/>\n          <button class="auth-pw-toggle"\n                  onclick="togglePwVisibility(\'cpw-current\',this)" type="button">Show</button>\n        </div>\n      </div>\n      <div class="auth-field">\n        <label class="auth-label">New Password</label>\n        <div class="auth-pw-wrap">\n          <input id="cpw-new" class="auth-input" type="password"\n                 placeholder="At least 6 characters"/>\n          <button class="auth-pw-toggle"\n                  onclick="togglePwVisibility(\'cpw-new\',this)" type="button">Show</button>\n        </div>\n      </div>\n      <div class="auth-field">\n        <label class="auth-label">Confirm New Password</label>\n        <div class="auth-pw-wrap">\n          <input id="cpw-confirm" class="auth-input" type="password"\n                 placeholder="Repeat new password"/>\n          <button class="auth-pw-toggle"\n                  onclick="togglePwVisibility(\'cpw-confirm\',this)" type="button">Show</button>\n        </div>\n      </div>\n      <button id="cpw-submit-btn" class="auth-btn" onclick="submitChangePassword()">\n        Update Password\n      </button>\n    </div>\n  </div>\n</div>\n\n</body>\n</html>'
@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html", headers={"Cache-Control":"no-store,no-cache,must-revalidate,max-age=0"})


@app.route("/api/health")
def health():
    return jsonify({"status":"ok","groq":bool(GROQ_API_KEY),"model":"llama-3.3-70b-versatile",
                    "ml":"GradientBoostingClassifier","pdf":PDF_AVAILABLE,"db":"sqlite","version":"7.0"})

@app.route("/version")
def version_check():
    return "EduMate360 v7.0 | build:20260315-110149 | OK", 200, {"Content-Type":"text/plain","Cache-Control":"no-store"}

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
    concept = d.get("concept","") or topic
    topic_guide = ai_topic_guide(topic, concept, d.get("question",""), d.get("correct_answer","")) if not correct else ""
    # Log for cognitive report
    if sid not in cog_log: cog_log[sid] = []
    cog_state = d.get("cog_state", "nocam")  # "nocam" = camera was off
    cog_log[sid].append({"emotion": cog_state, "topic": topic,
        "concept": concept, "correct": correct, "qIdx": d.get("q_idx", len(cog_log.get(sid,[])))  })
    # Auto-save wrong answers to Mistake Notebook
    if not correct:
        with sqlite3.connect(DB_PATH) as _mdb:
            _mdb.execute(
                """INSERT INTO mistakes
                   (session_id,topic,concept,question,your_answer,correct_answer,explanation,date,retried)
                   VALUES (?,?,?,?,?,?,?,?,0)""",
                (sid, topic, concept,
                 d.get("question",""),
                 d.get("student_answer",""),
                 d.get("correct_answer",""),
                 d.get("explanation",""),
                 datetime.now().strftime("%Y-%m-%d %H:%M")))
            _mdb.commit()
    new_badges = award_badges(sess)
    save_stats(sess)

    return jsonify({
        "correct": correct,
        "new_difficulty": ml["new_level"],
        "ml_action": ml["action"],
        "ml_reason": ml["reason"],
        "ml_confidence": ml["confidence"],
        "followup": followup,
        "topic_guide": topic_guide,
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
        "version": "7.0",
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


@app.route("/api/cognitive-report/<sid>")
def cognitive_report(sid):
    log = cog_log.get(sid, [])
    counts = {"focused":0,"loading":0,"overloaded":0,"fatigued":0,"anxious":0}
    # Map ALL possible values JS may send:
    # Fine-grained: focused|fatigued|loading|anxious|overloaded (sent via cogStateRaw)
    # Collapsed:    focused|confused(→loading)|stressed(→anxious) (fallback if cogStateRaw missing)
    _emap = {
        "focused":   "focused",
        "confused":  "loading",
        "stressed":  "anxious",
        "fatigued":  "fatigued",
        "loading":   "loading",
        "anxious":   "anxious",
        "overloaded":"overloaded",
    }
    cam_entries = 0
    for e in log:
        em = e.get("emotion","nocam")
        if em == "nocam": continue
        cam_entries += 1
        counts[_emap.get(em, "focused")] += 1
    cam_used = cam_entries > 0
    total = sum(counts.values()) or 1
    dominant = max(counts, key=counts.get) if cam_used else "nocam"
    focus_pct = round(counts.get("focused",0) / total * 100) if cam_used else 0
    # Per-topic wrong concepts
    per_topic = {}
    for e in log:
        if not e.get("correct"):
            t = e.get("topic","General")
            c = e.get("concept","") or t
            if t not in per_topic: per_topic[t] = []
            if c not in per_topic[t]: per_topic[t].append(c)
    # ── Full tips bank ────────────────────────────────────
    TIPS = {
        # ══════════════════════════════════════════════
        # 🎯 FOCUSED — sustain and maximise
        # ══════════════════════════════════════════════
        "focused": {
            "instant": [
                "🎯 You were in peak learning mode — this is exactly where you want to be!",
                "💧 Drink cold water every 20 min — dehydration silently kills prefrontal focus.",
                "⏱ You're in flow — don't break it. Silence notifications and keep going.",
                "🌬️ Take one slow deep breath right now — oxygen boosts sustained attention.",
                "🎵 Low-tempo instrumental music (60 BPM) deepens focused states further.",
            ],
            "study": [
                "📣 Teach each concept out loud to yourself — recall jumps 40% with active recall.",
                "✍️ After each correct answer, write WHY in one sentence — cements long-term memory.",
                "🔗 Link every new concept to something vivid you already know — builds memory scaffolding.",
                "📈 You are ready for harder difficulty — challenge yourself to Expert level.",
                "🔄 Use spaced repetition: revisit today's topics after 1 day, 3 days, then 7 days.",
                "📐 Use the Cornell method: notes on left, key questions on right, summary at bottom.",
            ],
            "score": [
                "🏆 Focus is excellent — now train speed. Set a 30-second timer per question.",
                "🎯 On timed exams, mark uncertain answers and return — don't let one block you.",
                "📊 Track which question types you miss most — MCQ, short answer, or concept-heavy.",
                "🧩 Attempt mixed-topic quizzes — interleaving improves exam performance by 43%.",
                "💡 Write practice exam questions yourself — constructing them boosts recall 2x.",
                "🔢 Use the 80/20 rule: 20% of concepts cover 80% of exam questions — find them.",
            ],
        },

        # ══════════════════════════════════════════════
        # 😴 FATIGUED — recovery + maintenance
        # ══════════════════════════════════════════════
        "fatigued": {
            "instant": [
                "😴 Mental fatigue detected — your hippocampus is full. Act on these NOW.",
                "💨 Physiological sigh: double-inhale through nose → long exhale through mouth. Repeat 3x.",
                "💦 Splash cold water on face and wrists — triggers the mammalian dive reflex (instant alert).",
                "🚶 Stand up immediately and walk for 2 minutes — movement releases norepinephrine.",
                "☀️ Look at a bright light or go near a window — light resets your circadian alertness.",
                "🍫 10g of dark chocolate (70%+ cocoa) — theobromine + small caffeine = gentle alertness.",
                "👁️ Do the 20-20-20 rule: look at something 20 feet away for 20 seconds — reduces eye fatigue.",
            ],
            "study": [
                "🚶 Pace while reviewing — movement boosts BDNF (brain-derived growth factor) by 30%.",
                "📇 When fatigued: flashcards ONLY — your brain can retrieve but not encode new info well.",
                "🔄 Switch to easier questions first — rebuild confidence and momentum.",
                "🎧 Listen to a podcast on the topic instead of reading — auditory learning uses less effort.",
                "📝 Write a 5-bullet summary of what you already know — activates long-term memory banks.",
                "🌿 Study in a slightly cool room (68°F/20°C) — cold temperatures maintain alertness.",
                "⏰ Use 10-minute micro-sessions — short bursts beat one long exhausted session every time.",
            ],
            "score": [
                "⚡ Power nap 10-20 min — NASA research: 20-min nap improves performance by 34%.",
                "🌙 7-8 hrs sleep tonight — your brain replays and consolidates today's learning during sleep.",
                "🎯 Schedule hardest topics for your PEAK alertness time (usually 10am-12pm or after a rest).",
                "🚫 Avoid screens 1 hr before sleep — blue light blocks melatonin and memory consolidation.",
                "📅 Don't cram — study fatigued material again tomorrow when fresh: retention triples.",
                "🧠 Eat a protein-rich snack (eggs, nuts, yoghurt) — tyrosine boosts dopamine and focus.",
                "🏃 20 minutes of exercise before your next session — proven to boost memory encoding by 20%.",
            ],
        },

        # ══════════════════════════════════════════════
        # 🧘 ANXIOUS — calm + ground + redirect
        # ══════════════════════════════════════════════
        "anxious": {
            "instant": [
                "🧘 Anxiety detected — you are in fight-or-flight. Let us fix that right now.",
                "📦 Box Breathing NOW: 4 sec inhale → 4 sec hold → 4 sec exhale → 4 sec hold. Do 3 cycles.",
                "💪 Unclench your jaw. Drop your shoulders. Relax your hands. Anxiety lives in posture.",
                "🖐️ Name 5 things you can see, 4 you can touch, 3 you can hear — grounding technique.",
                "🌊 Progressive muscle relaxation: tense each muscle group for 5 sec, then release.",
                "😄 Say 'I am EXCITED!' out loud — excitement and anxiety are the same physiology, reframed.",
                "🌬️ Exhale twice as long as you inhale (4 sec in, 8 sec out) — activates parasympathetic.",
            ],
            "study": [
                "✏️ Write your worries on paper for 2 minutes — physically offloads them from working memory.",
                "🟢 Start your next quiz with the EASIEST question — one win stops the anxiety spiral.",
                "🧠 Use 'I already know X' self-talk before each topic — activates existing knowledge networks.",
                "👥 Explain concepts to a study partner — social learning reduces exam anxiety by 28%.",
                "📋 Make a checklist of what you HAVE covered — anxiety comes from focus on what is missing.",
                "🎯 Focus on the process, not the outcome — 'I will try my best' beats 'I must score 90%'.",
                "📖 Re-read your correct answers first — builds confidence before tackling difficult areas.",
            ],
            "score": [
                "🦸 Power pose 2 min before an exam: stand tall, hands on hips — cortisol drops 25%.",
                "✅ Every wrong answer is a cheat code — it tells you exactly where to focus.",
                "⏱ On timed exams: skip anxiety-inducing questions first, collect easy marks, return later.",
                "💬 Positive self-talk script: 'I have prepared. I know this. I will recall it.' — say it.",
                "🧊 Hold ice cube for 10 sec (if anxious during exam) — sharp sensation resets the nervous system.",
                "📊 Lower stakes first: practice with 3-question quizzes before full 15-question sessions.",
                "🌙 Anxiety peaks when sleep-deprived — even 30 extra mins of sleep reduces exam anxiety 40%.",
            ],
        },

        # ══════════════════════════════════════════════
        # 🧩 OVERLOADED — declutter + restructure
        # ══════════════════════════════════════════════
        "overloaded": {
            "instant": [
                "🧩 Cognitive overload detected — your working memory (4 slots) is completely full.",
                "✋ STOP reading. Write ONE sentence: what is the single most important idea right now?",
                "🗺️ Draw a quick mind map on paper — visual processing uses a different brain pathway.",
                "🔇 Remove ALL distractions: close tabs, silence phone, clear desk — reduce cognitive load.",
                "💨 Take 5 slow breaths — overload triggers shallow breathing which worsens it.",
                "🌊 Look at something distant (outside a window) for 30 sec — panoramic vision calms amygdala.",
                "⏸️ Take a full 5-minute break away from the screen — mandatory, not optional.",
            ],
            "study": [
                "📦 Chunk the topic into MAX 4 sub-topics — working memory holds precisely 4 items.",
                "🔀 Interleave 2 topics every 15 min — prevents the interference that causes overload.",
                "👶 Feynman Technique: explain the concept as if teaching a 5-year-old. Gaps become obvious.",
                "📝 Use the 'brain dump' method: write everything you know about a topic without stopping.",
                "📐 Use diagrams, flowcharts, and tables instead of prose — reduces cognitive load by 40%.",
                "🎯 Study one sub-topic per session — depth beats breadth when overloaded.",
                "🔊 Read key points aloud — dual-coding (visual + auditory) reinforces without overloading.",
            ],
            "score": [
                "⏭️ Skip hard questions immediately and return — fresh neural pathways solve what overloaded ones can't.",
                "🌙 Sleep is NOT optional — overloaded material consolidates to long-term memory ONLY during sleep.",
                "📚 Study prerequisites before advanced topics — overload usually means a foundational gap.",
                "📅 Distribute learning: 3 sessions of 20 min beat 1 session of 60 min for overloaded topics.",
                "🔢 Prioritise: identify the 3 most likely exam concepts — master those before the rest.",
                "🧪 Do practice questions BEFORE re-reading — retrieval shows what is truly missing.",
                "📊 After each session, rate your understanding 1-5 per sub-topic — focus on 1s and 2s.",
            ],
        },

        # ══════════════════════════════════════════════
        # 🤔 LOADING / PROCESSING — deepen + leverage
        # ══════════════════════════════════════════════
        "loading": {
            "instant": [
                "🤔 Deep processing detected — your brain is forming new neural connections. This is GREAT.",
                "🎵 Hum quietly while thinking — activates the vagus nerve and boosts cognitive throughput.",
                "⏳ Give yourself thinking time — slow processing now means durable memory later.",
                "🌿 If stuck, look away from the screen for 20 sec — default mode network solves problems.",
                "🚶 Slow walk while thinking — rhythmic movement synchronises brainwaves for problem solving.",
                "✏️ Write the question in your own words before answering — clarifies processing load.",
            ],
            "study": [
                "🔗 Associate the new concept with something vivid and personal you already know well.",
                "🔤 Create a memorable mnemonic — acronyms and rhymes boost recall 3x over plain reading.",
                "❓ Ask 'why' five times about each concept — deep causal understanding beats surface memorisation.",
                "📝 When processing slows, switch to writing bullet points — externalises working memory.",
                "🗣️ Explain your reasoning out loud — metacognitive narration improves accuracy by 22%.",
                "🔄 Attempt the question, then read the explanation — prediction + feedback is the strongest loop.",
                "📖 Slow down: read 30% slower with full attention rather than fast with low retention.",
            ],
            "score": [
                "💡 Trust your first instinct on MCQs — second-guessing drops accuracy by 15% on average.",
                "⏱ Time-box your thinking: 90 sec max per question, then commit — trains exam pacing.",
                "🧩 Break complex questions into sub-parts: identify what is given, what is asked, what links them.",
                "📊 In exams, easier questions first — quick wins free up processing capacity for hard ones.",
                "✍️ Show your reasoning in written answers — partial credit often available for process.",
                "🎯 After each quiz, identify the ONE concept that confused you most — that is your next focus.",
                "📈 Processing deeply now means you will recall this under pressure — exam performance follows.",
            ],
        },
    }


    # ── Mixed-state tip blending ────────────────────────────
    # Find top 2 states with >15% presence (if camera was on)
    active_states = sorted(
        [(k, round(v/total*100)) for k,v in counts.items() if v > 0],
        key=lambda x: -x[1]
    )
    top_states = [s for s in active_states if s[1] >= 15][:2]

    # Define correct_count FIRST before using it
    correct_count = sum(1 for e in log if e.get("correct"))

    tips_payload = []
    if cam_used and top_states:
        for state_name, pct in top_states:
            state_tips = TIPS.get(state_name, TIPS["focused"])
            tips_payload.append({
                "state": state_name,
                "pct": pct,
                "instant":  state_tips["instant"],
                "study":    state_tips["study"],
                "score":    state_tips["score"]
            })
    else:
        # Camera off — give general performance tips based on accuracy
        acc = round(correct_count / len(log) * 100) if log else 0
        if acc >= 80:
            tips_payload.append({"state":"focused","pct":100,
                "instant": TIPS["focused"]["instant"],
                "study":   TIPS["focused"]["study"],
                "score":   TIPS["focused"]["score"]})
        else:
            tips_payload.append({"state":"loading","pct":100,
                "instant": ["💡 Enable camera next time for personalised mental state tips."],
                "study":   TIPS["loading"]["study"],
                "score":   ["📚 Review the weak topics in your Study Plan above."]})

    # Save emotion history for graph feature
    if cam_used and log:
        try:
            with sqlite3.connect(DB_PATH) as _edb:
                _edb.execute(
                    """INSERT INTO emotion_history
                       (session_id,quiz_date,dominant,focused_pct,fatigued_pct,
                        anxious_pct,overloaded_pct,loading_pct,total_q,topic)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (sid,
                     datetime.now().strftime("%Y-%m-%d %H:%M"),
                     dominant,
                     round(counts.get("focused",0)   / total * 100),
                     round(counts.get("fatigued",0)  / total * 100),
                     round(counts.get("anxious",0)   / total * 100),
                     round(counts.get("overloaded",0)/ total * 100),
                     round(counts.get("loading",0)   / total * 100),
                     len(log),
                     log[0].get("topic","") if log else ""))
                _edb.commit()
        except Exception:
            pass  # Never crash cognitive report due to history save
    cog_log.pop(sid, None)  # Clear after reading
    return jsonify({
        "states": {k: round(v/total*100) for k,v in counts.items()},
        "dominant": dominant,
        "focus_pct": focus_pct,
        "cam_used": cam_used,
        "per_topic": per_topic,
        "total_questions": len(log),
        "correct_count": correct_count,
        "tips": tips_payload
    })

# ════════════════════════════════════════════════════════
#  MISTAKE NOTEBOOK
# ════════════════════════════════════════════════════════

@app.route("/api/mistakes/<sid>")
def get_mistakes(sid):
    topic_filter = request.args.get("topic", "")
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        if topic_filter:
            rows = db.execute(
                "SELECT * FROM mistakes WHERE session_id=? AND topic=? ORDER BY date DESC",
                (sid, topic_filter)).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM mistakes WHERE session_id=? ORDER BY date DESC",
                (sid,)).fetchall()
        mistakes = [dict(r) for r in rows]
        topic_rows = db.execute(
            "SELECT DISTINCT topic FROM mistakes WHERE session_id=? ORDER BY topic",
            (sid,)).fetchall()
        topics = [t[0] for t in topic_rows]
    return jsonify({"mistakes": mistakes, "topics": topics})


@app.route("/api/mistakes/<sid>/retry/<int:mid>", methods=["POST"])
def retry_mistake(sid, mid):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT * FROM mistakes WHERE id=? AND session_id=?",
            (mid, sid)).fetchone()
        if not row:
            return jsonify({"error": "not found"}), 404
        db.execute("UPDATE mistakes SET retried=1 WHERE id=?", (mid,))
        db.commit()
    return jsonify({
        "ok": True,
        "question":  row["question"],
        "answer":    row["correct_answer"],
        "topic":     row["topic"],
        "concept":   row["concept"]
    })


@app.route("/api/mistakes/<sid>/clear", methods=["POST"])
def clear_mistakes(sid):
    with sqlite3.connect(DB_PATH) as db:
        db.execute("DELETE FROM mistakes WHERE session_id=?", (sid,))
        db.commit()
    return jsonify({"ok": True})


# ════════════════════════════════════════════════════════
#  EMOTION HISTORY
# ════════════════════════════════════════════════════════

@app.route("/api/emotion-history/<sid>")
def emotion_history_route(sid):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT * FROM emotion_history WHERE session_id=? ORDER BY quiz_date DESC LIMIT 20",
            (sid,)).fetchall()
        history = [dict(r) for r in rows]
    return jsonify({"history": history})


# ════════════════════════════════════════════════════════
#  AI TUTOR CHAT  (uses module-level _groq_client)
# ════════════════════════════════════════════════════════

_chat_sessions = {}  # {session_id: [{"role":..,"content":..}]}

@app.route("/api/chat", methods=["POST"])
def ai_chat():
    d       = request.json or {}
    sid     = d.get("session_id", "default")
    msg     = d.get("message", "").strip()
    topic   = d.get("topic", "General")
    concept = d.get("concept", "")
    if not msg:
        return jsonify({"error": "empty message"}), 400
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return jsonify({"error": "AI not available — check GROQ_API_KEY"}), 503

    system_prompt = (
        "You are a friendly, encouraging AI tutor inside EduMate 360. "
        f"The student is studying: {topic}. "
        + (f"They recently struggled with: {concept}. " if concept else "") +
        "Rules: explain simply with real-world analogies, guide don't just give answers, "
        "keep responses to 3-5 sentences unless asked for more, be encouraging."
    )

    # Build or restore session
    if sid not in _chat_sessions:
        # Try to recover last 10 messages from DB
        recovered = []
        try:
            with sqlite3.connect(DB_PATH) as _rdb:
                _rdb.row_factory = sqlite3.Row
                rows = _rdb.execute(
                    "SELECT role, content FROM chat_history "
                    "WHERE session_id=? ORDER BY id DESC LIMIT 10",
                    (sid,)).fetchall()
                recovered = [{"role": r["role"], "content": r["content"]}
                             for r in reversed(rows)]
        except Exception:
            pass
        _chat_sessions[sid] = [{"role": "system", "content": system_prompt}] + recovered
    else:
        _chat_sessions[sid][0] = {"role": "system", "content": system_prompt}

    _chat_sessions[sid].append({"role": "user", "content": msg})
    # Keep system prompt + last 10 messages max
    if len(_chat_sessions[sid]) > 11:
        _chat_sessions[sid] = [_chat_sessions[sid][0]] + _chat_sessions[sid][-10:]

    try:
        # Use the module-level _groq_client
        global _groq_client
        if not _groq_client:
            _groq_client = Groq(api_key=GROQ_API_KEY)
        resp = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=_chat_sessions[sid],
            max_tokens=400,
            temperature=0.7
        )
        reply = resp.choices[0].message.content.strip()
        _chat_sessions[sid].append({"role": "assistant", "content": reply})

        # Persist to DB
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        with sqlite3.connect(DB_PATH) as db:
            db.execute(
                "INSERT INTO chat_history (session_id,role,content,timestamp) VALUES (?,?,?,?)",
                (sid, "user", msg, now_str))
            db.execute(
                "INSERT INTO chat_history (session_id,role,content,timestamp) VALUES (?,?,?,?)",
                (sid, "assistant", reply, now_str))
            db.commit()
        return jsonify({"reply": reply})
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


@app.route("/api/chat/clear/<sid>", methods=["POST"])
def clear_chat(sid):
    _chat_sessions.pop(sid, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
