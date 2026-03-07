'use strict';
// ═══════════════════════════════════════════════════════
//  EduMate 360 v5.3 — Clean rewrite, zero crashes
//  Features: AI Quiz, ML Adaptive, Charts, Spaced Rep,
//            Leaderboard, Badges, PDF, Voice
// ═══════════════════════════════════════════════════════

const API = window.location.origin;
// Unique session per browser tab — stays consistent across quiz
const SESSION_ID = localStorage.getItem('em360_sid') || (function(){
  const id = 's_' + Math.random().toString(36).slice(2,10);
  try { localStorage.setItem('em360_sid', id); } catch(e){}
  return id;
})();

// ── CHART INSTANCES (kept so we can destroy before redraw)
const charts = { progress: null, pie: null, mastery: null };

// ── APP STATE
const state = {
  questions: [], qIdx: 0,
  topic: '', difficulty: 'medium',
  score: 0, xp: 0, level: 1,
  streak: 0, totalQ: 0, correctQ: 0,
  timerStart: 0, timerInterval: null,
  voiceEnabled: false,
  topics_studied: [], study_dates: [], badges: []
};

// ── HELPERS
function $(id){ return document.getElementById(id); }

function showToast(msg, dur) {
  const t = $('toast');
  if (!t) return;
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add('hidden'), dur || 3500);
}

function setLoading(on, msg) {
  const el = $('loading');
  if (!el) return;
  if (msg) $('loading-msg').textContent = msg;
  el.classList.toggle('hidden', !on);
}

function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const screen = $('screen-' + name);
  if (screen) screen.classList.add('active');
  const btn = document.querySelector('[data-screen="' + name + '"]');
  if (btn) btn.classList.add('active');
  if (name === 'dashboard') renderDashboard();
  if (name === 'leaderboard') loadLeaderboard();
  if (name === 'review') loadReview();
}

function switchTab(showId, hideId, btn) {
  $(showId).classList.remove('hidden');
  $(hideId).classList.add('hidden');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function calcDayStreak(dates) {
  if (!dates || !dates.length) return 0;
  const sorted = [...new Set(dates)].sort().reverse();
  let streak = 0;
  let cur = new Date(); cur.setHours(0,0,0,0);
  for (const d of sorted) {
    const dd = new Date(d); dd.setHours(0,0,0,0);
    const diff = (cur - dd) / 86400000;
    if (diff <= 1) { streak++; cur = dd; }
    else break;
  }
  return streak;
}

// ── UPDATE HOME STATS BAR
function updateHomeStats() {
  $('hs-streak').textContent = state.streak;
  $('hs-score').textContent = state.score;
  $('hs-acc').textContent = state.totalQ > 0
    ? Math.round(state.correctQ / state.totalQ * 100) + '%' : '0%';
  $('hs-topics').textContent = state.topics_studied.length;
  $('hs-days').textContent = calcDayStreak(state.study_dates);
  $('xp-num').textContent = state.xp;
  $('lv-num').textContent = state.level;
  const xpInLevel = state.xp % 200;
  $('xp-bar').style.width = (xpInLevel / 200 * 100) + '%';
  $('xp-to').textContent = (200 - xpInLevel) + ' XP to Level ' + (state.level + 1);
  if (state.badges && state.badges.length > 0) {
    $('home-badges').classList.remove('hidden');
    $('home-badge-list').innerHTML = state.badges
      .map(b => '<span class="badge">' + b + '</span>').join('');
  }
}

// ── CHECK DUE TOPICS ON HOME LOAD
async function checkDueTopics() {
  try {
    const res = await fetch(API + '/api/due-topics/' + SESSION_ID);
    if (!res.ok) return;
    const data = await res.json();
    if (data.due && data.due.length > 0) {
      const a = $('due-alert');
      a.innerHTML = '&#9201; <strong>' + data.due.length + ' topic(s) due for review:</strong> '
        + data.due.map(d => d.topic).join(', ')
        + ' <button onclick="showScreen(\'review\')" class="due-btn">Review Now</button>';
      a.classList.remove('hidden');
    }
  } catch(e) { /* silent */ }
}

// ── INIT
document.addEventListener('DOMContentLoaded', function() {
  // Wire nav buttons
  document.querySelectorAll('.nav-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      showScreen(btn.getAttribute('data-screen'));
    });
  });
  // Enter key on topic input
  var topicInput = $('topic-input');
  if (topicInput) {
    topicInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') generateQuestions();
    });
  }
  updateHomeStats();
  checkDueTopics();
});


// ═══════════════════════════════════════════════════════
//  QUESTION GENERATION
// ═══════════════════════════════════════════════════════

async function generateQuestions() {
  const topicEl = $('topic-input');
  const topic = topicEl ? topicEl.value.trim() : '';
  const diff = $('diff-select').value;
  const numQ = parseInt($('numq-select').value) || 5;
  const qType = $('type-select').value;

  if (!topic) { showToast('Please enter a topic first!'); topicEl && topicEl.focus(); return; }

  const errEl = $('gen-error');
  errEl.classList.add('hidden');
  $('generate-btn').disabled = true;
  setLoading(true, 'AI generating ' + numQ + ' ' + qType + ' questions on "' + topic + '"...');

  try {
    const res = await fetch(API + '/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic, difficulty: diff, num_questions: numQ,
        type: qType, session_id: SESSION_ID
      })
    });
    const data = await res.json();
    if (!res.ok || !data.questions || !data.questions.length) {
      throw new Error(data.error || 'No questions returned. Check your Groq API key on Render.');
    }
    state.questions = data.questions;
    state.topic = topic;
    state.difficulty = diff;
    state.qIdx = 0;
    if (!state.topics_studied.includes(topic)) state.topics_studied.push(topic);
    setLoading(false);
    $('generate-btn').disabled = false;
    showScreen('quiz');
    renderQuestion();
  } catch(err) {
    setLoading(false);
    $('generate-btn').disabled = false;
    errEl.textContent = '❌ ' + err.message;
    errEl.classList.remove('hidden');
  }
}

async function generateFromNotes() {
  const text = $('notes-input').value.trim();
  const diff = $('upload-diff').value;
  const numQ = parseInt($('upload-numq').value) || 5;
  const errEl = $('upload-error');
  errEl.classList.add('hidden');
  if (text.length < 50) { showToast('Please paste at least 50 characters of text'); return; }
  setLoading(true, 'Reading your notes and generating questions...');
  try {
    const res = await fetch(API + '/api/generate-from-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, difficulty: diff, num_questions: numQ, session_id: SESSION_ID })
    });
    const data = await res.json();
    if (!res.ok || !data.questions || !data.questions.length) throw new Error(data.error || 'Failed');
    state.questions = data.questions;
    state.topic = data.topic || 'Notes';
    state.difficulty = diff;
    state.qIdx = 0;
    setLoading(false);
    showScreen('quiz');
    renderQuestion();
  } catch(err) {
    setLoading(false);
    errEl.textContent = '❌ ' + err.message;
    errEl.classList.remove('hidden');
  }
}

function handlePDFSelect(input) {
  const file = input.files[0];
  if (file) {
    $('pdf-name').textContent = '📄 ' + file.name;
    $('pdf-btn').disabled = false;
  }
}

async function generateFromPDF() {
  const file = $('pdf-file').files[0];
  if (!file) { showToast('Please select a PDF file'); return; }
  const diff = $('pdf-diff').value;
  const numQ = parseInt($('pdf-numq').value) || 5;
  const errEl = $('pdf-error');
  errEl.classList.add('hidden');
  setLoading(true, 'Reading PDF and generating questions...');
  try {
    const fd = new FormData();
    fd.append('pdf', file);
    fd.append('difficulty', diff);
    fd.append('num_questions', numQ);
    fd.append('session_id', SESSION_ID);
    const res = await fetch(API + '/api/upload-pdf', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.questions || !data.questions.length) throw new Error(data.error || 'Failed');
    showToast('Read ' + (data.pages_read || '?') + ' pages → ' + data.total + ' questions!');
    state.questions = data.questions;
    state.topic = data.topic || 'PDF';
    state.difficulty = diff;
    state.qIdx = 0;
    setLoading(false);
    showScreen('quiz');
    renderQuestion();
  } catch(err) {
    setLoading(false);
    errEl.textContent = '❌ ' + err.message;
    errEl.classList.remove('hidden');
  }
}


// ═══════════════════════════════════════════════════════
//  QUIZ RENDERING
// ═══════════════════════════════════════════════════════

function renderQuestion() {
  const q = state.questions[state.qIdx];
  if (!q) return;
  const tot = state.questions.length;

  // Header
  $('q-topic-badge').textContent = state.topic;
  $('q-diff-badge').textContent = state.difficulty;
  $('q-diff-badge').className = 'q-diff-badge diff-' + state.difficulty;
  $('q-num').textContent = 'Q ' + (state.qIdx + 1) + '/' + tot;
  $('qprog').style.width = ((state.qIdx + 1) / tot * 100) + '%';

  // Question
  $('q-concept').textContent = q.concept || '';
  $('q-text').textContent = q.question;

  // Bg colour by difficulty
  const bg = {
    beginner:'rgba(14,165,233,0.08)', easy:'rgba(16,185,129,0.06)',
    medium:'rgba(245,158,11,0.07)',   hard:'rgba(239,68,68,0.07)',
    expert:'rgba(139,92,246,0.09)'
  };
  $('q-box').style.background = bg[state.difficulty] || bg.medium;

  // Live strip
  $('ls-score').textContent = state.score;
  $('ls-streak').textContent = state.streak;
  $('ls-acc').textContent = state.totalQ > 0
    ? Math.round(state.correctQ / state.totalQ * 100) + '%' : '0%';
  $('ml-tag').textContent = 'ML: ' + state.difficulty;

  // Hide feedback, reset timer
  $('feedback-wrap').classList.add('hidden');
  $('options-wrap').classList.add('hidden');
  $('short-wrap').classList.add('hidden');
  startTimer();

  // Voice
  if (state.voiceEnabled) speak(q.question);

  // Render inputs
  const type = q.type || 'mcq';
  if (type === 'short') {
    $('short-input').value = '';
    $('short-wrap').classList.remove('hidden');
  } else {
    const opts = q.options || [];
    $('options-wrap').innerHTML = opts.map(function(opt, i) {
      return '<button class="opt-btn" onclick="handleMCQ(this,' + i + ')">'
        + escHtml(opt) + '</button>';
    }).join('');
    $('options-wrap').classList.remove('hidden');
  }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function startTimer() {
  clearInterval(state.timerInterval);
  state.timerStart = Date.now();
  state.timerInterval = setInterval(function() {
    const secs = Math.round((Date.now() - state.timerStart) / 1000);
    $('timer-badge').textContent = '⏱ ' + secs + 's';
  }, 1000);
}

function toggleVoice() {
  state.voiceEnabled = !state.voiceEnabled;
  $('voice-btn').textContent = state.voiceEnabled ? '🔊 On' : '🔇 Voice';
  $('voice-btn').classList.toggle('active', state.voiceEnabled);
  if (state.voiceEnabled && state.questions[state.qIdx]) {
    speak(state.questions[state.qIdx].question);
  }
}

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.9;
  window.speechSynthesis.speak(u);
}

async function handleMCQ(btn, optIdx) {
  clearInterval(state.timerInterval);
  const timeSec = (Date.now() - state.timerStart) / 1000;
  const q = state.questions[state.qIdx];
  const opts = q.options || [];
  const selected = opts[optIdx];
  const correct = selected === q.answer;

  // Disable all options, highlight correct/wrong
  document.querySelectorAll('.opt-btn').forEach(function(b, i) {
    b.disabled = true;
    if (opts[i] === q.answer) b.classList.add('correct');
    else if (b === btn && !correct) b.classList.add('wrong');
  });

  await submitAnswer(correct, selected, q, timeSec);
}

async function submitShort() {
  clearInterval(state.timerInterval);
  const timeSec = (Date.now() - state.timerStart) / 1000;
  const q = state.questions[state.qIdx];
  const given = ($('short-input').value || '').trim().toLowerCase();
  const correct_lower = (q.answer || '').toLowerCase();
  // Match if answer contains key words from correct answer
  const keywords = correct_lower.split(/\s+/).filter(w => w.length > 4);
  const correct = keywords.length > 0
    ? keywords.some(k => given.includes(k))
    : given.includes(correct_lower.slice(0, 8));
  await submitAnswer(correct, given, q, timeSec);
}

async function submitAnswer(correct, studentAns, q, timeSec) {
  // Update local state immediately (optimistic)
  state.totalQ++;
  state.timerStart = state.timerStart; // keep reference
  if (correct) {
    state.correctQ++;
    state.streak++;
  } else {
    state.streak = 0;
  }

  // Show feedback right away
  const fm = $('fb-msg');
  fm.textContent = correct ? '✅ Correct!' : '❌ Wrong — Answer: ' + q.answer;
  fm.className = 'fb-msg ' + (correct ? 'correct-fb' : 'wrong-fb');
  $('fb-exp').innerHTML = '<strong style="color:#a78bfa">Explanation:</strong> '
    + escHtml(q.explanation || '');
  $('fb-followup').classList.add('hidden');
  $('ml-explain').textContent = '';
  $('feedback-wrap').classList.remove('hidden');

  // Update live strip immediately
  $('ls-score').textContent = state.score;
  $('ls-streak').textContent = state.streak;
  $('ls-acc').textContent = Math.round(state.correctQ / state.totalQ * 100) + '%';

  // Submit to backend
  try {
    const res = await fetch(API + '/api/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: SESSION_ID,
        correct: correct,
        time_sec: timeSec,
        topic: state.topic,
        question: q.question,
        student_answer: studentAns,
        correct_answer: q.answer,
        explanation: q.explanation || ''
      })
    });
    if (!res.ok) return;
    const data = await res.json();

    // Sync state from server response
    state.score    = data.stats.score;
    state.xp       = data.stats.xp;
    state.level    = data.stats.level;
    state.streak   = data.stats.streak;
    state.difficulty = data.new_difficulty;
    if (data.stats.badges) state.badges = data.stats.badges;

    // Update live strip with server values
    $('ls-score').textContent = state.score;
    $('ls-streak').textContent = state.streak;
    $('ls-acc').textContent = data.stats.accuracy + '%';
    $('ml-tag').textContent = 'ML: ' + data.new_difficulty;

    // Follow-up question
    if (data.followup) {
      $('fq-text').textContent = data.followup;
      $('fb-followup').classList.remove('hidden');
    }

    // ML explanation
    $('ml-explain').textContent = '🤖 ' + data.ml_reason
      + ' (confidence: ' + Math.round((data.ml_confidence || 0) * 100) + '%)';

    // Badges
    if (data.new_badges && data.new_badges.length > 0) {
      data.new_badges.forEach(function(b) { showToast('🏅 New badge: ' + b, 4000); });
    }

    // Difficulty change toast
    if (data.ml_action === 'harder') showToast('📈 Difficulty increased → ' + data.new_difficulty);
    else if (data.ml_action === 'easier') showToast('📉 Difficulty reduced → ' + data.new_difficulty);

    updateHomeStats();
  } catch(e) {
    // Backend error — local state is still correct, just no server sync
    console.warn('Submit error:', e.message);
  }
}

function nextQuestion() {
  state.qIdx++;
  if (state.qIdx >= state.questions.length) {
    // Quiz complete
    showToast('🎉 Quiz complete! Great work!', 3000);
    updateHomeStats();
    setTimeout(function() { showScreen('dashboard'); }, 1500);
    return;
  }
  renderQuestion();
}


// ═══════════════════════════════════════════════════════
//  DASHBOARD — Charts + Stats
// ═══════════════════════════════════════════════════════

function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); charts[key] = null; }
}

async function renderDashboard() {
  try {
    const res = await fetch(API + '/api/stats/' + SESSION_ID);
    if (!res.ok) return;
    const data = await res.json();

    // Sync state with server
    state.score  = data.score  || 0;
    state.xp     = data.xp    || 0;
    state.level  = data.level || 1;
    state.streak = data.streak || 0;
    state.totalQ   = data.total_q   || 0;
    state.correctQ = data.correct_q || 0;
    state.topics_studied = data.topics_studied || [];
    state.study_dates    = data.study_dates    || [];
    state.badges         = data.badges         || [];
    updateHomeStats();

    const acc = data.total_q > 0
      ? Math.round(data.correct_q / data.total_q * 100) : 0;

    // Big stats
    $('big-stats').innerHTML = [
      {i:'⚡',l:'XP',        v:data.xp,      c:'#a78bfa'},
      {i:'🏆',l:'Level',     v:data.level,   c:'#f59e0b'},
      {i:'🔥',l:'Streak',    v:data.streak,  c:'#f97316'},
      {i:'⭐',l:'Score',     v:data.score,   c:'#6366f1'},
      {i:'🎯',l:'Accuracy',  v:acc+'%',      c:'#10b981'},
      {i:'📝',l:'Answered',  v:data.total_q, c:'#06b6d4'}
    ].map(function(s) {
      return '<div class="bs"><div class="bs-icon">' + s.i + '</div>'
        + '<div class="bs-val" style="color:' + s.c + '">' + s.v + '</div>'
        + '<div class="bs-lbl">' + s.l + '</div></div>';
    }).join('');

    // Level bar
    const xpInLevel = data.xp % 200;
    $('lp-bar').style.width = (xpInLevel / 200 * 100) + '%';
    $('lp-lbl').textContent = 'Level ' + data.level;
    $('lp-xp').textContent  = (200 - xpInLevel) + ' XP to Level ' + (data.level + 1);

    // Charts
    renderLineChart(data.progress_data || []);
    renderPieChart(data.correct_q || 0, (data.total_q || 0) - (data.correct_q || 0));
    renderMasteryChart(data.topic_stats || []);
    renderTopicTable(data.topic_stats || []);
    renderCalendar(data.study_dates || []);

    // Topics
    $('topics-list').innerHTML = (data.topics_studied || []).length
      ? data.topics_studied.map(t => '<span class="topic-tag">' + escHtml(t) + '</span>').join('')
      : '<span class="empty-tag">No topics yet — start a quiz!</span>';

    $('weak-list').innerHTML = (data.weak_topics || []).length
      ? data.weak_topics.map(t => '<span class="weak-tag2">⚠️ ' + escHtml(t) + '</span>').join('')
      : '<span class="empty-tag">No weak topics — great job! ✅</span>';

    // Correct/wrong/total
    $('d-correct').textContent = data.correct_q || 0;
    $('d-wrong').textContent   = (data.total_q || 0) - (data.correct_q || 0);
    $('d-total').textContent   = data.total_q  || 0;

    // Badges
    $('d-badges').innerHTML = (data.badges || []).length
      ? data.badges.map(b => '<span class="badge">' + b + '</span>').join('')
      : '<span class="nb">Complete quizzes to earn badges!</span>';

  } catch(e) {
    console.warn('Dashboard error:', e.message);
  }
}

const CHART_OPTS = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks:{ color:'rgba(255,255,255,0.4)' }, grid:{ color:'rgba(255,255,255,0.05)' } },
    y: { ticks:{ color:'rgba(255,255,255,0.4)' }, grid:{ color:'rgba(255,255,255,0.05)' } }
  }
};

function renderLineChart(data) {
  destroyChart('progress');
  const ctx = $('progress-chart');
  if (!ctx || !data.length) return;
  charts.progress = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => 'Q' + d.n),
      datasets: [{
        label: 'Accuracy %', data: data.map(d => d.accuracy),
        borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)',
        borderWidth: 2.5, pointRadius: 3, fill: true, tension: 0.4
      }]
    },
    options: {
      ...CHART_OPTS,
      scales: {
        x: CHART_OPTS.scales.x,
        y: { ...CHART_OPTS.scales.y, beginAtZero: true, max: 100,
          ticks: { color: 'rgba(255,255,255,0.4)', callback: v => v + '%' } }
      }
    }
  });
}

function renderPieChart(correct, wrong) {
  destroyChart('pie');
  const ctx = $('pie-chart');
  if (!ctx || (correct === 0 && wrong === 0)) return;
  charts.pie = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Correct', 'Wrong'],
      datasets: [{
        data: [correct, wrong],
        backgroundColor: ['rgba(16,185,129,0.8)', 'rgba(239,68,68,0.7)'],
        borderColor: ['#10b981', '#ef4444'], borderWidth: 2
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true, position: 'bottom',
          labels: { color: 'rgba(255,255,255,0.6)', padding: 15, font: { size: 12 } }
        }
      }
    }
  });
}

function renderMasteryChart(topicStats) {
  destroyChart('mastery');
  const ctx = $('mastery-chart');
  if (!ctx || !topicStats.length) return;
  const labels  = topicStats.map(t => t.topic.length > 16 ? t.topic.slice(0,16)+'…' : t.topic);
  const values  = topicStats.map(t => t.mastery);
  const colors  = values.map(m => m>=80 ? 'rgba(16,185,129,0.8)'
                                : m>=50 ? 'rgba(245,158,11,0.8)' : 'rgba(239,68,68,0.7)');
  charts.mastery = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Mastery %', data: values, backgroundColor: colors, borderRadius: 6 }] },
    options: {
      ...CHART_OPTS,
      scales: {
        x: { ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { beginAtZero: true, max: 100,
          ticks: { color: 'rgba(255,255,255,0.4)', callback: v => v + '%' },
          grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

function renderTopicTable(topicStats) {
  const wrap = $('topic-table');
  if (!wrap) return;
  if (!topicStats.length) {
    wrap.innerHTML = '<p style="color:var(--sub);font-style:italic;padding:10px">Answer questions to see stats per topic</p>';
    return;
  }
  wrap.innerHTML = '<div class="tt-row header">'
    + '<div>Topic</div><div>Total</div><div>Correct</div><div>Mastery</div><div>Avg Time</div></div>'
    + topicStats.map(function(t) {
      return '<div class="tt-row">'
        + '<div style="font-weight:600">' + escHtml(t.topic) + '</div>'
        + '<div>' + t.total + '</div>'
        + '<div style="color:#10b981">' + t.correct + '</div>'
        + '<div>'
          + '<div style="font-size:.8rem;margin-bottom:3px">' + t.mastery + '%</div>'
          + '<div class="mastery-bar-wrap"><div class="mastery-bar" style="width:' + t.mastery + '%"></div></div>'
        + '</div>'
        + '<div style="color:#67e8f9">' + t.avg_time + 's</div>'
        + '</div>';
    }).join('');
}

function renderCalendar(studyDates) {
  const cal = $('streak-calendar');
  if (!cal) return;
  const studied = new Set(studyDates);
  const today = new Date();
  let html = '';
  for (let i = 89; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const iso = d.toISOString().split('T')[0];
    html += '<div class="cal-day'
      + (studied.has(iso) ? ' studied' : '')
      + (i === 0 ? ' today' : '')
      + '" title="' + iso + '"></div>';
  }
  cal.innerHTML = html;
}


// ═══════════════════════════════════════════════════════
//  LEADERBOARD
// ═══════════════════════════════════════════════════════

async function loadLeaderboard() {
  const lb = $('lb-table');
  lb.innerHTML = '<div class="lb-loading">Loading...</div>';
  try {
    const res = await fetch(API + '/api/leaderboard');
    if (!res.ok) throw new Error('Failed to load');
    const data = await res.json();
    const rows = data.leaderboard || [];
    if (!rows.length) {
      lb.innerHTML = '<div class="lb-loading">No scores yet — be the first!</div>';
      return;
    }
    const medals = ['🥇','🥈','🥉'];
    lb.innerHTML = rows.map(function(r) {
      return '<div class="lb-row-item">'
        + '<div class="lb-rank ' + (['gold','silver','bronze'][r.rank-1]||'') + '">'
          + (r.rank <= 3 ? medals[r.rank-1] : r.rank) + '</div>'
        + '<div class="lb-name">' + escHtml(r.name) + '</div>'
        + '<div><div class="lb-score">' + r.score + ' pts</div>'
          + '<div class="lb-acc">' + (r.accuracy||0) + '% · ' + (r.xp||0) + ' XP</div></div>'
        + '</div>';
    }).join('');
  } catch(e) {
    lb.innerHTML = '<div class="lb-loading">Failed to load leaderboard</div>';
  }
}

async function submitToLeaderboard() {
  const nameEl = $('lb-name');
  const name = (nameEl ? nameEl.value : '').trim();
  if (!name) { showToast('Enter your name first!'); if (nameEl) nameEl.focus(); return; }

  const errEl = $('lb-error');
  if (errEl) errEl.classList.add('hidden');

  try {
    // Get fresh server stats for accurate score
    const statsRes = await fetch(API + '/api/stats/' + SESSION_ID);
    const fresh = statsRes.ok ? await statsRes.json() : {};
    const finalScore = Math.max(state.score, fresh.score || 0);
    const finalXp    = Math.max(state.xp,    fresh.xp    || 0);
    const finalAcc   = fresh.accuracy || (state.totalQ > 0
      ? Math.round(state.correctQ / state.totalQ * 100) : 0);
    const topics = (fresh.topics_studied || state.topics_studied || []).length;

    const res = await fetch(API + '/api/leaderboard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, score: finalScore, xp: finalXp, accuracy: finalAcc, topics, session_id: SESSION_ID })
    });
    if (!res.ok) throw new Error('Server error');
    const data = await res.json();
    showToast('🏆 Submitted! Your rank: #' + data.rank, 4000);
    showScreen('leaderboard');
  } catch(e) {
    showToast('Failed to submit score');
    if (errEl) { errEl.textContent = '❌ ' + e.message; errEl.classList.remove('hidden'); }
  }
}


// ═══════════════════════════════════════════════════════
//  REVIEW — Spaced Repetition
// ═══════════════════════════════════════════════════════

async function loadReview() {
  const list  = $('review-list');
  const empty = $('review-empty');
  list.innerHTML = '';

  try {
    const [dueRes, statsRes] = await Promise.all([
      fetch(API + '/api/due-topics/' + SESSION_ID),
      fetch(API + '/api/stats/' + SESSION_ID)
    ]);
    const dueData   = dueRes.ok   ? await dueRes.json()   : { due: [] };
    const statsData = statsRes.ok ? await statsRes.json() : { topics_studied: [] };

    const due       = dueData.due || [];
    const allTopics = statsData.topics_studied || [];
    const dueSet    = new Set(due.map(d => d.topic));

    if (!allTopics.length && !due.length) {
      list.classList.add('hidden');
      empty.classList.remove('hidden');
      return;
    }

    empty.classList.add('hidden');
    list.classList.remove('hidden');
    let html = '';

    if (due.length) {
      html += '<div class="review-section-title">⏰ Due Now (' + due.length + ')</div>';
      html += due.map(function(d) {
        return '<div class="review-item due-now">'
          + '<div><div class="review-topic">' + escHtml(d.topic) + '</div>'
          + '<div class="review-interval">Due for review</div></div>'
          + '<button class="review-btn urgent" onclick="quickReview(\''
          + d.topic.replace(/'/g,"\\'") + '\')">Review Now →</button>'
          + '</div>';
      }).join('');
    }

    const upcoming = allTopics.filter(t => !dueSet.has(t));
    if (upcoming.length) {
      html += '<div class="review-section-title" style="margin-top:20px">📚 All Topics (' + upcoming.length + ')</div>';
      html += upcoming.map(function(t) {
        return '<div class="review-item">'
          + '<div><div class="review-topic">' + escHtml(t) + '</div>'
          + '<div class="review-interval">Not due yet</div></div>'
          + '<button class="review-btn" onclick="quickReview(\'' + t.replace(/'/g,"\\'") + '\')">'
          + 'Practice Again →</button>'
          + '</div>';
      }).join('');
    }

    list.innerHTML = html;
  } catch(e) {
    list.innerHTML = '<p style="color:var(--sub);padding:20px">Error loading topics: ' + e.message + '</p>';
  }
}

function quickReview(topic) {
  const inp = $('topic-input');
  if (inp) inp.value = topic;
  showScreen('home');
  setTimeout(generateQuestions, 200);
}


// ═══════════════════════════════════════════════════════
//  EXPORT REPORT
// ═══════════════════════════════════════════════════════

async function exportReport() {
  try {
    const res = await fetch(API + '/api/export-report/' + SESSION_ID);
    if (!res.ok) throw new Error('Server error');
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = 'EduMate360_Report_' + new Date().toISOString().split('T')[0] + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('📥 Report downloaded!');
  } catch(e) {
    showToast('Export failed: ' + e.message);
  }
}


// ═══════════════════════════════════════════════════════════════════
//  CAMERA + EMOTION ENGINE
//  Uses MediaPipe Tasks Vision FaceLandmarker
//  — No separate model files, no CDN failure
//  — 478 facial landmarks → cognitive state
// ═══════════════════════════════════════════════════════════════════

let faceLandmarker   = null;
let mpLoading        = false;
let camStream        = null;
let emotionInterval  = null;
let lastVideoTime    = -1;
let consecutiveAnxious = 0;
let breathingActive  = false;
let currentQEmotions = [];  // emotion log for current question

const COG_META = {
  focused:    { emoji:'🎯', label:'Focused',    color:'#10b981', tip:'Performance looks good — keep it up!' },
  loading:    { emoji:'🤔', label:'Processing', color:'#f59e0b', tip:'Take your time, think it through.' },
  overloaded: { emoji:'🧩', label:'Overloaded', color:'#f97316', tip:'Slow down — break the problem into parts.' },
  fatigued:   { emoji:'😴', label:'Fatigued',   color:'#94a3b8', tip:'Consider a short break soon.' },
  anxious:    { emoji:'🧘', label:'Anxious',    color:'#ef4444', tip:'Breathe — you know this material.' },
};

const BAR_COLORS = {
  happy:'#10b981', focused:'#6366f1', confused:'#f59e0b', tired:'#94a3b8', anxious:'#ef4444'
};

// ── Load MediaPipe FaceLandmarker ─────────────────────────────────
async function preloadFaceApi() {
  if (mpLoading || faceLandmarker) return;
  mpLoading = true;
  try {
    // Access the global Vision object from the loaded bundle
    const Vision = window.vision || window.mpVision;
    if (!Vision) {
      // Try dynamic import fallback
      const mod = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs');
      window.vision = mod;
    }
    const V = window.vision || window.mpVision;
    if (!V || !V.FaceLandmarker) {
      console.warn('MediaPipe Vision bundle not available yet');
      mpLoading = false;
      return;
    }
    const { FaceLandmarker, FilesetResolver } = V;
    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'CPU'
      },
      outputFaceBlendshapes: true,
      runningMode: 'VIDEO',
      numFaces: 1
    });
    console.log('✅ MediaPipe FaceLandmarker ready');
    const statusEl = $('face-status');
    if (statusEl) { statusEl.textContent = 'AI ready ✓'; statusEl.style.background = 'rgba(16,185,129,.8)'; }
  } catch(e) {
    console.warn('MediaPipe load error:', e.message);
    const statusEl = $('face-status');
    if (statusEl) { statusEl.textContent = 'AI unavailable'; statusEl.style.background = 'rgba(239,68,68,.7)'; }
  }
  mpLoading = false;
}

// ── Blendshapes → cognitive state ────────────────────────────────
function blendsToCog(blendshapes) {
  if (!blendshapes || !blendshapes.categories) return 'focused';
  const bs = {};
  blendshapes.categories.forEach(c => { bs[c.categoryName] = c.score; });
  const blink   = ((bs.eyeBlinkLeft||0) + (bs.eyeBlinkRight||0)) / 2;
  const browDn  = ((bs.browDownLeft||0) + (bs.browDownRight||0)) / 2;
  const browUp  = bs.browInnerUp || 0;
  const smile   = ((bs.mouthSmileLeft||0) + (bs.mouthSmileRight||0)) / 2;
  const frown   = ((bs.mouthFrownLeft||0) + (bs.mouthFrownRight||0)) / 2;
  const jawOpen = bs.jawOpen || 0;
  const squint  = ((bs.eyeSquintLeft||0) + (bs.eyeSquintRight||0)) / 2;
  if (browDn > 0.35 && smile < 0.1 && frown > 0.15) return 'anxious';
  if (squint > 0.3  && browDn > 0.25)                return 'overloaded';
  if (blink  > 0.4)                                   return 'fatigued';
  if (browUp > 0.3  && jawOpen > 0.1)                 return 'loading';
  return 'focused';
}

function blendsToBarData(blendshapes) {
  if (!blendshapes || !blendshapes.categories) return [];
  const bs = {};
  blendshapes.categories.forEach(c => { bs[c.categoryName] = c.score; });
  return [
    { name:'happy',   val: Math.round(((bs.mouthSmileLeft||0)+(bs.mouthSmileRight||0))/2*100) },
    { name:'focused', val: Math.round(Math.max(0, 1-(bs.browDownLeft||0)-(bs.eyeSquintLeft||0))*60) },
    { name:'confused',val: Math.round(((bs.browInnerUp||0)*0.5+(bs.jawOpen||0)*0.5)*100) },
    { name:'tired',   val: Math.round(((bs.eyeBlinkLeft||0)+(bs.eyeBlinkRight||0))/2*100) },
    { name:'anxious', val: Math.round(((bs.browDownLeft||0)+(bs.mouthFrownLeft||0))/2*100) },
  ];
}

// ── Main detection loop ───────────────────────────────────────────
async function runEmotionDetection() {
  const video    = $('webcam');
  const canvas   = $('face-canvas');
  const statusEl = $('face-status');
  if (!video || !canvas || video.paused || video.ended) return;

  if (!faceLandmarker) {
    if (statusEl) { statusEl.textContent = 'Loading AI...'; statusEl.style.background = 'rgba(245,158,11,.7)'; }
    if (!mpLoading) preloadFaceApi();
    return;
  }
  try {
    const nowMs = performance.now();
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, nowMs);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,160,120);

    if (!results.faceLandmarks || !results.faceLandmarks.length) {
      if (statusEl) { statusEl.textContent = 'No face detected'; statusEl.style.background = 'rgba(239,68,68,.7)'; }
      return;
    }
    // Draw face box
    const lm = results.faceLandmarks[0];
    const xs = lm.map(p=>p.x*160), ys = lm.map(p=>p.y*120);
    const x1=Math.min(...xs), x2=Math.max(...xs), y1=Math.min(...ys), y2=Math.max(...ys);
    ctx.strokeStyle='#6366f1'; ctx.lineWidth=2;
    ctx.strokeRect(x1-4, y1-4, x2-x1+8, y2-y1+8);
    if (statusEl) { statusEl.textContent='Face detected ✓'; statusEl.style.background='rgba(16,185,129,.7)'; }

    const bs = results.faceBlendshapes?.[0];
    const cogState = bs ? blendsToCog(bs) : 'focused';
    const meta = COG_META[cogState] || COG_META.focused;

    // Update cognitive state display
    const stateEl = $('emo-state');
    if (stateEl) { stateEl.textContent = meta.emoji+' '+meta.label; stateEl.style.color = meta.color; }
    const subEl = $('emo-sub');
    if (subEl) subEl.textContent = meta.tip;

    // Update cog badge in live strip
    const badge = $('cog-badge');
    if (badge) { badge.textContent = meta.emoji+' '+meta.label; badge.style.borderColor=meta.color; badge.style.color=meta.color; badge.classList.remove('hidden'); }

    // Emotion bars
    const bars = bs ? blendsToBarData(bs) : [];
    const barsEl = $('emo-bars');
    if (barsEl) barsEl.innerHTML = bars.map(b =>
      '<div class="emo-bar-row">'
      +'<div class="emo-bar-label">'+b.name+'</div>'
      +'<div class="emo-bar-track"><div class="emo-bar-fill" style="width:'+b.val+'%;background:'+(BAR_COLORS[b.name]||'#6366f1')+'"></div></div>'
      +'<div class="emo-bar-pct">'+b.val+'%</div>'
      +'</div>').join('');

    // Log emotion for this question
    currentQEmotions.push({ cogState, ts: Date.now() });

    // Anxiety check — trigger breathing exercise after 3 consecutive anxious detections
    if (cogState === 'anxious') {
      consecutiveAnxious++;
      if (consecutiveAnxious >= 3 && !breathingActive) triggerBreathingExercise();
    } else {
      consecutiveAnxious = 0;
    }

    // Send to backend (throttled — only on state change)
    const backendEmo = (cogState==='anxious'||cogState==='overloaded') ? 'stressed'
                     : (cogState==='fatigued'||cogState==='loading')   ? 'confused' : 'focused';
    if (backendEmo !== state.emotion) {
      state.emotion = backendEmo;
      fetch(API+'/api/emotion', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({session_id:SESSION_ID, emotion:backendEmo})
      }).catch(()=>{});
    }
  } catch(e) { console.warn('Detection error:', e.message); }
}

// ── Toggle camera on/off ──────────────────────────────────────────
async function toggleCam() {
  const btn   = $('cam-btn');
  const panel = $('cam-panel');
  if (!btn || !panel) return;

  if (camStream) {
    // Turn OFF
    clearInterval(emotionInterval);
    emotionInterval = null;
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
    lastVideoTime = -1;
    const video = $('webcam');
    if (video) video.srcObject = null;
    panel.classList.add('hidden');
    btn.textContent = '📷 Cam';
    btn.classList.remove('active');
    const badge = $('cog-badge');
    if (badge) badge.classList.add('hidden');
    return;
  }

  // Turn ON
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { width:160, height:120, facingMode:'user' }, audio:false });
    const video = $('webcam');
    if (video) { video.srcObject = camStream; await video.play().catch(()=>{}); }
    panel.classList.remove('hidden');
    btn.textContent = '📷 On';
    btn.classList.add('active');
    showToast('📷 Camera on — loading AI models...');
    // Start detection loop
    emotionInterval = setInterval(runEmotionDetection, 1200);
    // Load MediaPipe in background
    if (!faceLandmarker && !mpLoading) preloadFaceApi();
  } catch(e) {
    showToast('Camera access denied or unavailable');
    camStream = null;
  }
}

// ── Breathing exercise (triggered on 3x anxious) ──────────────────
function triggerBreathingExercise() {
  breathingActive = true;
  const overlay = document.createElement('div');
  overlay.className = 'breathing-overlay';
  overlay.id = 'breathing-overlay';
  const phases = ['Breathe In','Hold','Breathe Out','Hold'];
  let phase = 0, cycle = 0;
  function nextPhase() {
    if (cycle >= 2) { overlay.remove(); breathingActive=false; consecutiveAnxious=0; return; }
    overlay.innerHTML =
      '<div class="breathing-circle">'+phases[phase]+'</div>'
      +'<div class="breathing-text">Box Breathing — '+(cycle===0?'Cycle 1':'Cycle 2')+' of 2</div>'
      +'<button class="breathing-skip" onclick="document.getElementById('breathing-overlay').remove();window.breathingActive=false">Skip</button>';
    phase++;
    if (phase >= 4) { phase=0; cycle++; }
    setTimeout(nextPhase, 4000);
  }
  document.body.appendChild(overlay);
  nextPhase();
}
window.breathingActive = false;
