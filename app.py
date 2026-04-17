"""
Project B — FastAPI Web Application
Acmera Support Agent (LangGraph)

Endpoints:
  GET  /        → HTML UI
  GET  /health  → ALB health check
  POST /query   → run agent, return answer + full trajectory

Run locally:
    uvicorn app:app --reload --port 8001
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Acmera Support Agent — Project B", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int          # +1 = thumbs up, -1 = thumbs down
    comment: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "service": "project-b-agent"}


@app.post("/query")
def query_endpoint(req: QueryRequest):
    from scripts.agent import run_agent
    try:
        result = run_agent(req.query)
        return {
            "query":           result["query"],
            "answer":          result["answer"],
            "intent":          result["intent"],
            "tools_called":    result["tools_called"],
            "steps_taken":     result["steps_taken"],
            "should_escalate": result["should_escalate"],
            "elapsed_seconds": result["elapsed_seconds"],
            "trace_id":        result.get("trace_id"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    from langfuse import Langfuse
    try:
        lf = Langfuse()
        lf.score(
            trace_id=req.trace_id,
            name="user_feedback",
            value=req.rating,           # +1 or -1
            comment=req.comment or None,
        )
        lf.flush()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def ui():
    return PROJECT_B_HTML


PROJECT_B_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Acmera Support Agent</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
    }

    header {
      background: #1e293b;
      border-bottom: 1px solid #334155;
      padding: 16px 32px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    header .logo  { font-size: 20px; font-weight: 700; color: #8b5cf6; }
    header .subtitle { font-size: 13px; color: #64748b; margin-left: 4px; }
    header .badge {
      margin-left: auto;
      font-size: 11px;
      padding: 3px 10px;
      border-radius: 99px;
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
    }

    .container { max-width: 820px; margin: 0 auto; padding: 32px 24px; }

    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .card-title {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #64748b;
      margin-bottom: 14px;
    }

    textarea {
      width: 100%;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #e2e8f0;
      font-size: 14px;
      padding: 12px;
      resize: vertical;
      min-height: 80px;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: #8b5cf6; }

    .ask-btn {
      width: 100%;
      margin-top: 14px;
      padding: 12px;
      background: #8b5cf6;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    .ask-btn:hover   { background: #7c3aed; }
    .ask-btn:disabled { background: #334155; color: #64748b; cursor: not-allowed; }

    /* ── Trajectory ── */
    .trajectory { display: flex; flex-direction: column; gap: 10px; }

    .step {
      display: flex;
      gap: 14px;
      align-items: flex-start;
      opacity: 0;
      transform: translateY(8px);
      animation: fadeUp 0.3s forwards;
    }
    @keyframes fadeUp {
      to { opacity: 1; transform: translateY(0); }
    }

    .step-number {
      width: 26px; height: 26px;
      border-radius: 50%;
      background: #334155;
      display: flex; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 700; color: #94a3b8;
      flex-shrink: 0;
      margin-top: 2px;
    }
    .step-number.classify  { background: #1e40af; color: #93c5fd; }
    .step-number.tool      { background: #065f46; color: #6ee7b7; }
    .step-number.evaluate  { background: #78350f; color: #fcd34d; }
    .step-number.respond   { background: #4c1d95; color: #c4b5fd; }
    .step-number.escalate  { background: #7f1d1d; color: #fca5a5; }

    .step-body { flex: 1; }
    .step-label { font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 3px; }
    .step-detail { font-size: 13px; color: #cbd5e1; }

    .badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
    .badge {
      font-size: 11px;
      padding: 2px 9px;
      border-radius: 99px;
      border: 1px solid #334155;
      color: #94a3b8;
    }
    .badge.intent  { border-color: #3b82f6; color: #93c5fd; }
    .badge.tool    { border-color: #10b981; color: #6ee7b7; }
    .badge.verdict { border-color: #f59e0b; color: #fcd34d; }
    .badge.respond { border-color: #8b5cf6; color: #c4b5fd; }

    /* ── Answer ── */
    .answer-text {
      font-size: 15px;
      line-height: 1.7;
      color: #e2e8f0;
      white-space: pre-wrap;
    }

    /* ── Escalation banner ── */
    .escalation-banner {
      background: #450a0a;
      border: 1px solid #7f1d1d;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 13px;
      color: #fca5a5;
      margin-bottom: 14px;
    }

    /* ── Meta ── */
    .meta-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .meta-pill {
      font-size: 11px;
      padding: 3px 10px;
      border-radius: 99px;
      border: 1px solid #334155;
      color: #94a3b8;
    }

    /* ── Loading ── */
    .spinner {
      display: inline-block;
      width: 18px; height: 18px;
      border: 2px solid #334155;
      border-top-color: #8b5cf6;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-text { color: #64748b; font-size: 14px; padding: 30px 0; text-align: center; }

    .placeholder { color: #475569; font-size: 14px; text-align: center; padding: 40px 0; }

    .error-box {
      background: #450a0a;
      border: 1px solid #7f1d1d;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 14px;
      color: #fca5a5;
    }

    /* ── Sample queries ── */
    .samples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .sample {
      font-size: 12px;
      padding: 5px 12px;
      border: 1px solid #334155;
      border-radius: 6px;
      color: #64748b;
      cursor: pointer;
      transition: all 0.15s;
      background: none;
    }
    .sample:hover { border-color: #8b5cf6; color: #c4b5fd; }

    /* ── Feedback ── */
    .feedback-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 20px;
      padding-top: 16px;
      border-top: 1px solid #334155;
    }
    .feedback-label { font-size: 12px; color: #64748b; }
    .thumb-btn {
      background: none;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #94a3b8;
      font-size: 18px;
      width: 38px; height: 38px;
      cursor: pointer;
      transition: all 0.15s;
      display: flex; align-items: center; justify-content: center;
    }
    .thumb-btn:hover { border-color: #8b5cf6; color: #e2e8f0; }
    .thumb-btn.selected-up   { background: #14532d; border-color: #22c55e; color: #22c55e; }
    .thumb-btn.selected-down { background: #450a0a; border-color: #ef4444; color: #ef4444; }
    .feedback-thanks { font-size: 12px; color: #22c55e; display: none; }
  </style>
</head>
<body>
  <header>
    <span class="logo">Acmera</span>
    <span class="subtitle">Support Agent</span>
    <span class="badge">Project B — LangGraph Agent</span>
  </header>

  <div class="container">

    <!-- Query card -->
    <div class="card">
      <div class="card-title">Customer Query</div>
      <textarea id="queryInput" placeholder="Type your support question..."></textarea>
      <div class="samples">
        <button class="sample" onclick="setQuery(this)">What is the return window for electronics?</button>
        <button class="sample" onclick="setQuery(this)">Where is my order ORD-445521?</button>
        <button class="sample" onclick="setQuery(this)">I'm Premium Gold — do I get extended returns?</button>
        <button class="sample" onclick="setQuery(this)">My card was charged twice. I need a refund now.</button>
      </div>
      <button class="ask-btn" id="askBtn" onclick="runQuery()">Ask Agent</button>
    </div>

    <!-- Trajectory card -->
    <div class="card" id="trajectoryCard" style="display:none;">
      <div class="card-title">Agent Reasoning Trajectory</div>
      <div class="trajectory" id="trajectoryArea"></div>
    </div>

    <!-- Answer card -->
    <div class="card" id="answerCard" style="display:none;">
      <div class="card-title">Response</div>
      <div id="answerArea"></div>
    </div>

    <!-- Initial placeholder -->
    <div class="card" id="placeholderCard">
      <div class="placeholder">Ask a question to watch the agent reason step-by-step.</div>
    </div>

  </div>

  <script>
    function setQuery(btn) {
      document.getElementById("queryInput").value = btn.textContent;
    }

    let currentTraceId = null;

    const TOOL_ICONS = {
      policy_kb:      "📚",
      order_tracker:  "📦",
      account_lookup: "👤",
    };

    async function runQuery() {
      const query = document.getElementById("queryInput").value.trim();
      if (!query) return;

      const btn = document.getElementById("askBtn");
      btn.disabled = true;
      btn.textContent = "Agent thinking...";

      // Show loading, hide result cards
      document.getElementById("placeholderCard").style.display = "none";
      document.getElementById("trajectoryCard").style.display = "block";
      document.getElementById("answerCard").style.display = "none";
      document.getElementById("trajectoryArea").innerHTML =
        '<div class="loading-text"><span class="spinner"></span>Agent is reasoning...</div>';

      try {
        const res = await fetch("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Request failed");
        renderTrajectory(data);
        renderAnswer(data);
      } catch (err) {
        document.getElementById("trajectoryArea").innerHTML =
          `<div class="error-box">${err.message}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Ask Agent";
      }
    }

    function renderTrajectory(data) {
      const steps = [];
      let delay = 0;

      // Step 1: Classify
      steps.push(buildStep("classify", 1,
        "Classify Intent",
        `<div class="badge-row"><span class="badge intent">${data.intent}</span></div>`,
        delay));
      delay += 120;

      // Steps 2…N: Tool calls
      data.tools_called.forEach((tool, i) => {
        steps.push(buildStep("tool", i + 2,
          `Tool Call — ${tool}`,
          `<div class="badge-row">
             <span class="badge tool">${TOOL_ICONS[tool] || "🔧"} ${tool}</span>
           </div>`,
          delay));
        delay += 120;

        // Evaluate after each tool (except last — that's the respond/escalate)
        if (i < data.tools_called.length - 1) {
          steps.push(buildStep("evaluate", i + 3,
            "Evaluate",
            `<div class="badge-row"><span class="badge verdict">need more info → calling next tool</span></div>`,
            delay));
          delay += 120;
        }
      });

      // Final evaluate
      const finalVerdict = data.should_escalate ? "escalate" : "sufficient → respond";
      const finalClass   = data.should_escalate ? "evaluate" : "evaluate";
      steps.push(buildStep(finalClass, data.steps_taken + 2,
        "Evaluate",
        `<div class="badge-row"><span class="badge verdict">${finalVerdict}</span></div>`,
        delay));
      delay += 120;

      // Respond or escalate
      if (data.should_escalate) {
        steps.push(buildStep("escalate", data.steps_taken + 3,
          "Escalate to Human",
          `<div class="badge-row"><span class="badge" style="border-color:#ef4444;color:#fca5a5;">Handed off to support team</span></div>`,
          delay));
      } else {
        steps.push(buildStep("respond", data.steps_taken + 3,
          "Generate Response",
          `<div class="badge-row"><span class="badge respond">Answer ready</span></div>`,
          delay));
      }

      document.getElementById("trajectoryArea").innerHTML = steps.join("");
    }

    function buildStep(type, num, label, detail, delay) {
      return `
        <div class="step" style="animation-delay:${delay}ms">
          <div class="step-number ${type}">${num}</div>
          <div class="step-body">
            <div class="step-label">${label}</div>
            <div class="step-detail">${detail}</div>
          </div>
        </div>`;
    }

    function renderAnswer(data) {
      currentTraceId = data.trace_id || null;
      const answerCard = document.getElementById("answerCard");
      answerCard.style.display = "block";

      const escalationHtml = data.should_escalate
        ? `<div class="escalation-banner">⚠️ This query has been escalated to a human support agent.</div>`
        : "";

      const feedbackHtml = currentTraceId ? `
        <div class="feedback-row">
          <span class="feedback-label">Was this helpful?</span>
          <button class="thumb-btn" id="thumbUp"   onclick="sendFeedback(1)"  title="Helpful">👍</button>
          <button class="thumb-btn" id="thumbDown" onclick="sendFeedback(-1)" title="Not helpful">👎</button>
          <span class="feedback-thanks" id="feedbackThanks">Thanks for your feedback!</span>
        </div>` : "";

      document.getElementById("answerArea").innerHTML = `
        ${escalationHtml}
        <div class="meta-strip">
          <span class="meta-pill">${data.intent}</span>
          <span class="meta-pill">${data.steps_taken} tool call${data.steps_taken !== 1 ? "s" : ""}</span>
          <span class="meta-pill">${data.elapsed_seconds}s</span>
        </div>
        <div class="answer-text">${escapeHtml(data.answer)}</div>
        ${feedbackHtml}
      `;
    }

    async function sendFeedback(rating) {
      if (!currentTraceId) return;
      const upBtn   = document.getElementById("thumbUp");
      const downBtn = document.getElementById("thumbDown");
      const thanks  = document.getElementById("feedbackThanks");
      if (!upBtn) return;

      upBtn.disabled = true; downBtn.disabled = true;
      if (rating === 1)  upBtn.classList.add("selected-up");
      else               downBtn.classList.add("selected-down");

      try {
        await fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trace_id: currentTraceId, rating }),
        });
        if (thanks) thanks.style.display = "inline";
      } catch (e) {
        console.error("Feedback error:", e);
      }
    }

    function escapeHtml(str) {
      return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    document.getElementById("queryInput").addEventListener("keydown", e => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runQuery();
    });
  </script>
</body>
</html>"""
