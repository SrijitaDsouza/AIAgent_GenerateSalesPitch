"""
BrokerDesk backend: serves the landing page and runs the agents defined in
"Sales Manager.ipynb" when the user submits the Generate Sales Pitch dialog.

Flow (human in the loop):
  1. POST /api/sales-pitch/drafts  -> the 3 sales agents write drafts, the manager picks the best
  2. The user accepts the AI pick or chooses another draft on the page
  3. POST /api/sales-pitch/send    -> the Email Manager writes the subject, converts to HTML and
                                      sends with SendGrid; progress is streamed back to the page

Run:  uvicorn server:app --reload     (from the folder containing this file and the notebook)
Open: http://localhost:8000
"""
import ast
import asyncio
import copy
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import nbformat
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agents import Agent, FunctionTool, InputGuardrailTripwireTriggered, RunConfig, Runner, trace

BASE_DIR = Path(__file__).parent
NOTEBOOK_PATH = Path(os.getenv("SALES_NOTEBOOK", BASE_DIR / "Sales Manager.ipynb"))
AGENT: dict = {}


# ---------------------------------------------------------------- notebook loader
def _has_top_level_await(node: ast.AST) -> bool:
    """True if `node` awaits outside any function (e.g. the demo `await Runner.run(...)` cell)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
        return False
    if isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
        return True
    return any(_has_top_level_await(c) for c in ast.iter_child_nodes(node))


def load_notebook_namespace(path: Path) -> dict:
    """Execute the notebook's definitions (agents, tools, guardrail) without running its demo cell."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find the notebook at {path}")
    nb = nbformat.read(path, as_version=4)
    ns: dict = {"__name__": "sales_manager_notebook", "display": print}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        # Drop IPython magics / shell lines such as `!pip install ...` or `%load_ext ...`
        src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("!", "%")))
        tree = ast.parse(src)
        for stmt in tree.body:
            if _has_top_level_await(stmt):
                continue
            # Run each statement on its own so a failing test cell (e.g. a test email
            # sent with SendGrid) is skipped with a warning instead of stopping the server.
            try:
                exec(compile(ast.Module(body=[stmt], type_ignores=[]), str(path), "exec"), ns)
            except Exception as exc:
                print(f"[notebook] Skipped line {stmt.lineno}: {ast.unparse(stmt)[:80]!r} -> {type(exc).__name__}: {exc}")
    return ns


def build_sales_manager(ns: dict) -> Agent:
    """Use the notebook's `careful_sales_manager` if it defines one, otherwise build it from the notebook's parts."""
    if isinstance(ns.get("careful_sales_manager"), Agent):
        return ns["careful_sales_manager"]
    return Agent(
        name="Sales Manager",
        instructions=ns["sales_manager_instructions"],
        tools=ns["tools"],
        handoffs=ns["handoffs"],
        model=ns["gemini_model"],
        input_guardrails=[ns["guardrail_against_name"]],
    )


class DraftChoice(BaseModel):
    best_draft: int     # 1, 2 or 3
    reason: str         # one sentence, shown to the user


JUDGE_INSTRUCTIONS = (
    "You are a Sales Manager at a General Insurance Company in India. "
    "You are given a brief and three cold sales email drafts written by your sales agents. "
    "Choose the single most effective draft for this brief. "
    "Return best_draft as 1, 2 or 3, and reason as one short sentence explaining the choice."
)

# Tone of each sales agent, taken from instructions1/2/3 in the notebook
DRAFT_STYLES = ["Professional", "Witty", "Concise"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    ns = load_notebook_namespace(NOTEBOOK_PATH)
    manager = build_sales_manager(ns)
    missing = [n for n in ("sales_agent1", "sales_agent2", "sales_agent3") if n not in ns]
    if missing:
        raise RuntimeError(f"The notebook does not define {', '.join(missing)}")
    AGENT["manager"] = manager
    AGENT["sales_agents"] = [ns["sales_agent1"], ns["sales_agent2"], ns["sales_agent3"]]
    AGENT["guardrail"] = ns.get("guardrail_agent")
    AGENT["emailer"] = ns.get("emailer_agent") or manager.handoffs[0]
    # The manager's judgement step, with the same model as careful_sales_manager,
    # but it returns its pick instead of handing off, so the user can confirm it first.
    AGENT["judge"] = Agent(name="Sales Manager (pick best draft)", instructions=JUDGE_INSTRUCTIONS,
                           model=manager.model, output_type=DraftChoice)
    yield


app = FastAPI(title="BrokerDesk AI agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- request -> agent message
class PitchRequest(BaseModel):
    customer_name: str = ""
    designation: str = ""
    company: str = ""
    product: str = ""
    insurer: str = ""


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()[:120]


def build_message(req: PitchRequest) -> str:
    """Send out a cold sales email addressed to Dear Sir/Madam {designation} of {company},
    pitching for the {product} (if given) and on behalf of {insurer} (if given).
    Blank inputs -> generic email. Kept in sync with buildMessage() in index.html."""
    designation, company = _clean(req.designation), _clean(req.company)
    product, insurer = _clean(req.product), _clean(req.insurer)

    if not any([designation, company, product, insurer]):
        return "Send out a generic cold sales email addressed to Dear Sir/Madam."

    msg = "Send out a cold sales email addressed to Dear Sir/Madam"
    if designation:
        msg += f" {designation}"
    if company:
        msg += f" of {company}" if designation else f" at {company}"
    msg += ","
    if product:
        msg += f" pitching for the {product}"
    if insurer:
        msg += (" and" if product else "") + f" on behalf of {insurer}"
    return msg.rstrip(",") + "."


# ---------------------------------------------------------------- step 1: three drafts + AI pick
class NameFoundError(Exception):
    """The notebook's name-check guardrail found a personal name in the brief."""


@app.post("/api/sales-pitch/drafts")
async def sales_pitch_drafts(req: PitchRequest):
    message = build_message(req)
    try:
        # Same name check the notebook's input guardrail performs on the manager's input
        if AGENT["guardrail"] is not None:
            check = await Runner.run(AGENT["guardrail"], message)
            if getattr(check.final_output, "is_name_in_message", False):
                raise NameFoundError()

        with trace("Protected(Without name) Automated SDR"):
            runs = await asyncio.gather(*(Runner.run(a, message) for a in AGENT["sales_agents"]))
            drafts = [str(r.final_output or "").strip() for r in runs]

            judge_input = f"Brief: {message}\n\n" + "\n\n".join(
                f"--- Draft {i} ({DRAFT_STYLES[i - 1]}) ---\n{d}" for i, d in enumerate(drafts, start=1))
            choice = (await Runner.run(AGENT["judge"], judge_input)).final_output
    except (NameFoundError, InputGuardrailTripwireTriggered):
        raise HTTPException(422, "The name-check guardrail blocked this request because it found a personal name. "
                                 "Remove any person’s name from the designation or company fields and try again.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"The sales agents could not write the drafts: {exc}")

    best = choice.best_draft if choice and choice.best_draft in (1, 2, 3) else 1
    return {
        "message": message,
        "best": best,
        "reason": (choice.reason if choice else "") or "",
        "drafts": [{"number": i, "style": DRAFT_STYLES[i - 1], "body": d} for i, d in enumerate(drafts, start=1)],
    }


# ---------------------------------------------------------------- step 2: send the email the user chose
class SendRequest(BaseModel):
    draft: str


def _watch_tools(agent: Agent, queue: asyncio.Queue) -> Agent:
    """Copy of the Email Manager whose tools report to `queue` when they start and finish,
    so the page can show the final email while SendGrid is sending it."""
    wrapped = []
    for tool in agent.tools:
        if not isinstance(tool, FunctionTool):
            wrapped.append(tool)
            continue
        original, name = tool.on_invoke_tool, tool.name

        async def on_invoke(ctx, args, _orig=original, _name=name):
            await queue.put({"event": "tool_start", "tool": _name, "args": args})
            await asyncio.sleep(0.05)  # let the page receive the event before a blocking call (SendGrid)
            out = await _orig(ctx, args)
            await queue.put({"event": "tool_end", "tool": _name, "output": out})
            return out

        t = copy.copy(tool)
        t.on_invoke_tool = on_invoke
        wrapped.append(t)
    return agent.clone(tools=wrapped)


@app.post("/api/sales-pitch/send")
async def sales_pitch_send(req: SendRequest):
    draft = (req.draft or "").strip()
    if not draft:
        raise HTTPException(400, "Choose a draft to send.")

    queue: asyncio.Queue = asyncio.Queue()
    emailer = _watch_tools(AGENT["emailer"], queue)
    run_config = RunConfig(workflow_name="Protected(Without name) Automated SDR - send")

    async def run_emailer():
        try:
            await Runner.run(emailer, draft, run_config=run_config)
        except Exception as exc:
            await queue.put({"event": "error", "detail": f"The Email Manager could not finish: {exc}"})
        finally:
            await queue.put({"event": "done"})

    async def stream():
        line = lambda obj: json.dumps(obj) + "\n"
        task = asyncio.create_task(run_emailer())
        sent = False
        yield line({"stage": "preparing"})
        try:
            while True:
                ev = await queue.get()
                if ev["event"] == "done":
                    break
                if ev["event"] == "error":
                    yield line({"stage": "error", "detail": ev["detail"]})
                    return
                tool = ev.get("tool")
                if ev["event"] == "tool_start" and tool == "subject_writer":
                    yield line({"stage": "subject"})
                elif ev["event"] == "tool_start" and tool == "html_converter":
                    yield line({"stage": "html"})
                elif ev["event"] == "tool_start" and tool == "send_html_email":
                    try:
                        args = json.loads(ev["args"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    yield line({"stage": "sending", "subject": args.get("subject", ""),
                                "html_body": args.get("html_body", "")})
                elif ev["event"] == "tool_end" and tool == "send_html_email":
                    out = str(ev["output"])
                    if "success" in out.lower():
                        sent = True
                        yield line({"stage": "sent"})
                    else:
                        yield line({"stage": "error", "detail": f"SendGrid did not send the email: {out}"})
                        return
            if not sent:
                yield line({"stage": "error", "detail": "The Email Manager finished without calling send_html_email."})
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")
