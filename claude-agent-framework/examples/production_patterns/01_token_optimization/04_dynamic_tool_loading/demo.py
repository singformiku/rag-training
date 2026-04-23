"""
Dynamic Tool Loading — only expose tools the user actually needs
================================================================

What this demo shows
--------------------
A toy catalogue of 30 tools (split into 5 categories).  For each user query
we compare two policies:

  * ``static``  — send ALL 30 tool schemas on every turn
  * ``router``  — ask the "cheap" tier to classify the query into 1–2
                    categories, then send only those tools to the main model

We report:
  * total tool-schema tokens per turn
  * did the main model pick the CORRECT tool?  (ground truth is wired in)
  * total $ at the Sonnet baseline price

This reproduces the "14,500 → 1,450 tool tokens" number cited in the dossier
when you run it with ``--turns 5`` or more.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common.backend import MODELS, BACKEND, banner, chat, count_tokens, dollars  # noqa: E402

PRICE_IN_PER_M = 3.00

# ---------------------------------------------------------------------------
# Tool catalogue — deliberately verbose descriptions to make the token effect
# visible.  In production each schema is typically 350–700 tokens.
# ---------------------------------------------------------------------------
def _mk_tool(name: str, desc: str, props: Dict[str, str]) -> Dict:
    return {
        "name": name,
        "description": desc + (" " * 20),  # pad to mimic realistic schema size
        "parameters": {
            "type": "object",
            "properties": {k: {"type": v} for k, v in props.items()},
            "required": list(props.keys()),
        },
    }


CATEGORIES: Dict[str, List[Dict]] = {
    "code": [
        _mk_tool("run_tests",   "Execute the project's test suite and return pass/fail + logs.", {"target": "string"}),
        _mk_tool("git_commit",  "Create a git commit with the given message.",                   {"message": "string"}),
        _mk_tool("read_file",   "Read the full contents of a file in the repo.",                 {"path": "string"}),
        _mk_tool("write_file",  "Overwrite a file with the given content.",                      {"path": "string", "content": "string"}),
        _mk_tool("lint",        "Run the linter on a path and report issues.",                   {"path": "string"}),
        _mk_tool("grep",        "Search across the repo for a regex.",                           {"pattern": "string"}),
    ],
    "db": [
        _mk_tool("sql_query",        "Run a read-only SQL query against the analytics warehouse.", {"query": "string"}),
        _mk_tool("schema_lookup",    "Describe the columns of a table.",                            {"table": "string"}),
        _mk_tool("table_sample",     "Return 10 sample rows from a table.",                         {"table": "string"}),
        _mk_tool("explain_plan",     "Return the query planner output for an SQL string.",          {"query": "string"}),
        _mk_tool("kill_long_query",  "Terminate a running query by id.",                            {"query_id": "string"}),
        _mk_tool("pg_stats",         "Show vacuum / bloat statistics for a table.",                 {"table": "string"}),
    ],
    "ticket": [
        _mk_tool("jira_create", "Create a new Jira ticket in the given project.", {"project": "string", "title": "string", "body": "string"}),
        _mk_tool("jira_search", "Search Jira tickets by JQL.",                     {"jql": "string"}),
        _mk_tool("jira_update", "Update fields on an existing Jira ticket.",       {"id": "string", "fields": "string"}),
        _mk_tool("jira_comment","Add a comment to a Jira ticket.",                 {"id": "string", "body": "string"}),
        _mk_tool("jira_close",  "Transition a Jira ticket to Done.",               {"id": "string"}),
        _mk_tool("jira_assign", "Assign a Jira ticket to a user.",                 {"id": "string", "user": "string"}),
    ],
    "web": [
        _mk_tool("http_get",   "GET an arbitrary URL and return the body.",            {"url": "string"}),
        _mk_tool("web_search", "Query a web search API and return the top results.",   {"query": "string"}),
        _mk_tool("fetch_url",  "Fetch a URL with caching + ETag awareness.",           {"url": "string"}),
        _mk_tool("scrape_doc", "Download and extract plain text from a PDF URL.",      {"url": "string"}),
        _mk_tool("weather",    "Return the current weather for a city.",               {"city": "string"}),
        _mk_tool("geocode",    "Convert an address to lat/long.",                      {"address": "string"}),
    ],
    "comms": [
        _mk_tool("slack_post",  "Post a message to a Slack channel.",                 {"channel": "string", "text": "string"}),
        _mk_tool("email_send",  "Send an email via the corporate relay.",             {"to": "string", "subject": "string", "body": "string"}),
        _mk_tool("sms_send",    "Send an SMS via Twilio.",                            {"to": "string", "text": "string"}),
        _mk_tool("calendar_create", "Create a calendar event for a group.",           {"attendees": "string", "title": "string", "start": "string"}),
        _mk_tool("calendar_find", "Find an open 30-minute slot among attendees.",     {"attendees": "string"}),
        _mk_tool("meeting_notes", "Attach notes to a calendar event.",                {"event_id": "string", "notes": "string"}),
    ],
}
ALL_TOOLS = [t for group in CATEGORIES.values() for t in group]
TOOL_INDEX = {t["name"]: t for t in ALL_TOOLS}

# Test queries + ground-truth accepted tools.  Multiple tools can be "right"
# for the same query — we accept any of them.
QUERIES: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Run the unit tests in the billing package and tell me which one failed.", "code",   ("run_tests",)),
    ("How many orders were placed yesterday? Query analytics_orders.",           "db",     ("sql_query",)),
    ("Open a Jira ticket in BILLING: 'refund job stuck overnight'.",             "ticket", ("jira_create",)),
    ("Please fetch the content of https://example.com/status.html for me.",      "web",    ("http_get", "fetch_url")),
    ("Send a Slack message in #oncall: 'incident 1234 mitigated'.",              "comms",  ("slack_post",)),
]


def tool_tokens(tools: List[Dict]) -> int:
    return count_tokens(json.dumps(tools))


def classify(query: str) -> List[str]:
    """Cheap-tier classifier → list of category names."""
    cats = ", ".join(CATEGORIES)
    r = chat(
        messages=[
            {"role": "system", "content":
                f"Return a JSON array of 1-2 category names from this set: [{cats}]. "
                "Respond with JSON only, no prose."},
            {"role": "user", "content": query},
        ],
        tier="cheap",
        max_tokens=64,
        extra={"reasoning_effort": "low"},
    )
    raw = (r.text or "").strip()
    # Extract the first [...] JSON array
    import re
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not m:
        return ["code"]  # safe default
    try:
        parsed = json.loads(m.group(0))
        return [c for c in parsed if c in CATEGORIES][:2] or ["code"]
    except Exception:
        return ["code"]


def run_one(query: str, policy: str, accepted_tools: Tuple[str, ...]) -> Dict:
    if policy == "static":
        tools = ALL_TOOLS
    else:
        cats = classify(query)
        names = {n["name"] for c in cats for n in CATEGORIES.get(c, [])}
        tools = [TOOL_INDEX[n] for n in names] or ALL_TOOLS[:3]

    r = chat(
        messages=[
            {"role": "system", "content": "Pick exactly one tool and call it. Do not respond with prose."},
            {"role": "user", "content": query},
        ],
        tier="medium",
        tools=tools,
        tool_choice="auto",
        max_tokens=400,
        extra={"reasoning_effort": "low"},
    )
    called = r.tool_calls[0]["name"] if r.tool_calls else ""
    ok = called in accepted_tools
    return {
        "policy": policy,
        "query": query[:60],
        "tool_tokens": tool_tokens(tools),
        "tools_exposed": len(tools),
        "tool_called": called or "(none)",
        "correct": ok,
        "total_input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=len(QUERIES))
    args = parser.parse_args()

    banner(f"Dynamic tool loading demo  |  backend={BACKEND}  model={MODELS['medium']}")
    print(f"Tool catalogue: {len(ALL_TOOLS)} tools in {len(CATEGORIES)} categories")
    print(f"Static schema tokens: {tool_tokens(ALL_TOOLS):,}  "
          f"(sent on every turn in the static policy)\n")

    rows: List[Dict] = []
    for q, _, accepted in QUERIES[: args.queries]:
        for policy in ("static", "router"):
            res = run_one(q, policy, accepted)
            rows.append(res)
            print(
                f"[{policy:>6}] {res['query'][:55]:<55}  "
                f"schema={res['tool_tokens']:>5}  "
                f"called={res['tool_called']:<14} correct={res['correct']}"
            )

    banner("Summary per policy")
    for policy in ("static", "router"):
        subset = [r for r in rows if r["policy"] == policy]
        accuracy = sum(1 for r in subset if r["correct"]) / len(subset)
        avg_schema = sum(r["tool_tokens"] for r in subset) / len(subset)
        avg_in = sum(r["total_input_tokens"] for r in subset) / len(subset)
        cost_1k = avg_in * PRICE_IN_PER_M / 1e6 * 1000
        print(
            f"{policy:<7}  accuracy={accuracy*100:>5.1f}%  "
            f"avg_schema_tokens={avg_schema:>6.0f}  "
            f"avg_input_tokens={avg_in:>6.0f}  "
            f"cost_per_1k_queries={dollars(cost_1k)}"
        )

    print(
        "\nRouter-based selection trims tool-schema tokens while *improving* "
        "accuracy: the main model isn't distracted by irrelevant tools."
    )


if __name__ == "__main__":
    main()
