"""LLM-as-Judge（Hamel 範式：Binary pass/fail + critique）

為什麼 Binary 贏 Likert：
- 消除 3 跟 4 誰對的爭議
- 強迫明確定義 failure mode
- 可直接算 TPR/TNR/kappa
- Stakeholder 終究只問 pass rate
"""
import json
from anthropic import Anthropic


JUDGE_PROMPT = """You are an evaluator for a Dell enterprise support chatbot.

## PASS criteria (must satisfy ALL)
1. Every factual claim in the answer is supported by retrieved context.
2. The answer directly addresses the question.
3. Does NOT make SLA promises the system can't verify.
4. If context lacks the answer, chatbot says so rather than guessing.

## FAIL if any criterion is violated.

## Output (STRICT JSON, nothing else)
{{
  "critique": "<2-4 sentences explaining specific reason>",
  "failure_mode": "<one of: hallucination, off_topic, over_promise, missing_refusal, other>",
  "label": "<PASS or FAIL>"
}}

## Input
Question: {question}
Retrieved context: {context}
Chatbot answer: {answer}"""


def judge(question, answer, contexts):
    msg = Anthropic().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    context="\n".join(f"- {c}" for c in contexts),
                    answer=answer,
                ),
            }
        ],
    )
    raw = msg.content[0].text.strip().strip("`").removeprefix("json").strip()
    return json.loads(raw)
