"""Metadata Filtering + Self-query

不要用 LangChain SelfQueryRetriever 黑盒。
用 Instructor + Pydantic 自己做，可控可測。
"""
from datetime import date
from typing import Literal, Optional, List
import anthropic
import instructor
from pydantic import BaseModel, Field

anth_client = instructor.from_anthropic(anthropic.Anthropic())


class DateRange(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None


class SearchFilter(BaseModel):
    semantic_query: str = Field(description="去除 filter 訊息後的純語意")
    category: Optional[Literal["blog", "paper", "docs", "news"]] = None
    source: Optional[List[str]] = None
    date_range: Optional[DateRange] = None
    min_length: Optional[int] = None


def extract_filter(user_query):
    return anth_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""今天是 {date.today()}。從查詢中抽 filter。
"最近/近期" → 最近 3 個月；"2024 年" → start=2024-01-01, end=2024-12-31；
"深入/深度" → min_length=2000；無指定留 None。
semantic_query 是去除 filter 後的純語意。
查詢：{user_query}""",
            }
        ],
        response_model=SearchFilter,
    )


def to_chroma_where(f: SearchFilter):
    conds = []
    if f.category:
        conds.append({"category": {"$eq": f.category}})
    if f.source:
        conds.append({"source": {"$in": f.source}})
    if f.date_range and f.date_range.start:
        conds.append({"date": {"$gte": f.date_range.start.isoformat()}})
    if f.date_range and f.date_range.end:
        conds.append({"date": {"$lte": f.date_range.end.isoformat()}})
    if f.min_length:
        conds.append({"length": {"$gte": f.min_length}})
    return {"$and": conds} if len(conds) > 1 else (conds[0] if conds else {})
