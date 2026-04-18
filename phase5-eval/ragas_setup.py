"""Ragas + Claude setup

pip install "ragas>=0.2.14,<0.4" langchain-anthropic langchain-openai datasets

⚠️ judge LLM 最好與 generator 不同家（避免 self-preference bias）
Generator=Claude，judge 就用 GPT-4o 或 Claude Opus（不同 model variant）
"""
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# judge temperature 一定 0
_chat = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=2048,
)
evaluator_llm = LangchainLLMWrapper(_chat)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small")
)
