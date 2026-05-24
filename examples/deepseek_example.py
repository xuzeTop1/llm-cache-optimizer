"""DeepSeek prefix-cache example using the OpenAI-compatible adapter."""

import os

from llm_cache_optimizer import CacheAwareOpenAI


client = CacheAwareOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-xxx"),
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
)

client.add_core(
    "You are a helpful coding assistant. Keep answers concise, practical, "
    "and grounded in Python examples. Explain tradeoffs when relevant. "
    "This stable system layer should be reused across turns so DeepSeek can "
    "match the prompt prefix and serve cached tokens."
)
client.add_static_context("Project docs here. Keep this stable across the session.")

questions = ["Explain decorators", "Show a retry pattern", "Write a cache layer"]

for question in questions:
    response = client.chat(question)
    print(response.choices[0].message.content)
    print(client.cache_report())
