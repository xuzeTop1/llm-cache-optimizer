"""Session memory demo with local summary and keyword extraction."""

from llm_cache_optimizer import CacheAwareClient


client = CacheAwareClient()
client.add_core("You are a cache-aware agent runtime.")
client.add_static_context("Project: llm-cache-optimizer")

client.chat("We are turning this repository from best practices into a Python runtime.")
client.history.append(
    {
        "role": "assistant",
        "content": "We added CanonicalSerializer, PromptBuilder, CacheAwareClient, and metrics.",
    }
)
client.chat("Next we need an OpenAI adapter and a demo that summarizes memory.")

memory = client.refresh_memory()

print(memory)
print()
print(client.memory_report())
print()
print(client.messages("Use the memory to plan the next change."))
