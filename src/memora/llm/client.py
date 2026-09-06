# Thin wrapper over the `ollama` client, pointed at
# memora.config.settings.ollama_host / llm_model. Swapping providers means
# swapping this module, not the callers.


def generate(prompt: str) -> str:
    raise NotImplementedError
