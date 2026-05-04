class PromptLLMChain:
    """Small compatibility wrapper for legacy `.run(...)` call sites."""

    def __init__(self, llm, prompt):
        self._chain = prompt | llm

    def invoke(self, inputs):
        return self._chain.invoke(inputs)

    def run(self, inputs):
        result = self.invoke(inputs)
        return getattr(result, "content", result)
