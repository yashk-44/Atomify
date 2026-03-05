class KeyMatchError(Exception):
    def __init__(self, missing_keys=None):
        self.missing_keys = missing_keys or set()

    def __str__(self):
        if self.missing_keys:
            return f"LLM response missing keys: {self.missing_keys}"
        return "LLM response not matching the exact key specifications!"
