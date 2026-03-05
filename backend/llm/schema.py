from llm_errors import KeyMatchError

HABIT_GEN_REQUIRED_KEYS = {"habit", "frequency", "quantity", "notes"}
HABIT_INS_REQUIRED_KEYS = {"overview", "praises", "suggestions", "score"}


def validate_schema(data, required_keys):
    if not isinstance(data, dict):
        raise TypeError("LLM response is not in the form of a dictionary")

    missing = required_keys - data.keys()
    if missing:
        raise KeyMatchError(missing)

    return data
