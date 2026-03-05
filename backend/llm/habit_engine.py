import ollama
from typing import Optional
from schema import *

MODEL = "neural-chat:latest"


def _habit_insight_prompt(habit, quantity, streak, time_available, time_spent, quantity_done,
                          completed, user_notes):
    """
    Returns a formatted prompt for the LLM model (neural-chat) to provide insights on a habit completed by a user.
    :param habit: the user's habit
    :param quantity: the quantity of what is being done in this habit
    :param streak:
    :param time_available:
    :param time_spent:
    :param completed:
    :param user_notes:
    :return:
    """
    return (f"""
        You are a strict habit generation engine.
        
        Using principles from the book Atomic Habits by James Clear,
        provide constructive feedback and a rating on a scale of 1-5 (with 0.5 decimal increments allowed) with 5 being
        the highest score and 1 being the lowest. The feedback should highlight what the user did well and what the user
        can improve on. Treat user_notes as important information to determine how well the user did this habit 
        given their day.
        
        habit: {habit}
        suggested_quantity: {quantity}
        quantity_done: {quantity_done}
        habit_streak: {streak}
        time_available: {time_available}
        time_spent_on_activity: {time_spent}
        completed: {completed}
        user_notes: {user_notes}
        
        where habit is the user's habit, suggested_quantity is the quantity of the action that should be completed to 
        satisfy this habit, quantity_done is how much of the activity did the user actually complete, habit_streak is
        the number of consecutive days the user completed this activity, time_available is how much free-time the user 
        had during the day, time_spent_on_activity is how much time the user spent on this activity, completed is 
        whether the user completed this habit, user_notes is the user's perspective to help you understand whether the 
        user did well given their situation for the day.
        
        Return ONLY valid JSON.
        Do NOT include explanations.
        Do NOT include markdown.
        Do NOT include additional commentary.
        
        The JSON must follow exactly this structure:
        
        {{
            "overview": string,
            "praises": string,
            "suggestions": string,
            "score": float
        }}
        
        Where:
        \'overview\' is how you think the user did overall (in 1-2 sentences) 
        \'praises\' is what you think the user did great on when doing this habit given their situation,
        \'quantity\' is what you think the user could have done better on with this habit in the future,
        \'score\' is how you rate the user's habit work today on a scale of 1-5 with 1 being poor and 5 being perfect,
        with 0.5 decimal increments allowed.
        Provide feedback as if you are a habit coach: be warm, honest, but try to keep it brief.
    """)


def _generate_habit_prompt(goal: str, time_availability: str, max_deadline: str = "No deadline"):
    """
    Returns a formatted prompt as input for the LLM model (neural-chat) to generate a habit
    :param goal:
    :param time_availability:
    :param max_deadline:
    :return:
    """
    return (f"""You are a strict habit generation engine.
        
        Using principles from the book Atomic Habits by James Clear,
        generate a realistic and honest habit.
        
        User goal: {goal}
        Expected date to achieve goal by: {max_deadline}
        Time availability: {time_availability}
        
        Return ONLY valid JSON.
        Do NOT include explanations.
        Do NOT include markdown.
        Do NOT include additional commentary.
        
        The JSON must follow exactly this structure:
        
        {{
            "habit": string,
            "frequency": string,
            "quantity": string,
            "notes": string
        }}
        
        Where \'habit\' is what the user should do, \'frequency\' is how many times per week this habit should be done,
        \'quantity\' is a measure of what should be done in the activity (example: read 10 pages), \'notes\' contains
        any further instructions and suggestions to do this habit effectively.
        Be realistic and direct.
        """)


class HabitEngine:
    """
    A LLM-powered habit-generation and coaching engine.
    """

    def __init__(self, model: str = MODEL) -> None:
        """
        Initializes the engine.
        """
        self._client = ollama.Client()
        self._model = model

    def _generate_structured_response(self, prompt, schema, temperature: float = 0.4):
        """
        A helper function to generate and return a JSON-structured response.
        :param prompt:
        :param schema:
        :param temperature:
        :return:
        """
        import json
        try:
            response_raw = self._client.generate(
                model=self._model,
                prompt=prompt,
                format=schema,
                options={"temperature": temperature}
            ).response
            return json.loads(response_raw)
        except json.JSONDecodeError:
            raise ValueError("Model returned invalid JSON")

    def generate_habit(self, goal: str, time_availability: str, max_deadline: Optional[str] = None):
        """
        Returns a habit with the suggested frequency along with other info to help guide the user to better
        execute this habit daily.
        :param goal: the user's end goal (this habit will help get the user there)
        :param time_availability: How available the user is (# hours per day on average)
        :param max_deadline: When the user wants to achieve this goal by (if they have a "deadline")
        :return: the suggested habit based on all the user's needs
        """
        max_deadline = max_deadline or "no deadline"
        prompt = _generate_habit_prompt(goal, time_availability, max_deadline)

        resp_schema = {
            "type": "object",
            "properties": {
                "habit": {"type": "string"},
                "frequency": {"type": "string"},
                "quantity": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["habit", "frequency", "quantity", "notes"],
            "additionalProperties": False
        }

        response = self._generate_structured_response(prompt, resp_schema, 0.4)
        response_validated = validate_schema(response, HABIT_GEN_REQUIRED_KEYS)
        return response_validated

    # TODO: CREATE A FUNCTION RESPONSIBLE FOR HABIT-COMPLETION INSIGHTS
    def habit_insight(self, habit_info: dict):
        """
        Returns insights on a habit "activity" that a user has just completed.
        :param habit_info: the information about the habit, including the frequency, suggested quantity, the "streak"
        and any user notes.
        :return: insight on a habit, providing what the user did great and what they can improve on.
        """
        habit, quantity = habit_info["habit"], habit_info["quantity"]
        streak = habit_info["streak"]
        time_available = habit_info["time_available"]
        time_spent = habit_info["time_spent"]
        quantity_done = habit_info["quantity_done"]
        completed = habit_info["completed"]
        user_notes = habit_info["user_notes"]

        resp_schema = {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "praises": {"type": "string"},
                "suggestions": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["overview", "praises", "suggestions", "score"],
            "additionalProperties": False
        }

        prompt = _habit_insight_prompt(habit, quantity, streak, time_available, time_spent, quantity_done, completed,
                                       user_notes)

        response = self._generate_structured_response(prompt, resp_schema, 0.6)
        response_validated = validate_schema(response, HABIT_INS_REQUIRED_KEYS)
        return response_validated


# TODO: CREATE A TEST SCRIPT TO SEE IF GENERATE_HABIT AND HABIT_INSIGHT WORKS
if __name__ == "__main__":
    from datetime import datetime
    import json

    print("\n=== ATOMIFY TEST SCRIPT ===\n")

    engine = HabitEngine()

    # -------------------------
    # Test 1: Habit Generation
    # -------------------------

    goal = "Become confident at public speaking"
    time_availability = "1 hour per day"
    deadline = datetime(2026, 5, 4).strftime("%Y-%m-%d")

    print("Generating habit...\n")

    try:
        habit = engine.generate_habit(
            goal=goal,
            time_availability=time_availability,
            max_deadline=deadline
        )

        print("Generated Habit:")
        print(json.dumps(habit, indent=4))

    except Exception as e:
        print("Habit generation failed:", e)
        exit()

    # -------------------------
    # Test 2: Simulate Completion
    # -------------------------

    print("\nSimulating habit completion...\n")

    habit_activity = {
        "habit": habit["habit"],
        "quantity": habit["quantity"],
        "streak": 3,
        "time_available": "1 hour",
        "time_spent": "45 minutes",
        "quantity_done": "8 minutes of speaking practice",
        "completed": True,
        "user_notes": "Felt nervous but improved after the first few minutes."
    }

    try:
        insight = engine.habit_insight(habit_activity)

        print("Habit Insight:")
        print(json.dumps(insight, indent=4))

    except Exception as e:
        print("Habit insight failed:", e)

    print("\n=== TEST COMPLETE ===")
