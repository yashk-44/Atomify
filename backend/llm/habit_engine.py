import ollama
import json
from typing import Optional
from schema import *

MODEL = "neural-chat:latest"

# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

HABIT_COACH_SYSTEM_PROMPT = """
You are Atomify, a habit coach grounded in behavioral science.

Your coaching philosophy draws directly from James Clear's Atomic Habits:
- Habits must be obvious, attractive, easy, and satisfying (the Four Laws)
- Identity-based habits outlast outcome-based ones ("I am a reader" vs "I want to read more")
- Habit stacking (after I do X, I will do Y) dramatically improves consistency
- Reducing friction is more powerful than increasing motivation
- The goal is not perfection — it is not breaking the chain

Your tone: warm but honest. You do not sugarcoat poor effort, but you never discourage.
Your responses: concise, specific, and always grounded in one of the Four Laws when relevant.
Your outputs: always valid JSON, exactly matching the schema provided. No markdown. No commentary.
"""

HABIT_GENERATION_SYSTEM_PROMPT = """
You are Atomify's habit generation engine, grounded in behavioral science.

You generate ONE specific, realistic habit based on the user's goal, time, and deadline.

Rules you must follow when generating a habit:
1. Apply the Four Laws of Behavior Change (Atomic Habits): make it obvious, attractive, easy, satisfying
2. Prefer habit stacking — anchor the habit to an existing routine the user likely has
3. Be specific. Not "exercise more" — "do 10 push-ups after your morning coffee"
4. Be realistic given the user's available time. Do not over-prescribe.
5. If a deadline is provided, calibrate the habit's intensity to that timeframe
6. Suggest an identity statement the user can internalize (e.g. "I am someone who...")

Your outputs: always valid JSON, exactly matching the schema provided. No markdown. No commentary.
"""


# ─────────────────────────────────────────────
# USER MESSAGE BUILDERS
# ─────────────────────────────────────────────

def _build_habit_generation_user_message(goal, time_availability, skill_level, strengths, internals, externals, max_deadline) -> str:
    return (f"""
Generate a habit for the following user based on the user's GoalObject and the user's PrelimInfo:

<GoalObject>
Goal: {goal}
Target deadline: {max_deadline}
Time available per day: {time_availability}
</GoalObject>

<PrelimInfo>
Skill level: {skill_level}
Strengths: {strengths}
Internal Limitations: {internals}
External Limitations: {externals}
</PrelimInfo>

Refer to the definitions of the PrelimInfo keys using the PrelimInfoDefs below

<PrelimInfoDefs>
Skill level: <how skilled the user thinks they are in the habit that connects to their final goal>,
Strengths: <what the user thinks they are "good" at that is related to their goal>,
Internal Limitations: <what the user thinks they are "poor" at that is related to their goal>,
External Limitations: <any external factors that might "limit" the user's ability to perform their habit - this should 
determine how "simple" the habit should be at the start>
</PrelimInfoDefs>

Return ONLY valid JSON matching the HabitObject schema:

<HabitObject>
{{
    "habit": "<specific action to do — be concrete, not vague>",
    "identity_statement": "<an identity the user can adopt, e.g. 'I am someone who reads daily'>",
    "habit_stack": "<an existing routine to attach this habit to, e.g. 'After my morning coffee, I will...'>",
    "frequency": "<how many times per week>",
    "quantity": "<measurable target per session, e.g. '10 pages' or '20 minutes' — this should not be blank - use the 
    value from "habit" whenever needed>",
    "suggested_time": "<realistic time to complete one session, e.g. '15 minutes'>",
    "friction_reduction": "<one concrete way to make this habit easier to start>",
    "notes": "<any other relevant coaching notes>"
}}
</HabitObject>

Make sure to follow the Guidelines below:

<Guidelines>
"quantity" recommends the "how much" of the suggested habit, so any quantifiable aspects for the recommended 
habit should not be in the "habit" field itself, rather in the "quantity" field. Example, if the suggested habit is:
"practice speaking in a mirror for 15 minutes with atleast 3 topics of your choice" then the resulting HabitObject 
should contain "habit" as "practice speaking in a mirror with multiple topics", "quantity" as "atleast 3 topics", 
and "suggested_time" as "15 minutes". 

"quantity" should NEVER be empty.
</Guidelines>
""")


def _build_habit_insight_user_message(
        habit: str, estimated_time: str, quantity: str, streak: int,
        time_available: str, time_spent: str, quantity_done: str,
        completed: bool, user_notes: str
) -> str:
    # Determine completion context for the model
    quantity_context = (
        f"Target quantity: {quantity}\nQuantity completed: {quantity_done}"
        if quantity and quantity_done
        else "No quantity target was set for this habit."
    )

    time_context = (
        f"Estimated session time: {estimated_time}\nTime actually spent: {time_spent}"
        if estimated_time and time_spent
        else "No time tracking data available."
    )

    return f"""
Evaluate the following Habit using the SessionDetails and provide coaching feedback based on the UserNotes.

<Habit>
Habit: {habit}
Current streak: {streak} consecutive days
Marked as completed: {completed}
</Habit>

<SessionDetails>
{time_context}
{quantity_context}
Free time the user had today: {time_available}
</SessionDetails>

<UserNotes>
"{user_notes}"
</UserNotes>

--- YOUR TASK ---
1. Read the user's account carefully. Their context matters — a good session under hard circumstances 
   deserves more credit than a perfect session on an easy day.
2. Determine if the user genuinely completed the spirit of the habit, not just the letter of it.
3. Score fairly. Reserve 5.0 for exceptional effort. A solid, consistent session is 3.5–4.0.
4. Ground at least one piece of feedback in the Four Laws (obvious / attractive / easy / satisfying).

Return ONLY valid JSON matching this exact InsightObject schema:

<InsightObject>
{{
    "overview": "<1-2 sentences: how the user did overall. Include specifics from session data and their notes>",
    "praises": "<what the user did well — be specific, not generic. Reference their notes if relevant>",
    "suggestions": "<one concrete thing the user can do differently next time — grounded in Atomic Habits principles>",
    "atomic_habits_principle": "<which of the Four Laws is most relevant to this session and why, in 1 sentence>",
    "score": <float between 1.0 and 5.0, increments of 0.5>
}}
</InsightObject>

Ensure that the InsightObject follows these Guidelines

<Guidelines>
1. the praises should focus on what the user did well on the habit whilst also taking into consideration the additional context which
may not be related to the habit itself but can explain why or why not the user performed well.
2. Similarly with suggestions it should focus on what the user did not do well on the habit whilst also taking into consideration the additional context which
may not be related to the habit itself but can explain why or why not the user performed well.
3. Make sure that the "praises" and "suggestions" address the user using second-person perspective (so using "you" instead of "they", etc.)
</Guidelines>
"""


# ─────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────

class HabitEngine:
    """
    A LLM-powered habit generation and coaching engine.
    Uses a system/user message split for better model grounding.
    """

    def __init__(self, model: str = MODEL) -> None:
        self._client = ollama.Client()
        self._model = model

    def _generate_structured_response(
            self,
            system_prompt: str,
            user_message: str,
            schema: dict,
            temperature: float = 0.4
    ) -> dict:
        try:
            stream = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_message.strip()},
                ],
                stream=True,
                format=schema,
                options={"temperature": temperature}
            )

            content = ""
            for chunk in stream:
                piece = chunk.message.content
                if piece:
                    print(piece, end="", flush=True)
                    content += piece

            print()  # newline after stream ends
            return json.loads(content)

        except json.JSONDecodeError:
            raise ValueError("Model returned invalid JSON — consider lowering temperature or simplifying the prompt.")

    def generate_habit(
            self,
            goal: str,
            time_availability: str,
            skill_level: str,
            strengths: str,
            internals: str,
            externals: str,
            max_deadline: Optional[str] = None,
    ) -> dict:
        """
        Generates a personalized habit grounded in Atomic Habits principles.

        :param goal: The user's end goal (e.g. "get fit", "learn guitar")
        :param time_availability: How much free time the user has per day (e.g. "1 hour")
        :param skill_level: How skilled the user is in the area related to their goal
        :param strengths: What the user is good at related to their goal
        :param internals: Internal limitations (e.g. mindset, discipline challenges)
        :param externals: External limitations (e.g. schedule, equipment, environment)
        :param max_deadline: Optional target date (e.g. "2026-12-01")
        :return: Structured habit recommendation
        """
        deadline = max_deadline or "no specific deadline"
        user_message = _build_habit_generation_user_message(
            goal, time_availability, skill_level, strengths, internals, externals, deadline
        )

        schema = {
            "type": "object",
            "properties": {
                "habit": {"type": "string"},
                "identity_statement": {"type": "string"},
                "habit_stack": {"type": "string"},
                "frequency": {"type": "string"},
                "quantity": {"type": "string"},
                "estimated_time": {"type": "string"},
                "friction_reduction": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": [
                "habit", "identity_statement", "habit_stack",
                "frequency", "quantity", "estimated_time",
                "friction_reduction", "notes"
            ],
            "additionalProperties": False
        }

        response = self._generate_structured_response(
            system_prompt=HABIT_GENERATION_SYSTEM_PROMPT,
            user_message=user_message,
            schema=schema,
            temperature=0.5  # Slightly higher — habit generation benefits from some creativity
        )

        return validate_schema(response, HABIT_GEN_REQUIRED_KEYS)

    def habit_insight(self, habit_info: dict) -> dict:
        """
        Evaluates a completed (or attempted) habit session and returns coaching feedback.

        :param habit_info: Dict containing habit details and session data.
            Required keys: habit, estimated_time, quantity, streak,
                           time_available, time_spent, quantity_done,
                           completed, user_notes
        :return: Structured coaching feedback with score
        """
        user_message = _build_habit_insight_user_message(
            habit=habit_info["habit"],
            estimated_time=habit_info["estimated_time"],
            quantity=habit_info["quantity"],
            streak=habit_info["streak"],
            time_available=habit_info["time_available"],
            time_spent=habit_info["time_spent"],
            quantity_done=habit_info["quantity_done"],
            completed=habit_info["completed"],
            user_notes=habit_info["user_notes"],
        )

        schema = {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "praises": {"type": "string"},
                "suggestions": {"type": "string"},
                "atomic_habits_principle": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["overview", "praises", "suggestions", "atomic_habits_principle", "score"],
            "additionalProperties": False
        }

        response = self._generate_structured_response(
            system_prompt=HABIT_COACH_SYSTEM_PROMPT,
            user_message=user_message,
            schema=schema,
            temperature=0.6  # Slightly higher for more nuanced coaching language
        )

        return validate_schema(response, HABIT_INS_REQUIRED_KEYS)


# ─────────────────────────────────────────────
# TEST SCRIPT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime

    print("\n=== ATOMIFY HABIT ENGINE TEST ===\n")

    engine = HabitEngine()

    # ── Test 1: Habit Generation ──────────────────────────

    goal = "Become confident at public speaking"
    time_availability = "1 hour per day"
    deadline = datetime(2026, 8, 1).strftime("%Y-%m-%d")
    skill_level = "Beginner — little to no experience with public speaking"
    strengths = "Good at writing and preparing notes, comfortable in one-on-one conversations"
    internals = "Freezes under pressure, speaks too fast when nervous, struggles with eye contact"
    externals = "Works full-time, limited time on weekdays, no access to a speaking club currently"

    print(f"Goal:                 {goal}")
    print(f"Skill level:          {skill_level}")
    print(f"Time available:       {time_availability}")
    print(f"Deadline:             {deadline}")
    print(f"Strengths:            {strengths}")
    print(f"Internal limitations: {internals}")
    print(f"External limitations: {externals}")
    print("\nGenerating habit...\n")

    try:
        habit = engine.generate_habit(
            goal=goal,
            time_availability=time_availability,
            skill_level=skill_level,
            strengths=strengths,
            internals=internals,
            externals=externals,
            max_deadline=deadline
        )
        print("\nGenerated Habit:")
        print(json.dumps(habit, indent=4))
    except Exception as e:
        print("Habit generation failed:", e)
        exit()

    # ── Test 2: Habit Insight ─────────────────────────────

    print("\n" + "─" * 40)
    print("Simulating habit session completion...\n")

    habit_activity = {
        "habit": habit["habit"],
        "estimated_time": habit["estimated_time"],
        "quantity": habit.get("quantity", ""),
        "streak": 3,
        "time_available": "1 hour",
        "time_spent": "45 minutes",
        "quantity_done": "12 minutes",
        "completed": True,
        "user_notes": "Felt nervous at first but pushed through. Did a 3-minute mirror speech and one recording. Hard day at work before this."
    }

    print(f"Habit:          {habit_activity['habit']}")
    print(f"Estimated time: {habit_activity['estimated_time']}")
    print(f"Time spent:     {habit_activity['time_spent']}")
    print(f"Quantity target:{habit_activity['quantity']}")
    print(f"Quantity done:  {habit_activity['quantity_done']}")
    print(f"Streak:         {habit_activity['streak']} days")
    print(f"Completed:      {habit_activity['completed']}")
    print(f"Notes:          {habit_activity['user_notes']}")
    print("\nGenerating coaching feedback...\n")

    try:
        insight = engine.habit_insight(habit_activity)
        print("\nCoaching Feedback:")
        print(json.dumps(insight, indent=4))
    except Exception as e:
        print("Habit insight failed:", e)

    print("\n=== TEST COMPLETE ===")
