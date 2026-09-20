import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class TutorModel:

    def __init__(self):
        print("Initializing Gemini AI Tutor...")

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

        self.client = genai.Client(api_key=api_key)

        self.model = "gemini-3.6-flash"

        print("Gemini model:", self.model)
        print("Gemini AI Tutor ready.")

    def build_system_instruction(
        self,
        language="english",
        level="beginner",
        mode="tutor"
    ):

        language_map = {
            "english": "English",
            "telugu": "Telugu",
            "hindi": "Hindi"
        }

        language_name = language_map.get(
            language.lower(),
            "English"
        )

        if mode.lower() == "code tutor":
            mode_instruction = """
You are a programming tutor.

When explaining programming:
- Explain the logic step by step.
- Give clean and correct code.
- Explain the code after the code.
- Mention time and space complexity when relevant.
- Use simple examples.
"""
        else:
            mode_instruction = """
You are an educational AI tutor.

Teach concepts clearly and patiently.
Use simple explanations, examples, analogies,
and step-by-step reasoning when useful.
"""

        return f"""
You are an AI Tutor for students.

Preferred language: {language_name}
Learning level: {level}

{mode_instruction}

Important rules:

1. Answer the student's actual question.
2. Never give unrelated information.
3. Keep explanations appropriate for the student's level.
4. Use headings and bullet points when useful.
5. Explain difficult concepts from the basics.
6. Use real-world examples when useful.
7. If the student asks for code, provide working code.
8. If the student asks for an exam answer, structure it clearly.
9. Use previous conversation context when provided.
10. Respond in {language_name}.
"""

    def generate(
        self,
        message,
        language="english",
        level="beginner",
        mode="tutor",
        history=None,
        max_new_tokens=500,
        temperature=0.7,
        top_k=40
    ):

        system_instruction = self.build_system_instruction(
            language=language,
            level=level,
            mode=mode
        )

        # Build conversation as plain text.
        conversation = system_instruction + "\n\n"

        if history:

            conversation += "Previous conversation:\n"

            for item in history[-10:]:

                role = item.get("role", "user")
                content = item.get("content", "").strip()

                if not content:
                    continue

                if role == "user":
                    conversation += f"Student: {content}\n"

                elif role == "assistant":
                    conversation += f"Tutor: {content}\n"

            conversation += "\n"

        conversation += f"Student: {message.strip()}\n"
        conversation += "Tutor:"

        response = self.client.models.generate_content(
            model=self.model,
            contents=conversation
        )

        answer = response.text

        if not answer:
            return "Sorry, I could not generate a response."

        return answer.strip()