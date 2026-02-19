import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, List, Dict

load_dotenv()


class TestOutput(BaseModel):
    test_cases: Optional[List[Dict]]
    reasoning: Optional[str]


class TestCaseGenerator:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")

        self.system_prompt = """
        You are an expert software testing engineer.

        Your task is to generate comprehensive and high-quality test cases
        for the given programming problem.

        Strict Rules:
        1. Return ONLY valid JSON.
        2. Follow the exact schema provided.
        3. Do NOT include markdown formatting.
        4. Do NOT include explanations outside the JSON.
        5. Do NOT generate code implementations.
        6. Each test case must contain clear input and expected_output fields.
        7. Cover normal cases, edge cases, and boundary cases.
        8. Ensure test cases are deterministic and executable.
        9. Do NOT include unnecessary text.
        10. If assumptions are required, clearly state them inside the "reasoning" field.

        The response must strictly follow the structured output schema.
        """

    def generate_tests(self, question):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        structured_model = model.with_structured_output(TestOutput)

        response = structured_model.invoke(messages)

        return response
