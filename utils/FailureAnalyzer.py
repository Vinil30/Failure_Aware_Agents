# utils/FailureAnalyzer.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class FailureOutput(BaseModel):
    failure_reason: Optional[str]
    severity: Optional[str]


class FailureAnalyzer:

    def __init__(self):
        self.system_prompt = """
        You are an expert debugging and program analysis system.

        Analyze why the generated code failed.

        You will receive:
        - Original problem
        - Generated code
        - Test cases
        - Execution results

        Strict Rules:
        1. Return ONLY valid JSON.
        2. Do NOT include markdown.
        3. Provide concise root cause reasoning.
        4. Classify severity as: LOW, MEDIUM, HIGH.
        """

    def analyze(self, question, code, test_cases, execution_result):

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

        structured_model = model.with_structured_output(FailureOutput)

        payload = f"""
        Problem:
        {question}

        Code:
        {code}

        Test Cases:
        {test_cases}

        Execution Result:
        {execution_result}
        """

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=payload)
        ]

        return structured_model.invoke(messages)
