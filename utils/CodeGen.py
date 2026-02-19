import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Optional
load_dotenv()

class Output(BaseModel):
    code: Optional[str]
    reasoning: Optional[str]

class CodeGenerator:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.system_prompt = """
            You are an expert competitive programmer and software engineer.
            Your task is to generate the most optimal, correct, and production-ready code
            for the given problem.
            Strict Rules:
            1. Return ONLY valid JSON.
            2. Follow the exact schema provided.
            3. Do NOT include markdown formatting.
            4. Do NOT include explanations outside the JSON.
            5. Do NOT include example usage.
            6. Do NOT include test cases.
            7. Do NOT include comments unless absolutely necessary.
            8. Code must be syntactically correct and directly executable.
            9. Prefer time and space optimal solutions.
            10. If assumptions are required, clearly state them inside the "reasoning" field.

            The response must strictly follow the structured output schema.
            """


    def generate_code(self, question):
        prompt = question
        messages = [
            SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)
        ]
        model = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            temperature = 0
        )
        structured_model = model.with_structured_output(Output)
        response = structured_model.invoke(messages)
        return response