

from typing import TypedDict, Optional, List, Dict
from langgraph.graph import StateGraph, START, END
from utils.CodeGen import CodeGenerator
from utils.TestCaseGen import TestCaseGenerator
from utils.CodeExec import CodeExecutor
from utils.FailureAnalyzer import FailureAnalyzer

class AgentState(TypedDict):
    question: str
    code: Optional[str]
    test_cases: Optional[List[Dict]]
    execution_result: Optional[Dict]
    failed_cases: Optional[int]
    passed_cases: Optional[int]
    failure_reason: Optional[str]
    severity: Optional[str]

code_generator = CodeGenerator()
test_generator = TestCaseGenerator()
executor = CodeExecutor()
failure_analyzer = FailureAnalyzer()

def codegen_node(state: AgentState):
    response = code_generator.generate_code(state["question"])

    return {
        "code": response.code
    }


def testgen_node(state: AgentState):
    response = test_generator.generate_tests(state["question"])
    return {
        "test_cases": response.test_cases
    }


def execute_node(state: AgentState):
    result = executor.execute(state["code"], state["test_cases"])

    if result["status"] != "success":
        return {
            "execution_result": result,
            "failed_cases": len(state["test_cases"]) if state["test_cases"] else 0,
            "passed_cases": 0
        }
    results = result["results"]

    passed = sum(1 for r in results if r.get("passed"))
    failed = sum(1 for r in results if not r.get("passed"))

    return {
        "execution_result": result,
        "failed_cases": failed,
        "passed_cases": passed
    }


def failure_node(state: AgentState):
    failure = failure_analyzer.analyze(
        state["question"],
        state["code"],
        state["test_cases"],
        state["execution_result"]
    )

    return {
        "failure_reason": failure.failure_reason,
        "severity": failure.severity
    }

def should_analyze_failure(state: AgentState):
    if state.get("failed_cases", 0) > 0:
        return "failure"
    return "end"

builder = StateGraph(AgentState)

builder.add_node("codegen", codegen_node)
builder.add_node("testgen", testgen_node)
builder.add_node("execute", execute_node)
builder.add_node("failure", failure_node)

builder.add_edge(START, "codegen")
builder.add_edge("codegen", "testgen")
builder.add_edge("testgen", "execute")

builder.add_conditional_edges(
    "execute",
    should_analyze_failure,
    {
        "failure": "failure",
        "end": END
    }
)

builder.add_edge("failure", END)
graph = builder.compile()
