from typing import TypedDict, Optional, List, Dict, Tuple
from langgraph.graph import StateGraph, START, END
from utils.CodeGen import CodeGenerator
from utils.TestCaseGen import TestCaseGenerator
from utils.CodeExec import CodeExecutor
from utils.FailureAnalyzer import FailureAnalyzer
import pandas as pd
import numpy as np
import ast
from radon.complexity import cc_visit
import faiss
from sentence_transformers import SentenceTransformer
import os

class AgentState(TypedDict):
    question: str
    code: Optional[str]
    test_cases: Optional[List[Dict]]
    execution_result: Optional[Dict]
    failed_cases: Optional[int]
    passed_cases: Optional[int]
    failure_reason: Optional[str]
    severity: Optional[str]
    risk_score: Optional[float]
    regeneration_count: Optional[int]
    code_history: Optional[List[str]]
    reasoning: Optional[str]  # Added reasoning trace

class RiskEstimator:
    """Risk estimation model that predicts failure probability"""
    
    def __init__(self, model_path: str = "failure_risk_model.pkl", 
                 csv_file: str = "failure_risk_dataset.csv",
                 faiss_file: str = "history_index.faiss",
                 label_file: str = "history_labels.npy"):
        
        self.csv_file = csv_file
        self.faiss_file = faiss_file
        self.label_file = label_file
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384
        
        # Load or initialize FAISS index for history
        if os.path.exists(faiss_file):
            self.faiss_index = faiss.read_index(faiss_file)
            self.history_labels = list(np.load(label_file))
        else:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.history_labels = []
            
        # Load trained model (if exists)
        self.model = None
        if os.path.exists(model_path):
            import joblib
            self.model = joblib.load(model_path)
            
        # Hyperparameters
        self.top_k_history = 3
        self.prior_smoothing = 0.1
        
    def compute_features(self, question: str, code: str, confidence: float = 0.5) -> Dict:
        """Extract features from code and question"""
        # AST nodes count
        ast_nodes = 0
        try:
            tree = ast.parse(code)
            ast_nodes = sum(1 for _ in ast.walk(tree))
        except:
            pass
            
        # Complexity analysis
        complexity = 0
        try:
            comp = cc_visit(code)
            complexity = sum(c.complexity for c in comp) / len(comp) if comp else 0
        except:
            pass
            
        # Prior history from similar prompts
        emb = self.embedding_model.encode(question[:500])
        prior = self.compute_prior(emb)
        
        return {
            "prompt_len": len(question),
            "code_len": len(code),
            "ast_nodes": ast_nodes,
            "avg_complexity": complexity,
            "confidence": confidence,
            "prior_history": prior
        }
    
    def compute_prior(self, emb: np.ndarray) -> float:
        """Compute prior probability from historical embeddings"""
        if len(self.history_labels) < self.top_k_history:
            return 0.5
            
        q = np.array([emb]).astype("float32")
        d, i = self.faiss_index.search(q, self.top_k_history)
        
        sims = 1 / (1 + d[0])
        labs = np.array(self.history_labels)[i[0]]
        
        raw = np.sum(sims * labs) / np.sum(sims)
        return raw * (1 - 2 * self.prior_smoothing) + self.prior_smoothing
    
    def predict_risk(self, question: str, code: str, confidence: float = 0.5) -> Tuple[float, Dict]:
        """Predict risk score (0-1) where higher = more likely to fail"""
        features = self.compute_features(question, code, confidence)
        
        # If we have a trained model, use it
        if self.model is not None:
            feature_vector = np.array([
                features["prompt_len"],
                features["code_len"],
                features["ast_nodes"],
                features["avg_complexity"],
                features["confidence"],
                features["prior_history"]
            ]).reshape(1, -1)
            
            risk_score = self.model.predict_proba(feature_vector)[0][1]  # Probability of failure
        else:
            # Fallback: heuristic-based risk estimation
            risk_score = self._heuristic_risk(features)
            
        return risk_score, features
    
    def _heuristic_risk(self, features: Dict) -> float:
        """Fallback heuristic when model isn't trained"""
        risk = 0.0
        
        # Code length risk (longer code = higher risk)
        if features["code_len"] > 500:
            risk += 0.3
        elif features["code_len"] > 200:
            risk += 0.15
            
        # Complexity risk
        if features["avg_complexity"] > 10:
            risk += 0.3
        elif features["avg_complexity"] > 5:
            risk += 0.15
            
        # Low confidence
        if features["confidence"] < 0.3:
            risk += 0.2
            
        # Prior history risk
        if features["prior_history"] > 0.7:
            risk += 0.2
        elif features["prior_history"] > 0.4:
            risk += 0.1
            
        return min(risk, 1.0)
    
    def update_history(self, question: str, code: str, failed: bool):
        """Update historical data with new sample"""
        emb = self.embedding_model.encode(question[:500])
        
        # Update FAISS index
        self.faiss_index.add(np.array([emb]).astype("float32"))
        self.history_labels.append(1 if failed else 0)
        
        # Save updates
        faiss.write_index(self.faiss_index, self.faiss_file)
        np.save(self.label_file, np.array(self.history_labels))
        
        # Optionally save to CSV for future training
        features = self.compute_features(question, code)
        features["failed"] = 1 if failed else 0
        
        df_new = pd.DataFrame([features])
        if os.path.exists(self.csv_file):
            df_new.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.csv_file, index=False)

# Initialize components
code_generator = CodeGenerator()
test_generator = TestCaseGenerator()
executor = CodeExecutor()
failure_analyzer = FailureAnalyzer()
risk_estimator = RiskEstimator()

# Configuration
RISK_THRESHOLD = 0.6  # If risk > 0.6, trigger failure analysis
MAX_REGENERATIONS = 3  # Maximum number of regeneration attempts

def codegen_node(state: AgentState):
    """Generate code with reasoning trace"""
    response = code_generator.generate_code(state["question"])
    
    # Extract reasoning if available (assuming response has reasoning attribute)
    reasoning = getattr(response, 'reasoning', '')
    
    return {
        "code": response.code,
        "reasoning": reasoning,
        "regeneration_count": state.get("regeneration_count", 0) + 1,
        "code_history": state.get("code_history", []) + [response.code]
    }

def testgen_node(state: AgentState):
    """Generate test cases"""
    response = test_generator.generate_tests(state["question"])
    return {"test_cases": response.test_cases}

def risk_estimation_node(state: AgentState):
    """Estimate risk of code failure"""
    # Get confidence from generation (can be extracted from reasoning or defaults)
    confidence = 0.5  # You can extract this from the generation model
    
    risk_score, features = risk_estimator.predict_risk(
        state["question"],
        state["code"],
        confidence
    )
    
    return {
        "risk_score": risk_score,
        "features": features  # Store for debugging/logging
    }

def execute_node(state: AgentState):
    """Execute code with test cases"""
    result = executor.execute(state["code"], state["test_cases"])
    
    if result["status"] != "success":
        # Execution error (e.g., syntax error, runtime error)
        return {
            "execution_result": result,
            "failed_cases": len(state["test_cases"]) if state["test_cases"] else 0,
            "passed_cases": 0,
            "error_type": "execution_error"
        }
    
    results = result["results"]
    passed = sum(1 for r in results if r.get("passed"))
    failed = sum(1 for r in results if not r.get("passed"))
    
    return {
        "execution_result": result,
        "failed_cases": failed,
        "passed_cases": passed,
        "error_type": "test_failure" if failed > 0 else "success"
    }

def failure_analysis_node(state: AgentState):
    """Analyze failure and provide reasoning"""
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

def regeneration_node(state: AgentState):
    """Regenerate code based on failure analysis"""
    # Build enhanced prompt with failure context
    enhanced_prompt = f"""
Original problem: {state["question"]}

Previous attempts: {len(state.get("code_history", []))}

Previous code that failed:
{state["code"]}

Failure reason: {state.get("failure_reason", "Unknown")}

Please provide an improved solution that addresses the issues above.
"""
    
    response = code_generator.generate_code(enhanced_prompt)
    
    return {
        "code": response.code,
        "reasoning": getattr(response, 'reasoning', ''),
        "regeneration_count": state.get("regeneration_count", 0) + 1,
        "code_history": state.get("code_history", []) + [response.code]
    }

def should_check_risk(state: AgentState):
    """Check if we should analyze risk before execution"""
    # Always check risk if we have code and tests
    return "risk_estimation"

def should_execute(state: AgentState):
    """Check if we should execute based on risk score"""
    risk_score = state.get("risk_score", 0)
    
    # If risk is too high, skip execution and go to failure analysis
    if risk_score > RISK_THRESHOLD:
        return "high_risk"
    return "execute"

def should_handle_failure(state: AgentState):
    """Determine next steps after execution"""
    failed_cases = state.get("failed_cases", 0)
    error_type = state.get("error_type", "")
    regen_count = state.get("regeneration_count", 0)
    
    # Check if we have failures
    if failed_cases > 0 or error_type == "execution_error":
        # Check if we can regenerate
        if regen_count < MAX_REGENERATIONS:
            return "regenerate"
        else:
            return "analyze_failure"
    
    # Success case - update history
    risk_estimator.update_history(state["question"], state["code"], failed=False)
    return "end"

def should_regenerate(state: AgentState):
    """Check if we should regenerate after analysis"""
    # After failure analysis, regenerate if we haven't exceeded limit
    if state.get("regeneration_count", 0) < MAX_REGENERATIONS:
        return "regenerate"
    return "end"

# Build the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("codegen", codegen_node)
builder.add_node("testgen", testgen_node)
builder.add_node("risk_estimation", risk_estimation_node)
builder.add_node("execute", execute_node)
builder.add_node("failure_analysis", failure_analysis_node)
builder.add_node("regenerate", regeneration_node)

# Define edges
builder.add_edge(START, "codegen")
builder.add_edge("codegen", "testgen")

# Conditional after test generation
builder.add_conditional_edges(
    "testgen",
    should_check_risk,
    {
        "risk_estimation": "risk_estimation"
    }
)

# Conditional after risk estimation
builder.add_conditional_edges(
    "risk_estimation",
    should_execute,
    {
        "high_risk": "failure_analysis",
        "execute": "execute"
    }
)

# Conditional after execution
builder.add_conditional_edges(
    "execute",
    should_handle_failure,
    {
        "regenerate": "regenerate",
        "analyze_failure": "failure_analysis",
        "end": END
    }
)

# Conditional after failure analysis
builder.add_conditional_edges(
    "failure_analysis",
    should_regenerate,
    {
        "regenerate": "regenerate",
        "end": END
    }
)

# After regeneration, go back to test generation
builder.add_edge("regenerate", "testgen")

# Compile the graph
graph = builder.compile()

# Optional: Add visualization
def visualize_graph():
    """Generate graph visualization (requires graphviz)"""
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")

# Example usage
def run_pipeline(question: str):
    """Run the complete pipeline"""
    initial_state = {
        "question": question,
        "code": None,
        "test_cases": None,
        "execution_result": None,
        "failed_cases": 0,
        "passed_cases": 0,
        "failure_reason": None,
        "severity": None,
        "risk_score": None,
        "regeneration_count": 0,
        "code_history": [],
        "reasoning": None
    }
    
    result = graph.invoke(initial_state)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Pipeline completed for: {question[:100]}...")
    print(f"Final status: {'Success' if result['failed_cases'] == 0 else 'Failed'}")
    print(f"Passed tests: {result['passed_cases']}")
    print(f"Failed tests: {result['failed_cases']}")
    print(f"Regeneration attempts: {result['regeneration_count']}")
    
    if result.get('failure_reason'):
        print(f"Failure reason: {result['failure_reason']}")
    
    return result

if __name__ == "__main__":
    # Test with a sample question
    sample_question = "Write a function that returns the sum of two numbers"
    result = run_pipeline(sample_question)