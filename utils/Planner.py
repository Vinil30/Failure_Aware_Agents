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
import torch
import torch.nn as nn
import joblib

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
    reasoning: Optional[str]
    features: Optional[Dict]

class ANNRiskEstimator:
    """ANN-based risk estimation model that predicts failure probability"""
    
    def __init__(self, 
                 model_folder: str = "saved_model",
                 csv_file: str = "failure_risk_dataset.csv",
                 faiss_file: str = "history_index.faiss",
                 label_file: str = "history_labels.npy"):
        
        self.csv_file = csv_file
        self.faiss_file = faiss_file
        self.label_file = label_file
        self.model_folder = model_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384
        
        # Load FAISS index for history
        if os.path.exists(faiss_file):
            self.faiss_index = faiss.read_index(faiss_file)
            self.history_labels = list(np.load(label_file))
        else:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.history_labels = []
        
        # Load trained ANN model artifacts
        self.scaler = joblib.load(f"{model_folder}/scaler.pkl")
        self.feature_names = joblib.load(f"{model_folder}/feature_names.pkl")
        
        # Load threshold
        if os.path.exists(f"{model_folder}/best_threshold.txt"):
            with open(f"{model_folder}/best_threshold.txt", 'r') as f:
                self.threshold = float(f.read().strip())
        else:
            self.threshold = 0.5
        
        # Load the model architecture
        checkpoint = torch.load(f"{model_folder}/failure_risk_model.pth", map_location=self.device)
        input_dim = checkpoint['input_dim']
        
        # Define the same architecture as training
        class ANN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.4),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.4),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.model = ANN(input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Hyperparameters
        self.top_k_history = 3
        self.prior_smoothing = 0.1
        
        print(f"✅ ANN Risk Estimator loaded!")
        print(f"   - Threshold: {self.threshold:.3f}")
        print(f"   - Features: {len(self.feature_names)}")
        print(f"   - Device: {self.device}")
    
    def compute_features(self, question: str, code: str, confidence: float = 0.92) -> Dict:
        """Extract features robustly (aligned with training distribution)"""

        # AST nodes
        try:
            tree = ast.parse(code)
            ast_nodes = sum(1 for _ in ast.walk(tree))
        except:
            ast_nodes = max(10, len(code) // 5)   # safe fallback

        # Complexity
        try:
            comp = cc_visit(code)
            complexity = sum(c.complexity for c in comp) / len(comp) if comp else 2.5
        except:
            complexity = 2.5  # realistic fallback

        # Prior history
        emb = self.embedding_model.encode(question[:500])
        prior = self.compute_prior(emb)

        # Clamp values to training-like ranges (VERY IMPORTANT)
        confidence = float(np.clip(confidence, 0.85, 0.97))
        prior = float(np.clip(prior, 0.3, 0.8))
        complexity = float(np.clip(complexity, 1.0, 5.0))

        features = {
            "prompt_len": len(question),
            "code_len": len(code),
            "ast_nodes": ast_nodes,
            "avg_complexity": complexity,
            "confidence": confidence,
            "prior_history": prior
        }

        # engineered features (same as training)
        features["complexity_per_len"] = features["avg_complexity"] / (features["code_len"] + 1)
        features["ast_per_len"] = features["ast_nodes"] / (features["code_len"] + 1)
        features["log_code_len"] = np.log1p(features["code_len"])

        return features
        
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
    
    def predict_risk(self, question: str, code: str, confidence: float = 0.92):
        """Stable and correct ANN inference"""

        features = self.compute_features(question, code, confidence)

        # Convert to DataFrame (same as training pipeline)
        df = pd.DataFrame([features])

        # Align columns EXACTLY like training
        df = df.reindex(columns=self.feature_names, fill_value=0)

        # Replace any remaining bad values with median-like fallback
        df = df.fillna(df.median())

        # Scale
        X = self.scaler.transform(df)

        # Predict
        with torch.no_grad():
            tensor_input = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_input)
            prob = torch.sigmoid(logits).cpu().numpy()[0][0]

        # Clamp extreme outputs (important for stability)
        prob = float(np.clip(prob, 0.01, 0.99))

        return prob, features
    
    def update_history(self, question: str, code: str, failed: bool):
        """Update historical data with new sample"""
        emb = self.embedding_model.encode(question[:500])
        
        # Update FAISS index
        self.faiss_index.add(np.array([emb]).astype("float32"))
        self.history_labels.append(1 if failed else 0)
        
        # Save updates
        faiss.write_index(self.faiss_index, self.faiss_file)
        np.save(self.label_file, np.array(self.history_labels))
        
        # Optionally save to CSV for future retraining
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
risk_estimator = ANNRiskEstimator(model_folder="saved_model")  # ← CHANGE THIS

# Configuration
RISK_THRESHOLD = 0.6  # If risk > 0.6, trigger failure analysis
MAX_REGENERATIONS = 3

# ... (rest of your code remains exactly the same from here)

def codegen_node(state: AgentState):
    """Generate code with reasoning trace"""
    response = code_generator.generate_code(state["question"])
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
    confidence = 0.5
    
    risk_score, features = risk_estimator.predict_risk(
        state["question"],
        state["code"],
        confidence
    )
    
    return {
        "risk_score": risk_score,
        "features": features
    }

def execute_node(state: AgentState):
    """Execute code with test cases"""
    result = executor.execute(state["code"], state["test_cases"])
    
    if result["status"] != "success":
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
import torch
import numpy as np

def compute_confidence(outputs):
    """
    Compute confidence from token probabilities
    """

    scores = outputs.scores  # logits per step
    if scores is None:
        return 0.9  # fallback

    probs = []

    for step_logits in scores:
        step_probs = torch.softmax(step_logits, dim=-1)
        max_prob = torch.max(step_probs).item()  # confidence of chosen token
        probs.append(max_prob)

    # Average confidence across tokens
    confidence = float(np.mean(probs))

    return confidence
def regeneration_node(state: AgentState):
    """Regenerate code based on failure analysis"""
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
    return "risk_estimation"

def should_execute(state: AgentState):
    risk_score = state.get("risk_score", 0)
    
    if risk_score > RISK_THRESHOLD:
        return "high_risk"
    return "execute"

def should_handle_failure(state: AgentState):
    failed_cases = state.get("failed_cases", 0)
    error_type = state.get("error_type", "")
    regen_count = state.get("regeneration_count", 0)
    
    if failed_cases > 0 or error_type == "execution_error":
        if regen_count < MAX_REGENERATIONS:
            return "regenerate"
        else:
            return "analyze_failure"
    
    risk_estimator.update_history(state["question"], state["code"], failed=False)
    return "end"

def should_regenerate(state: AgentState):
    if state.get("regeneration_count", 0) < MAX_REGENERATIONS:
        return "regenerate"
    return "end"

# Build the graph
builder = StateGraph(AgentState)

builder.add_node("codegen", codegen_node)
builder.add_node("testgen", testgen_node)
builder.add_node("risk_estimation", risk_estimation_node)
builder.add_node("execute", execute_node)
builder.add_node("failure_analysis", failure_analysis_node)
builder.add_node("regenerate", regeneration_node)

builder.add_edge(START, "codegen")
builder.add_edge("codegen", "testgen")

builder.add_conditional_edges(
    "testgen",
    should_check_risk,
    {"risk_estimation": "risk_estimation"}
)

builder.add_conditional_edges(
    "risk_estimation",
    should_execute,
    {"high_risk": "failure_analysis", "execute": "execute"}
)

builder.add_conditional_edges(
    "execute",
    should_handle_failure,
    {"regenerate": "regenerate", "analyze_failure": "failure_analysis", "end": END}
)

builder.add_conditional_edges(
    "failure_analysis",
    should_regenerate,
    {"regenerate": "regenerate", "end": END}
)

builder.add_edge("regenerate", "testgen")

graph = builder.compile()

def visualize_graph():
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except:
        print("Graph visualization not available")

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
        "reasoning": None,
        "features": None
    }
    
    result = graph.invoke(initial_state)
    
    print(f"\n{'='*50}")
    print(f"Pipeline completed for: {question[:100]}...")
    print(f"Final status: {'Success' if result['failed_cases'] == 0 else 'Failed'}")
    print(f"Passed tests: {result['passed_cases']}")
    print(f"Failed tests: {result['failed_cases']}")
    print(f"Regeneration attempts: {result['regeneration_count']}")
    
    if result.get('failure_reason'):
        print(f"Failure reason: {result['failure_reason']}")
    
    if result.get('risk_score'):
        print(f"Risk score: {result['risk_score']:.3f}")
    
    return result

if __name__ == "__main__":
    sample_question = "Write a function that returns the sum of two numbers"
    result = run_pipeline(sample_question)