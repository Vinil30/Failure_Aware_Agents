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
    final_status: Optional[str]
class ANNRiskEstimator:
    def __init__(self, 
                 model_folder: str = "/kaggle/working/Failure_Aware_Agents/utils/saved_model/",
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
        
        # Load threshold from training
        if os.path.exists(f"{model_folder}/best_threshold.txt"):
            with open(f"{model_folder}/best_threshold.txt", 'r') as f:
                self.threshold = float(f.read().strip())
        else:
            self.threshold = 0.5
        
        # Load the model architecture
        checkpoint = torch.load(f"{model_folder}/failure_risk_model.pth", map_location=self.device)
        input_dim = checkpoint['input_dim']
        
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
        
        # EXACT MATCH to dataset generation
        self.top_k_history = 2  # ← Changed from 3 to 2
        self.prior_smoothing = 0.1  # ← EXACT match
        
        print(f"✅ ANN Risk Estimator loaded!")
        print(f"   - Threshold: {self.threshold:.3f}")
        print(f"   - Features: {len(self.feature_names)}")
        print(f"   - Device: {self.device}")
    
    def compute_features(self, question: str, code: str, confidence: float = None, logprob: float = None) -> Dict:
        """
        EXACT feature extraction matching dataset generation script.
        
        Args:
            question: The problem prompt
            code: Generated code
            confidence: Optional pre-computed confidence (0-1)
            logprob: Optional logprob to compute confidence from
        """
        
        # AST nodes - EXACT match to dataset generation
        ast_nodes = 0
        try:
            tree = ast.parse(code)
            ast_nodes = sum(1 for _ in ast.walk(tree))
        except Exception:
            ast_nodes = 0  # ← EXACT match: set to 0 on failure
        
        # Complexity - EXACT match to dataset generation
        avg_complexity = 0.0
        try:
            comp = cc_visit(code)
            avg_complexity = sum(c.complexity for c in comp) / len(comp) if comp else 0.0
        except Exception:
            avg_complexity = 0.0  # ← EXACT match: set to 0 on failure
        
        # Confidence - EXACT match to dataset generation
        if confidence is None:
            if logprob is not None:
                confidence = float(np.clip(np.exp(logprob), 0.0, 1.0))
            else:
                # Default from dataset median (since training data had this range)
                confidence = 0.89  # ← Dataset median, not hardcoded high
        
        # Prior history - EXACT match to compute_prior() below
        emb = self.embedding_model.encode(question[:500])
        prior = self.compute_prior(emb)
        
        features = {
            "prompt_len": len(question),
            "code_len": len(code),
            "ast_nodes": ast_nodes,
            "avg_complexity": avg_complexity,
            "confidence": confidence,
            "prior_history": prior
        }
        
        # Engineered features (same as training)
        features["complexity_per_len"] = features["avg_complexity"] / (features["code_len"] + 1)
        features["ast_per_len"] = features["ast_nodes"] / (features["code_len"] + 1)
        features["log_code_len"] = np.log1p(features["code_len"])
        
        return features
    
    def compute_prior(self, emb: np.ndarray) -> float:
        """
        EXACT match to dataset generation's compute_prior()
        Uses TOP_K_HISTORY = 2, PRIOR_SMOOTHING = 0.1
        """
        if len(self.history_labels) < self.top_k_history:
            return 0.5  # ← EXACT match to dataset generation
        
        q = np.array([emb]).astype("float32")
        d, i = self.faiss_index.search(q, self.top_k_history)
        sims = 1.0 / (1.0 + d[0])  # ← EXACT formula
        labs = np.array(self.history_labels)[i[0]]
        raw = float(np.sum(sims * labs) / np.sum(sims))
        
        # EXACT smoothing formula from dataset generation
        return raw * (1 - 2 * self.prior_smoothing) + self.prior_smoothing
    
    def predict_risk(self, question: str, code: str, confidence: float = None, logprob: float = None):
        """Predict risk using EXACT feature extraction matching training data"""
        
        features = self.compute_features(question, code, confidence, logprob)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Align columns EXACTLY like training
        df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Replace NaN/Inf with 0 (dataset generation didn't have these issues)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        # Scale
        X = self.scaler.transform(df)
        
        # Predict
        with torch.no_grad():
            tensor_input = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_input)
            prob = torch.sigmoid(logits).cpu().numpy()[0][0]
        
        # NO CLAMPING - let natural output through
        return prob, features
    
    def update_history(self, question: str, code: str, failed: bool):
        """Update historical data with new sample - EXACT match to dataset generation"""
        emb = self.embedding_model.encode(question[:500])
        
        # Update FAISS index
        self.faiss_index.add(np.array([emb]).astype("float32"))
        self.history_labels.append(1 if failed else 0)
        
        # Save updates
        faiss.write_index(self.faiss_index, self.faiss_file)
        np.save(self.label_file, np.array(self.history_labels))
        
        # Append to CSV (matching dataset format)
        features = self.compute_features(question, code)
        features["passed_tests"] = 0 if failed else 1
        features["test_error"] = "" if not failed else "user_feedback"
        features["failed"] = 1 if failed else 0
        
        df_new = pd.DataFrame([features])
        if os.path.exists(self.csv_file):
            df_new.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.csv_file, index=False)

# Initialize components
code_generator = CodeGenerator()
failure_analyzer = FailureAnalyzer()
risk_estimator = ANNRiskEstimator(model_folder="/kaggle/working/Failure_Aware_Agents/utils/saved_model/")

# Configuration
RISK_THRESHOLD = 0.501  # If risk > 0.6, trigger failure analysis
MAX_REGENERATIONS = 3

# ── Node definitions ──────────────────────────────────────────────────────────

def codegen_node(state: AgentState):
    """Generate code with reasoning trace"""
    response = code_generator.generate_code(state["question"])
    reasoning = getattr(response, 'reasoning', '')
    
    return {
        "code": response.code,
        "reasoning": reasoning,
        "regeneration_count": state.get("regeneration_count", 0)+1,
        "code_history": state.get("code_history", []) + [response.code]
    }

def risk_estimation_node(state: AgentState):

    try:
        confidence = code_generator.get_last_raw_confidence()
        logprob = code_generator.get_last_logprob()

    except:
        confidence = 0.89
        logprob = -100.0

    risk_score, features = risk_estimator.predict_risk(
        state["question"],
        state["code"],
        confidence=confidence,
        logprob=logprob
    )

    # IMPORTANT FIX
    final_status = (
        "FAILED"
        if risk_score > RISK_THRESHOLD
        else "SUCCESS"
    )

    return {
        "risk_score": risk_score,
        "features": features,
        "final_status": final_status
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

# ── Routing functions ─────────────────────────────────────────────────────────

def should_execute(state: AgentState):

    if state.get("final_status") == "FAILED":
        return "high_risk"

    risk_estimator.update_history(
        state["question"],
        state["code"],
        failed=False
    )

    return "end"

def should_regenerate(state: AgentState):

    if state.get("regeneration_count", 0) < MAX_REGENERATIONS:
        return "regenerate"

    risk_estimator.update_history(
        state["question"],
        state["code"],
        failed=True
    )

    return "end"

# ── Graph construction ────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

builder.add_node("codegen", codegen_node)
builder.add_node("risk_estimation", risk_estimation_node)
builder.add_node("failure_analysis", failure_analysis_node)
builder.add_node("regenerate", regeneration_node)

# codegen → risk_estimation (always)
builder.add_edge(START, "codegen")
builder.add_edge("codegen", "risk_estimation")

# risk_estimation → failure_analysis  OR  end
builder.add_conditional_edges(
    "risk_estimation",
    should_execute,
    {"high_risk": "failure_analysis", "end": END}
)

# failure_analysis → regenerate  OR  end
builder.add_conditional_edges(
    "failure_analysis",
    should_regenerate,
    {"regenerate": "regenerate", "end": END}
)

# regenerate loops back to risk_estimation (re-score new code)
builder.add_edge("regenerate", "risk_estimation")

graph = builder.compile()

# ── Utilities ─────────────────────────────────────────────────────────────────

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
        "features": None,
        "final_status": None
    }
    
    result = graph.invoke(initial_state)
    
    print(f"\n{'='*50}")
    print(f"Pipeline completed for: {question[:100]}...")
    print(f"Regeneration attempts: {result['regeneration_count']}")
    
    if result.get('failure_reason'):
        print(f"Failure reason: {result['failure_reason']}")
    
    if result.get('risk_score'):
        print(f"Risk score: {result['risk_score']:.3f}")
    
    return result

if __name__ == "__main__":
    sample_question = "Write a function that returns the sum of two numbers"
    result = run_pipeline(sample_question)
