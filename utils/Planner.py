from typing import TypedDict, Optional, List, Dict
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
import sys
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
        checkpoint = torch.load(
            f"{model_folder}/failure_risk_model.pth",
            map_location=self.device
        )
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
        self.top_k_history = 2
        self.prior_smoothing = 0.1

        # ── FIX: derive cold-start prior from actual training data ────────────
        # Training data has prior_history values of 0.1 and 0.9 (extremes) plus
        # interpolated values — never 0.5.  Using 0.5 at inference puts us
        # out-of-distribution and suppresses risk scores.
        # We use the dataset's mean failure rate as the cold-start prior instead.
        if os.path.exists(csv_file):
            try:
                df_hist = pd.read_csv(csv_file)
                if "failed" in df_hist.columns and len(df_hist) > 0:
                    raw_rate = float(df_hist["failed"].mean())
                    # Apply the same smoothing formula used in compute_prior so
                    # the cold-start value lives in the same range as warm values
                    self.cold_start_prior = (
                        raw_rate * (1 - 2 * self.prior_smoothing) + self.prior_smoothing
                    )
                else:
                    self.cold_start_prior = 0.9  # conservative: assume failure
                if "confidence" in df_hist.columns and len(df_hist) > 0:
                    self.dataset_confidence_mean = float(df_hist["confidence"].mean())
                else:
                    self.dataset_confidence_mean = 0.88
            except Exception:
                self.cold_start_prior = 0.9
                self.dataset_confidence_mean = 0.88
        else:
            # No CSV available — be conservative
            self.cold_start_prior = 0.9
            self.dataset_confidence_mean = 0.88

        print(f"✅ ANN Risk Estimator loaded!")
        print(f"   - Threshold          : {self.threshold:.3f}")
        print(f"   - Features           : {len(self.feature_names)}")
        print(f"   - Device             : {self.device}")
        print(f"   - Cold-start prior   : {self.cold_start_prior:.3f}")
        print(f"   - Confidence default : {self.dataset_confidence_mean:.3f}")

    def compute_features(
        self,
        question: str,
        code: str,
        confidence: float = None,
        logprob: float = None
    ) -> Dict:
        """
        EXACT feature extraction matching dataset generation script.
        """
        # AST nodes
        ast_nodes = 0
        try:
            tree = ast.parse(code)
            ast_nodes = sum(1 for _ in ast.walk(tree))
        except Exception:
            ast_nodes = 0

        # Cyclomatic complexity
        avg_complexity = 0.0
        try:
            comp = cc_visit(code)
            avg_complexity = sum(c.complexity for c in comp) / len(comp) if comp else 0.0
        except Exception:
            avg_complexity = 0.0

        # Confidence
        if confidence is None:
            if logprob is not None:
                confidence = float(np.clip(np.exp(logprob), 0.0, 1.0))
            else:
                # Use dataset mean — not hardcoded 0.89 which was causing
                # the model to always predict low risk
                confidence = self.dataset_confidence_mean

        # Prior history
        emb = self.embedding_model.encode(question[:500])
        prior = self.compute_prior(emb)

        features = {
            "prompt_len"      : len(question),
            "code_len"        : len(code),
            "ast_nodes"       : ast_nodes,
            "avg_complexity"  : avg_complexity,
            "confidence"      : confidence,
            "prior_history"   : prior,
        }

        # Engineered features (same as training)
        features["complexity_per_len"] = features["avg_complexity"] / (features["code_len"] + 1)
        features["ast_per_len"]        = features["ast_nodes"]       / (features["code_len"] + 1)
        features["log_code_len"]       = np.log1p(features["code_len"])

        return features

    def compute_prior(self, emb: np.ndarray) -> float:
        """
        EXACT match to dataset generation's compute_prior().
        Uses TOP_K_HISTORY=2, PRIOR_SMOOTHING=0.1.

        FIX: cold-start now returns self.cold_start_prior (derived from dataset
        failure rate) instead of 0.5, which was never seen during training.
        """
        if len(self.history_labels) < self.top_k_history:
            return self.cold_start_prior   # ← was hardcoded 0.5

        q = np.array([emb]).astype("float32")
        d, i = self.faiss_index.search(q, self.top_k_history)
        sims = 1.0 / (1.0 + d[0])
        labs = np.array(self.history_labels)[i[0]]
        raw  = float(np.sum(sims * labs) / np.sum(sims))

        return raw * (1 - 2 * self.prior_smoothing) + self.prior_smoothing

    def predict_risk(
        self,
        question: str,
        code: str,
        confidence: float = None,
        logprob: float = None
    ):
        """Predict risk using EXACT feature extraction matching training data."""

        features = self.compute_features(question, code, confidence, logprob)

        df = pd.DataFrame([features])
        df = df.reindex(columns=self.feature_names, fill_value=0)
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        X = self.scaler.transform(df)

        with torch.no_grad():
            tensor_input = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_input)
            prob   = torch.sigmoid(logits).cpu().numpy()[0][0]

        return float(prob), features

    def update_history(self, question: str, code: str, failed: bool):
        """Update historical data with new sample — EXACT match to dataset generation."""
        emb = self.embedding_model.encode(question[:500])

        self.faiss_index.add(np.array([emb]).astype("float32"))
        self.history_labels.append(1 if failed else 0)

        faiss.write_index(self.faiss_index, self.faiss_file)
        np.save(self.label_file, np.array(self.history_labels))

        features = self.compute_features(question, code)
        features["passed_tests"] = 0 if failed else 1
        features["test_error"]   = "" if not failed else "user_feedback"
        features["failed"]       = 1 if failed else 0

        df_new = pd.DataFrame([features])
        if os.path.exists(self.csv_file):
            df_new.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.csv_file, index=False)


# ── Initialize components ─────────────────────────────────────────────────────

code_generator = CodeGenerator()
failure_analyzer = FailureAnalyzer()
risk_estimator = ANNRiskEstimator(
    model_folder="/kaggle/working/Failure_Aware_Agents/utils/saved_model/"
)

# Configuration — single source of truth, imported by runner.py
RISK_THRESHOLD   = 0.501
MAX_REGENERATIONS = 3


# ── Node definitions ──────────────────────────────────────────────────────────

def codegen_node(state: AgentState):
    """Generate code with reasoning trace."""
    response  = code_generator.generate_code(state["question"])
    reasoning = getattr(response, 'reasoning', '')

    return {
        "code"             : response.code,
        "reasoning"        : reasoning,
        "regeneration_count": state.get("regeneration_count", 0),
        "code_history"     : state.get("code_history", []) + [response.code],
    }


def risk_estimation_node(state: AgentState):
    """Score the generated code with the ANN and decide pass/fail."""
    try:
        confidence = code_generator.get_last_raw_confidence()
        logprob    = code_generator.get_last_logprob()
    except Exception:
        confidence = None   # let compute_features use dataset mean
        logprob    = None

    risk_score, features = risk_estimator.predict_risk(
        state["question"],
        state["code"],
        confidence=confidence,
        logprob=logprob,
    )

    # Derive status purely from the numeric score — never rely on a string
    # that LangGraph might drop when routing to END via a conditional edge.
    final_status = "FAILED" if risk_score > RISK_THRESHOLD else "SUCCESS"

    print(
        f"[DEBUG risk_estimation_node] "
        f"risk={risk_score:.4f} threshold={RISK_THRESHOLD} "
        f"→ {final_status}  regen={state.get('regeneration_count', 0)}",
        file=sys.stderr
    )

    return {
        "risk_score"  : risk_score,   # always a plain Python float
        "features"    : features,
        "final_status": final_status,
    }


def failure_analysis_node(state: AgentState):
    """Analyze why the code was flagged as high-risk."""
    failure = failure_analyzer.analyze(
        state["question"],
        state["code"],
        state["test_cases"],
        state["execution_result"],
    )

    return {
        "failure_reason": failure.failure_reason,
        "severity"      : failure.severity,
    }


def regeneration_node(state: AgentState):
    """Regenerate code informed by the failure analysis."""
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
        "code"             : response.code,
        "reasoning"        : getattr(response, 'reasoning', ''),
        "regeneration_count": state.get("regeneration_count", 0) + 1,
        "code_history"     : state.get("code_history", []) + [response.code],
    }


# ── Routing functions ─────────────────────────────────────────────────────────

def should_execute(state: AgentState):
    """Route based on risk_score (the raw float), never on final_status string."""
    risk_score = state.get("risk_score")

    if risk_score is None:
        print("[DEBUG should_execute] risk_score is None → high_risk", file=sys.stderr)
        return "high_risk"

    if risk_score > RISK_THRESHOLD:
        print(
            f"[DEBUG should_execute] {risk_score:.4f} > {RISK_THRESHOLD} → high_risk",
            file=sys.stderr
        )
        return "high_risk"

    # Accepted — update history and exit
    print(
        f"[DEBUG should_execute] {risk_score:.4f} <= {RISK_THRESHOLD} → end (SUCCESS)",
        file=sys.stderr
    )
    risk_estimator.update_history(state["question"], state["code"], failed=False)
    return "end"


def should_regenerate(state: AgentState):
    """Decide whether to retry or give up after failure analysis."""
    regen_count = state.get("regeneration_count", 0)

    if regen_count < MAX_REGENERATIONS:
        print(
            f"[DEBUG should_regenerate] {regen_count} < {MAX_REGENERATIONS} → regenerate",
            file=sys.stderr
        )
        return "regenerate"

    # Max attempts exhausted
    print(
        f"[DEBUG should_regenerate] {regen_count} >= {MAX_REGENERATIONS} → end (FAILED)",
        file=sys.stderr
    )
    risk_estimator.update_history(state["question"], state["code"], failed=True)
    return "end"


# ── Graph construction ────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

builder.add_node("codegen",          codegen_node)
builder.add_node("risk_estimation",  risk_estimation_node)
builder.add_node("failure_analysis", failure_analysis_node)
builder.add_node("regenerate",       regeneration_node)

builder.add_edge(START, "codegen")
builder.add_edge("codegen", "risk_estimation")

builder.add_conditional_edges(
    "risk_estimation",
    should_execute,
    {"high_risk": "failure_analysis", "end": END},
)

builder.add_conditional_edges(
    "failure_analysis",
    should_regenerate,
    {"regenerate": "regenerate", "end": END},
)

builder.add_edge("regenerate", "risk_estimation")

graph = builder.compile()


# ── Utilities ─────────────────────────────────────────────────────────────────

def visualize_graph():
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        print("Graph visualization not available")


def run_pipeline(question: str) -> dict:
    """
    Run the complete failure-aware pipeline for a single question.

    Returns the final state dict with guaranteed keys:
        risk_score   : float
        final_status : 'SUCCESS' | 'FAILED'
        regeneration_count : int
        code         : str
        failure_reason : str | None
    """
    initial_state = {
        "question"          : question,
        "code"              : None,
        "test_cases"        : None,
        "execution_result"  : None,
        "failed_cases"      : 0,
        "passed_cases"      : 0,
        "failure_reason"    : None,
        "severity"          : None,
        "risk_score"        : None,
        "regeneration_count": 0,
        "code_history"      : [],
        "reasoning"         : None,
        "features"          : None,
        "final_status"      : None,
    }

    result = graph.invoke(initial_state)

    # ── CRITICAL FIX: derive final_status from risk_score, not graph state ────
    #
    # LangGraph returns the state snapshot at the END node.  When a conditional
    # edge routes directly to END (the SUCCESS path), the state dict that comes
    # back from graph.invoke() sometimes has final_status=None because the END
    # node itself never wrote to that key — it was only written inside
    # risk_estimation_node, and LangGraph's reducer doesn't guarantee the last
    # written value survives the conditional-edge jump to END in all versions.
    #
    # risk_score is a plain Python float returned by risk_estimation_node and is
    # always present.  We re-derive final_status from it here as the single
    # authoritative source of truth.
    risk_score = result.get("risk_score")

    if risk_score is not None:
        final_status = "SUCCESS" if risk_score <= RISK_THRESHOLD else "FAILED"
    else:
        final_status = "FAILED"   # absolute fallback — should never happen

    result["final_status"] = final_status

    # ── Summary print (captured by runner.py's redirect_stdout) ───────────────
    print(f"\n{'='*50}")
    print(f"Pipeline completed for: {question[:100]}...")
    print(f"Regeneration attempts: {result.get('regeneration_count', 0)}")

    if result.get('failure_reason'):
        print(f"Failure reason: {result['failure_reason']}")

    if risk_score is not None:
        print(f"Risk score: {risk_score:.3f}")

    # Exactly this format — runner.py regex depends on it
    print(f"Final status: {final_status}")

    return result


if __name__ == "__main__":
    sample_question = "Write a function that returns the sum of two numbers"
    result = run_pipeline(sample_question)
