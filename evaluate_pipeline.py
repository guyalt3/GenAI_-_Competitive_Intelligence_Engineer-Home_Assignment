"""Automated Evaluation Suite for the Competitor Intelligence RAG Pipeline."""

import io
import json
import os
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent import get_competitor_insights, initialize_environment

GOLDEN_DATASET = [
    # --- Static questions, quick cache-hit ---
    {
        "query": "What are the core technical advantages of Snyk in the SCA market?",
        "competitor": "Snyk",
        "type": "static_baseline"
    },
    {
        "query": "How does Snyk integrate with developer IDEs and repositories?",
        "competitor": "Snyk",
        "type": "static_baseline"
    },
    {
        "query": "What are GitHub Advanced Security's main security features?",
        "competitor": "GitHub Advanced Security",
        "type": "static_baseline"
    },
    # --- Current questions, cache might include partial answers---
    {
        "query": "What are the latest announcements Snyk made recently regarding AI features?",
        "competitor": "Snyk",
        "type": "recent_news_2026"
    },
    {
        "query": "Tell me about Snyk's acquisition of Invariant Labs and its purpose.",
        "competitor": "Snyk",
        "type": "recent_news_2026"
    },
    {
        "query": "What agent security tools or features did Snyk launch at RSAC 2026?",
        "competitor": "Snyk",
        "type": "recent_news_2026"
    },
    {
        "query": "What is GitHub Secret Protection and what features does it include?",
        "competitor": "GitHub Advanced Security",
        "type": "recent_news_2026"
    },
    # --- Broad-scope market questions - cache-miss ---
    {
        "query": "What are the overall latest security features across all competitors recently?",
        "competitor": "All Competitors",
        "type": "global_market"
    },
    {
        "query": "How are Snyk and GitHub competing in the developer-centric AI security space?",
        "competitor": "All Competitors",
        "type": "global_market"
    },
    {
        "query": "What are the major vulnerability management trends across security vendors?",
        "competitor": "All Competitors",
        "type": "global_market"
    }
]

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI quality assurance judge specializing in RAG systems.\n"
            "Your task is to evaluate a generated answer based STRICTLY on the user query and the compiled context provided.\n\n"
            "You must output a valid JSON containing exactly three scores between 0.0 and 1.0:\n"
            
            "--- SCORE 1-5 RUBRICS ---\n\n"
            "1. faithfulness (Grounding):\n"
            "   - 1.0: Every single claim in the answer is 100% supported by the context.\n"
            "   - 0.75: Mostly supported, but minor unverified assumptions are made.\n"
            "   - 0.50: Moderate support, but contains significant claims or numbers not found in the context.\n"
            "   - 0.25: Weak support, many claims are guessed or hallucinated.\n"
            "   - 0.0: Completely unsupported or directly contradicts the context.\n\n"
            
            "2. answer_relevance (Clarity & Focus):\n"
            "   - 1.0: Directly answers the user's query perfectly and concisely without filler.\n"
            "   - 0.75: Answers the query but includes slightly redundant or unnecessary information.\n"
            "   - 0.50: Addresses the topic but misses the core intent or includes massive off-topic sections.\n"
            "   - 0.25: Barely relevant, talks around the topic without answering the actual question.\n"
            "   - 0.0: Completely off-topic or fails to provide an answer.\n\n"
            
            "3. context_precision (Signal-to-Noise Ratio):\n"
            "   - 1.0: The context is highly focused, containing ONLY necessary DevSecOps/AppSec intelligence to answer the query.\n"
            "   - 0.75: The context has the right answers, but includes some minor irrelevant text/formatting fluff.\n"
            "   - 0.50: The context contains the answer but is flooded with non-DevSecOps noise (e.g., physical security, hardware firewalls, unrelated news).\n"
            "   - 0.25: Mostly noise, requires picking a single sentence from a sea of irrelevant text.\n"
            "   - 0.0: Completely empty, broken, or entirely noisy context with zero utility.\n\n"
            
            "Output format must be exactly: {{\"faithfulness\": X.X, \"answer_relevance\": X.X, \"context_precision\": X.X}}"
        ),
        (
            "human",
            "User Query:\n{query}\n\n"
            "Compiled Context (The Source Material):\n{context}\n\n"
            "Generated Answer:\n{answer}\n\n"
            "Return ONLY the requested JSON object."
        )
    ]
)


def run_evaluation() -> None:
    """Execute the evaluation suite across the test dataset."""
    initialize_environment()
    # judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/gpt-4o-mini",  # המודל דרך OpenRouter
        temperature=0,
    )
    json_parser = JsonOutputParser()
    judge_chain = JUDGE_PROMPT | judge_llm | json_parser

    print(f"=== Starting Automated RAG Evaluation Suite ({len(GOLDEN_DATASET)} test cases) ===")
    
    results = []
    
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_precision = 0.0

    for idx, test_case in enumerate(GOLDEN_DATASET, start=1):
        query = test_case["query"]
        competitor = test_case["competitor"]
        q_type = test_case["type"]
        
        print(f"\n[{idx}/10] Testing [{competitor}] - Type: {q_type}...")
        print(f"Query: '{query}'")

        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer):
            answer, context = get_competitor_insights(query, competitor, include_jfrog_impact=False)
        
        logs_text = log_buffer.getvalue()
        routing_mode = "Cache Hit" if "Cache Hit!" in logs_text else "Cache Miss (Web Search)"
        print(f"Execution Mode: {routing_mode}")

        try:
            scores = judge_chain.invoke({"query": query, "context": context, "answer": answer})
            f_score = scores.get("faithfulness", 0.0)
            r_score = scores.get("answer_relevance", 0.0)
            p_score = scores.get("context_precision", 0.0)
        except Exception as e:
            print(f"  [Warning] Judge failed to parse response: {e}")
            f_score, r_score, p_score = 0.0, 0.0, 0.0

        print(f"  Scores -> Faithfulness: {f_score} | Relevance: {r_score} | Context Precision: {p_score}")
        
        total_faithfulness += f_score
        total_relevance += r_score
        total_precision += p_score

        results.append({
            "query": query,
            "context": context, 
            "answer": answer,
            "competitor": competitor,
            "type": q_type,
            "routing": routing_mode,
            "scores": {"faithfulness": f_score, "answer_relevance": r_score, "context_precision": p_score}
        })

    count = len(GOLDEN_DATASET)
    avg_f = total_faithfulness / count
    avg_r = total_relevance / count
    avg_p = total_precision / count

    print("\n" + "="*60)
    print("=== FINAL EVALUATION METRICS SUMMARY (RAG TRIAD) ===")
    print("="*60)
    print(f"📊 System Faithfulness (No Hallucinations): {avg_f:.2f} / 1.00")
    print(f"🎯 System Answer Relevance:                 {avg_r:.2f} / 1.00")
    print(f"🔍 System Context Precision:                {avg_p:.2f} / 1.00")
    print("="*60)

    report_path = Path("data_input/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {"avg_faithfulness": avg_f, "avg_answer_relevance": avg_r, "avg_context_precision": avg_p},
        "details": results
    }
    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full report exported successfully to '{report_path}'.\n")


if __name__ == "__main__":
    run_evaluation()