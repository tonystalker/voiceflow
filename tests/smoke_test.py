"""Quick smoke test — verifies all components load and the agent graph runs end-to-end."""
import sys
sys.path.insert(0, ".")

print("1/5  Loading config...")
from app.config import settings
print(f"     Groq model : {settings.groq_model}")
print(f"     Qdrant     : {settings.qdrant_host}:{settings.qdrant_port}")

print("2/5  Loading RAG retriever...")
from app.rag.retriever import RAGRetriever
retriever = RAGRetriever()
results = retriever.search("how do I reset my password", top_k=2)
print(f"     Top hit    : score={results[0]['score']:.3f}")
print(f"     Answer     : {results[0]['text'][:80]}...")

print("3/5  Loading LangGraph agent...")
from app.agent.graph import agent_graph
print(f"     Graph nodes: {list(agent_graph.nodes.keys())}")

print("4/5  Running a full agent turn (FAQ)...")
from app.agent.state import ConversationState
state: ConversationState = {
    "call_sid": "SMOKE_TEST",
    "turn_count": 1,
    "transcript_partial": "",
    "transcript_final": "How do I reset my online banking password?",
    "intent": None,
    "retrieved_context": [],
    "llm_response": None,
    "tool_result": None,
    "escalate_flag": False,
    "barge_in_detected": False,
    "latency_log": {"stt_ms": 0, "llm_ms": None, "tts_first_chunk_ms": None, "total_ms": None},
}
result = agent_graph.invoke(state)
print(f"     Intent     : {result['intent']}")
print(f"     Response   : {result['llm_response']}")

print("5/5  Running a full agent turn (account query)...")
state2 = dict(state)
state2["turn_count"] = 2
state2["transcript_final"] = "Can you check my account ACC-1001?"
result2 = agent_graph.invoke(state2)
print(f"     Intent     : {result2['intent']}")
print(f"     Response   : {result2['llm_response']}")

print("\n[PASS] All checks passed - pipeline is fully operational!")
