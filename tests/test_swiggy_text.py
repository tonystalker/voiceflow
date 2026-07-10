import asyncio
from app.agent.graph import agent_graph
from app.agent.state import ConversationState

async def run_test():
    print("Testing Swiggy text flow...")
    state: ConversationState = {
        "call_sid": "TEST",
        "turn_count": 1,
        "transcript_partial": "",
        "transcript_final": "order biryani from a nearby restaurant",
        "intent": None,
        "retrieved_context": [],
        "llm_response": None,
        "tool_result": None,
        "pending_action": None,
        "confirmed": None,
        "escalate_flag": False,
        "barge_in_detected": False,
        "latency_log": {
            "stt_ms": None,
            "llm_ms": None,
            "tts_first_chunk_ms": None,
            "total_ms": None,
        }
    }
    
    print(f"[YOU] {state['transcript_final']}")
    
    # We use ainvoke now!
    final_state = await agent_graph.ainvoke(state)
    
    intent = final_state.get("intent")
    print(f"\n[INTENT] {intent}")
    
    if final_state.get("tool_result"):
        print(f"\n[TOOL RESULT] {final_state['tool_result']}")
        
    print(f"\n[TONY] {final_state.get('llm_response', '')}")
    
    # ── Turn 2: User confirms ──
    print("\n--- TURN 2 ---")
    state2: ConversationState = {
        "call_sid": "TEST",
        "turn_count": 2,
        "transcript_partial": "",
        "transcript_final": "yes, please go ahead",
        "intent": None,
        "retrieved_context": [],
        "llm_response": None,
        "tool_result": None,
        "pending_action": final_state.get("pending_action"),
        "confirmed": None,
        "escalate_flag": False,
        "barge_in_detected": False,
        "latency_log": {
            "stt_ms": None,
            "llm_ms": None,
            "tts_first_chunk_ms": None,
            "total_ms": None,
        }
    }
    print(f"[YOU] {state2['transcript_final']}")
    final_state2 = await agent_graph.ainvoke(state2)
    
    intent2 = final_state2.get("intent")
    print(f"\n[INTENT] {intent2}")
    
    if final_state2.get("tool_result"):
        print(f"\n[TOOL RESULT] {final_state2['tool_result']}")
        
    print(f"\n[TONY] {final_state2.get('llm_response', '')}")

if __name__ == "__main__":
    asyncio.run(run_test())
