content = '''
---

## 6. V2 Bottlenecks & Fixes (Assistant Mode & Swiggy MCP)

### Bug 1 — Async Callback Crash in Deepgram SDK v3

**Problem:**
Deepgram silently closed its connection, waiting for audio, because the background task to listen for the query crashed with `TypeError: a coroutine was expected, got None`. The mic never started listening.

**Root Cause:**
Deepgram v3 SDK uses `asyncio.create_task` inside its `_emit` handler for callbacks. We mistakenly ran `asyncio.create_task(self.listen_for_query())` from a synchronous loop context and didn't wait or retrieve exceptions properly, so the task crashed silently and the error was swallowed.

**Fix:**
Renamed the method to `start_deepgram` and attached a done-callback: `task.add_done_callback(lambda t: t.exception() and logger.error(...))`. This exposed the underlying error.

### Bug 2 — Wake-Word Audio Bleeding into STT

**Problem:**
STT latencies appeared as 34s long. The wake word "Hey Jarvis" audio tail was fed into Deepgram, which then waited for more speech that didn't arrive, delaying endpointing.

**Root Cause:**
Transition from `IDLE` to `LISTENING` was so fast that physical trailing audio in the PyAudio buffer was pumped to Deepgram.

**Fix:**
Added `await asyncio.sleep(0.5)` and a secondary queue flush loop before feeding Deepgram in `LISTENING` state.

### Bug 3 — Groq Token Rate Limits (HTTP 413) with Swiggy MCP

**Problem:**
The Swiggy node crashed with `unhandled errors in a TaskGroup (1 sub-exception)` which was a `groq.APIStatusError: Error code 413`.

**Root Cause:**
Swiggy MCP exposes 18 complex tools. Injecting all 18 tool schemas into LangChain consumed 12,349 tokens, completely exceeding `llama-3.1-8b-instant`'s 6,000 TPM limit.

**Fix:**
Filtered the toolset before passing to the ReAct agent, leaving only 6 essential tools (Cart, Search, Order). Upgraded to `llama-3.3-70b-versatile` which has a 12,000 TPM limit.

### Bug 4 — Missing State Context for Confirmation (Hallucination Loop)

**Problem:**
Saying "I would like to buy a chicken" routed to the `faq` node instead of Swiggy. The FAQ node asked "Would you like to order via Swiggy?", but saying "Yes" routed to `out_of_scope`.

**Root Cause:**
`intent_classification_node` relied on strict keywords. Because "buy" and "chicken" weren't in the list, it routed to `faq`. Since `faq` is handled by `response_generation_node`, it didn't set `pending_action`, leaving the graph with no context for "Yes".

**Fix:**
Expanded the heuristic keyword list in `intent_classification_node` to include "buy", "chicken", "roll", etc., routing direct food actions to the `action_intent` node.

### Bug 5 — Groq Schema Type Validation (String vs Int)

**Problem:**
Groq threw a 400 error on the Swiggy `search_restaurants` tool call because it generated `"0"` (string) for the `offset` parameter, which the MCP schema expects as an integer.

**Fix:**
Appended a strict prompt instruction directly into the Swiggy ReAct agent: `CRITICAL: When calling tools ... ensure numeric parameters like 'offset' are passed as raw JSON integers (e.g. 0), NOT strings`.
'''

with open('V1_DEVELOPMENT_LOG.md', 'a', encoding='utf-8') as f:
    f.write(content)
