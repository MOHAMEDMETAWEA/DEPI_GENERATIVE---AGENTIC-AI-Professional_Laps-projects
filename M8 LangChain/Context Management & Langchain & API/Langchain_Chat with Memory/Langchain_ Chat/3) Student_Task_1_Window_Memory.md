# Task 1: Sliding Window Memory Upgrade

## Goal
After you understand `LangChain + Session Memory` in the current project, your task is to upgrade the chat so it:
- keeps the full conversation in `raw_history` (for display and debugging)
- but sends only the last fixed number of messages to the LLM (Sliding Window)

---

## What You Need to Implement
1. Add a new setting called `MEMORY_WINDOW_SIZE` (for example from `.env`, with default value `6`).
2. Update the `chat()` logic in `app.py` so that:
    - all messages are still saved in `raw_history`.
    - messages sent to the model are only the last `N` messages (based on `MEMORY_WINDOW_SIZE`).
3. In `api.py`, update the history endpoint:
    - `GET /history/{session_id}?limit=20`
    - if `limit` is provided, return only the last `limit` items.
4. Add a new endpoint:
    - `GET /session/{session_id}/stats`
    - returns:
      - `total_messages`
      - `human_messages`
      - `ai_messages`

---

## Important Constraints
  - Do not change the current response format of `/chat`.
  - Do not break session isolation (`session_id`).
  - Any new session must work without errors even if it has no history.

---

## Acceptance Criteria (all must pass)
1. If a session has 30 messages, `/history/{session_id}` returns all of them.
2. If you call `/history/{session_id}?limit=5`, it returns only the last 5.
3. The LLM must not receive more than `MEMORY_WINDOW_SIZE` history messages per request.
4. `/session/{session_id}/stats` shows correct numbers that match history.
5. `/clear` and `Show History` in `ui_gradio_api.py` still work with no change in core behavior.

---

## Quick Manual Test
1. Run API:
    ```bash
    uvicorn api:api --reload --host 127.0.0.1 --port 8000
    ```
2. Send 8 messages to the same `session_id`.
3. Check:
    ```bash
    GET /history/student1
    GET /history/student1?limit=3
    GET /session/student1/stats
    ```
4. Try a second session (`student2`) and make sure there is no data mixing.

---

## Bonus (Optional)
  - Support `MEMORY_WINDOW_SIZE` per session instead of one global value.
  - Add validation to reject `limit <= 0`.
  - Log the active window value for each request.

---

## Submission
- Updated code.
- A short file `TASK3_NOTES.md` that includes:
  - What did you change?
  - What design decisions did you make?
  - Screenshot output (or text output) for the 3 main test requests.
