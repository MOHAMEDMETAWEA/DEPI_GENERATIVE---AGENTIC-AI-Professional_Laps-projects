# Task 2: Summary Buffer Memory Upgrade

## Goal
The required upgrade is to move memory handling from `full history` to a stronger production pattern:
- `Rolling Summary` for older conversation
- + `Recent Messages` (last N messages)

This way, the model gets long context in a compressed form with lower cost.

---

## Required Idea
Instead of sending full history to the LLM every time:
1. Keep a text `summary` for each `session_id`.
2. Also keep only the last `N` messages as `recent messages`.
3. Before each model call, send:
    - summary of previous conversation
    - last N messages
    - current user question

---

## What You Need to Implement
1. In `app.py`, add to session state:
    - `summary: str` (starts as empty value)
    - `recent_history` (or any suitable structure for recent messages)

2. Add settings (preferably from `.env` with default values):
    - `RECENT_WINDOW_SIZE=6`
    - `SUMMARY_UPDATE_EVERY=4` (how often, in messages/turns, the summary is updated)

3. Add a summarization step inside `chat()` logic:
    - after a fixed number of messages (`SUMMARY_UPDATE_EVERY`), generate an updated summary.
    - the new summary must be `rolling` (based on old summary + newest part of conversation).

4. Update the prompt to include a clear place for summary:
    - logical example:
      - system instruction
      - `Conversation Summary: {summary}`
      - `{history}` (only the last N messages)
      - new question

5. In `api.py`, add a new endpoint:
    - `GET /summary/{session_id}`
    - returns:
      - `session_id`
      - `summary`
      - `recent_count`

---

## Important Constraints
  - Do not change the response format of `/chat`.
  - Keep saving full `raw_history` for display/debugging purposes.
  - Session isolation must stay 100% correct.
  - If there is no summary yet, the system must still work normally.

---

## Acceptance Criteria
1. After sending multiple messages (for example 10 messages), `/summary/{session_id}` has a non-empty summary.
2. The LLM does not receive full history; only summary + last `RECENT_WINDOW_SIZE` messages.
3. `/history/{session_id}` still returns full history as before.
4. `/clear` clears:
    - `raw_history`
    - `summary`
    - any recent buffer
5. No context leakage between `student1` and `student2`.

---

## Quick Manual Test
1. Run API:
    ```bash
    uvicorn api:api --reload --host 127.0.0.1 --port 8000
    ```

2. Use `session_id=student1` and send 8-12 messages.

3. Check:
    ```bash
    GET /summary/student1
    GET /history/student1
    ```

4. Ask a question that depends on very old information from the start of the conversation, and verify the answer is still correct (because of summary).

5. Run clear, then check:
    ```bash
    POST /clear
    GET /summary/student1
    GET /history/student1
    ```

---

## Bonus (Optional)
  - Add endpoint `POST /summary/{session_id}/refresh` to force an immediate re-summary.
  - Add `summary_version` or `updated_at` to monitor summary updates.
  - Add protection against summary growth (for example max length then re-compress).

---

## Submission
- Updated code.
  - File `TASK4_NOTES.md` that includes:
    - How did you build the rolling summary?
    - When does it update? Why did you choose this timing?
    - Real before/after examples (short text or JSON from endpoints).
