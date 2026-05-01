# Context Management
Controlling What Gets Sent With Each Prompt (Production Patterns)

When you add “memory” to a chat app, you’re really managing **context**:
what previous information you choose to attach to the next prompt.  
In production, you usually **do not** send the full conversation every time.  
Instead, you use one (or a mix) of the patterns below.

---

## 1) Sliding Window (Last N Messages)

### Idea
Only send the **most recent** messages (e.g., last 6–12).  
Older messages remain stored, but they are **not injected** into the next prompt.

### Why it’s used
- Stable and simple
- Predictable cost and latency
- Works well for most normal conversations

### Technical implementation steps
- Store the full message history per session/user.
- Before each model call, select only the last **N** messages.
- Inject those selected messages into the prompt’s `{history}` placeholder.
- Keep N configurable (different N for different apps).
- Add safety: if the user asks about older details, you can increase N or use retrieval (Pattern #5).

---

## 2) Token Budget Trimming (Context by Size)

### Idea
Instead of “last N messages,” send “as much history as fits in a **token budget**.”  
You keep adding recent messages until you hit the limit, then drop older ones.

### Why it’s used
- More accurate cost control than “N messages”
- Avoids model errors when context window is exceeded
- Adapts to short vs. long messages automatically

### Technical implementation steps
- Decide a token budget for history (example: 1,500 tokens).
- Estimate token count for each message (best: model tokenizer; acceptable: approximation).
- Build the injected history from newest → oldest until budget is reached.
- Reserve extra budget for:
  - system instructions
  - user question
  - any retrieved documents/tools output
- Log token usage for monitoring (production observability).

---

## 3) Summary Buffer (Rolling Summary + Recent Messages)

### Idea
Keep a **short summary** of older conversation + the **recent** messages.  
The model sees: summary (compressed memory) + last few turns (details).

### Why it’s used
- Preserves long-term context without sending everything
- Good for long conversations (support bots, tutors, assistants)

### Technical implementation steps
- Maintain two memory parts per session:
  - `summary` (short text)
  - `recent_messages` (last N turns)
- After every X turns (or when size exceeds a threshold):
  - generate/update the summary using a summarization prompt
  - clear or reduce the older messages that were summarized
- Inject into prompt in this structure:
  - system instructions
  - “Conversation Summary: …”
  - recent messages
  - new user question
- Add a “summary quality” check (keep it factual, short, and user-specific).

---

## 4) Fact Memory (User Profile / Key Facts Store)

### Idea
Store **stable facts** separately from raw chat history:
name, preferences, constraints, goals, important decisions.

### Why it’s used
- Reduces noise from full transcripts
- Improves personalization and consistency
- Prevents losing key facts when you trim history

### Technical implementation steps
- Create a structured “facts store” per session/user (simple key-value or JSON).
- After each turn, run a small “extract facts” step:
  - add new facts
  - update changed facts
  - delete incorrect facts
- Inject only the facts (not the full chat) into the prompt:
  - “User Facts: …”
- Keep facts small and curated (avoid storing everything).
- Add privacy rules: never store sensitive data unless required and permitted.

---

## 5) Memory Retrieval (Selective Recall from Past)

### Idea
Treat past messages like a searchable database.
For each new question, retrieve only the **most relevant** past snippets.

### Why it’s used
- Best for long-running conversations
- High relevance, low token cost
- Works well when the user references older topics

### Technical implementation steps
- Store past messages (or summaries) with metadata:
  - timestamp, topic, user vs assistant
- Build an index for retrieval:
  - keyword search (BM25) and/or embeddings
- On every new user question:
  - search the memory store
  - fetch top relevant past snippets
- Inject into prompt as “Relevant Past Context” (not full history).
- Add rules to avoid injecting irrelevant or sensitive memory.

---

## 6) Topic-Based Context (Scoped Memory by Thread)

### Idea
Only include memory that matches the current **topic/thread**.
If topic changes, don’t carry old context.

### Why it’s used
- Prevents cross-topic confusion
- Useful for multi-topic assistants (students ask unrelated questions)

### Technical implementation steps
- Add a lightweight “topic detector” step each turn:
  - classify the question into a topic label
- Maintain memory per topic:
  - `topic_id → messages/summary/facts`
- When generating:
  - inject only the active topic memory
- If topic is unclear:
  - ask a single clarifying question or choose the closest topic.

---

## 7) Context Sanitization (Cleaning Before Injection)

### Idea
Before sending anything to the model, clean the context to reduce:
repetition, irrelevant lines, long logs, prompt injection attempts.

### Why it’s used
- Improves answer quality
- Reduces token waste
- Adds security hardening

### Technical implementation steps
- Remove duplicated assistant answers or boilerplate.
- Strip very long messages or replace them with short summaries.
- Remove tool logs / stack traces unless needed.
- Apply safety filters to user-provided instructions inside history.
- Keep system instructions authoritative (never overridden by history).

---

# Recommended Production Combinations

## A) Typical Chat Assistant
- Sliding Window + Token Budget Trimming
- Optional: Fact Memory for personalization

## B) Long Tutoring / Support Chat
- Summary Buffer + Recent Messages
- Fact Memory (user goals + constraints)

## C) Research / Complex Multi-Session
- Memory Retrieval + Fact Memory
- Topic-Based Context (if users jump between topics)

---

# What to Measure (So You Know It Works)
- Cost per turn (tokens) and latency
- Answer relevance to the latest question
- “Remembers correct facts” rate
- Hallucination rate when history is trimmed
- User satisfaction (did it forget important details?)

---

# Practical Next Steps
1) Start with **Sliding Window** (simple and stable).
2) Add **Token Budget Trimming** to prevent context overflow.
3) If chats get long, add **Summary Buffer**.
4) For best quality, add **Fact Memory** (small curated profile).
5) For long-running sessions, add **Memory Retrieval** (selective recall).