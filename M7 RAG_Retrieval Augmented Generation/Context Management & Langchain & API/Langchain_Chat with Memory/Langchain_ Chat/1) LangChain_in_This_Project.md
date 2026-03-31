# Introduction to LangChain

LangChain is a framework that helps you build applications powered by Large Language Models (LLMs) like GPT.

Instead of calling the model directly every time, LangChain gives you **structure + tools** to build real systems.

👉 Think of it like:
- Python = general programming  
- LangChain = framework to build AI systems on top of LLMs  

---

# What does LangChain give you?

LangChain is NOT just about calling an API.  
It helps you manage the full workflow around the LLM.

Main components:

- **Prompt Templates** → organize how you talk to the model  
- **Chains** → connect steps together (Prompt → Model → Output)  
- **Memory** → keep conversation history  
- **Retrievers (RAG)** → connect your data (PDFs, DB, etc.)  
- **Agents** → let the model use tools (search, code, APIs)  

👉 So instead of writing everything from scratch, you reuse these building blocks.

---

# When should you use LangChain?

Use LangChain when:

### 1. You are building AI apps
- Chatbots  
- RAG systems (chat with your data)  
- AI assistants  
- Multi-step workflows  

---

### 2. You need memory
- conversation history  
- session tracking  

---

### 3. You need to connect data
- PDFs  
- databases  
- APIs  

---

### 4. You build agents
- tool calling  
- decision-making systems  

---

# When NOT to use LangChain?

Use normal programming when:

### 1. Simple use case
- one prompt → one response  
👉 Just call OpenAI / API directly  

---

### 2. You need full control
- strict performance  
- custom pipelines  
- low-level optimization  

---

### 3. Production-critical systems
Sometimes LangChain adds abstraction you don’t need.  
👉 Direct implementation can be more stable.

---

# Real-World Insight (Important)

LangChain is great for:
👉 **prototyping + building fast AI systems**

But in production:
👉 many teams **replace parts of LangChain** with custom code

Why?
- better control  
- better performance  
- easier debugging  

---

# Final Mental Model

Think like this:

👉 If you are exploring or building AI workflows  
→ **Use LangChain**

👉 If you already know exactly what you want  
→ **Build it manually (normal programming)**

---

--- 

# Using LangChain in This Project

## Overview

This project builds a **chat system with memory** using LangChain.

The goal is simple:

> Create a chatbot that remembers previous messages per user session and responds based on conversation context.

---

# Core Components

The system uses LangChain as an **orchestration layer** to connect:

- Prompt (how we ask the model)
- LLM (the model itself)
- Memory (conversation history)

---

# 1️ LLM Integration

### What we use

LangChain provides a unified interface to interact with language models.

### Idea

Instead of calling APIs directly, LangChain wraps the model and gives a standard way to:

- send input
- receive output
- switch providers easily

### Why it matters

- You can change the model without rewriting logic
- Same structure works for OpenAI, HuggingFace, etc.


---

### Technical Steps

#### Step 1 — Define Model Configuration
- Choose model provider (HuggingFace Router)
- Set model name
- Configure temperature

#### Step 2 — Initialize LLM Wrapper
- Use LangChain LLM interface
- Pass API key and base URL
- Abstract away raw HTTP calls

#### Step 3 — Standardize Interaction
- Ensure all inputs/outputs follow same format
- Enable switching models without changing logic

---

### Result

- Model becomes plug-and-play
- No direct dependency on provider-specific APIs

---

# 2️ Prompt + Model Pipeline

### Concept

LangChain allows chaining components into a pipeline:

> Prompt → Model → Output

### How it works (conceptually)

1. The prompt defines how the question is formatted
2. The model receives the formatted input
3. The output is returned as a response

### Why this is powerful

- Clear separation between logic and model
- Easy to modify prompts without touching system design
- Reusable pipelines
---

### Technical Steps

#### Step 1 — Define Prompt Template
- Create structured prompt
- Include placeholders (e.g., question, history)

#### Step 2 — Bind Prompt to Model
- Create a pipeline connecting:
  - Prompt
  - LLM

#### Step 3 — Standardize Input Flow
- Input always enters through the prompt
- Prompt formats data before sending to model

#### Step 4 — Output Handling
- Receive structured response from model
- Extract final text output

###  Result

- Clean separation between logic and model
- Easy to modify prompts without touching system core

---

# 3️ Memory System (Key Feature)

### Problem

By default, LLMs are stateless:
- They do not remember previous messages

### Solution

LangChain introduces **memory abstraction**

### What happens

For each request:

1. Retrieve previous messages (history)
2. Combine them with the current question
3. Send everything to the model

---

### Technical Steps

#### Step 1 — Define Memory Store
- Create a storage structure per session
- Map:
  session_id → memory object

#### Step 2 — Initialize Memory Object
- Use LangChain chat history abstraction
- Store messages in structured format

#### Step 3 — Implement History Retrieval
- Build function to:
  - receive session_id
  - return corresponding memory

#### Step 4 — Attach Memory to Pipeline
- Wrap pipeline with memory handler
- Ensure memory is automatically:
  - read before request
  - updated after response

### Result

The model behaves like a real conversation:

- understands context
- refers to previous answers
- maintains flow

---

# 4️ Session-Based Memory

### Idea

Each user has a separate session:

> session_id → conversation history

### Why important

- Multiple users can use the system at the same time
- Each user has isolated memory
- No mixing between conversations

###  Technical Steps

#### Step 1 — Create Session Store
- Use dictionary-like structure
- Key: session_id
- Value: session data (memory + raw history)

#### Step 2 — Ensure Session Exists
- On each request:
  - check if session exists
  - if not → create new session

#### Step 3 — Pass Session ID to LangChain
- Attach session_id in runtime config
- Use it to fetch correct memory

#### Step 4 — Maintain Isolation
- Ensure no overlap between sessions
- Each session handled independently
---

# 5️ Dual History Design

This project uses two types of history:

### A) LangChain Memory

Used internally by the system:

- sent to the model
- affects responses

### B) Raw History

Used for:

- debugging
- UI display
- tracking conversation manually

#### Key Insight

> Only LangChain memory affects the AI response.

---


## End-to-End Flow

### Technical Steps

#### Step 1 — Receive User Input
- Validate input (non-empty)

#### Step 2 — Retrieve Session
- Ensure session exists
- Load memory

#### Step 3 — Store User Message
- Append to raw history

#### Step 4 — Invoke Pipeline
- Pass:
  - question
  - session_id

#### Step 5 — Memory Injection (LangChain)
- Automatically:
  - fetch history
  - inject into prompt

#### Step 6 — Model Generates Response
- LLM processes:
  history + question

#### Step 7 — Store AI Response
- Append to raw history

#### Step 8 — Return Output
- Send response back to user

---

# Production Considerations

## 1. Memory Storage

Current approach:
- In-memory storage (temporary)

Limitations:
- Data is lost when server restarts
- Not scalable

Better alternatives:
- Redis (fast, scalable)
- PostgreSQL (persistent)
- Vector DB (for advanced memory)

---

## 2. Context Size Problem

Sending full history every time is expensive.

Solutions:
- Window memory (last N messages)
- Summarization memory
- Retrieval-based memory

---

## 3. Multi-User Scaling

As users grow:

- memory must be externalized
- session management becomes critical
- latency must be controlled

---

# Key Takeaways

- LangChain is not just a wrapper → it is an orchestration system
- It connects prompt + model + memory into one pipeline
- Memory is what makes the chatbot feel intelligent
- Session-based design is essential for real applications
- Production systems require optimized memory strategies