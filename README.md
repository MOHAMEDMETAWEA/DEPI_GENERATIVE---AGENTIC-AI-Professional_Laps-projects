# 🤖 DEPI: Generative & Agentic AI Professional 🚀

Welcome to the comprehensive repository for the **Generative & Agentic AI Professional** track. This repository serves as a centralized hub for all labs, projects, and practical assignments throughout this 120+ hour intensive program.

## 📌 Quick Overview

This program takes you from **Python fundamentals** through **advanced Autonomous Agents and Retrieval-Augmented Generation (RAG)** systems. Each module builds on previous knowledge with hands-on notebooks, real-world datasets, and production-ready code patterns.

**Total Modules**: 11 (Foundations + 8 Core Modules + 2 Integration/Deployment Streams)  
**Learning Format**: Interactive Jupyter notebooks, Python scripts, and practical labs  
**Target Audience**: Developers, data scientists, and AI practitioners  
**Prerequisites**: Basic programming knowledge (provided in foundational modules)

---

## 🛤️ Course Journey & Syllabus

This track is designed to take a practitioner from foundational programming skills through the cutting edge of Autonomous Agents and Multi-Agent Systems. Below is the complete roadmap of the curriculum spanning 120+ hours.

### 📊 Program Structure

The program is organized into three progressive layers:

1. **🔧 Foundations (AI- Workshop Material)** — Essential Python and SQL skills required for AI/ML work
2. **🧠 Core AI/ML Track (M2–M7)** — Advanced Machine Learning through Agentic AI and RAG systems
3. **🌐 Production & Integration (Hugging Face & FastAPI)** — Deployment patterns and real-world applications

### 🗺️ Curriculum Roadmap

| Module | Focus Area | Status |
| :--- | :--- | :--- |
| **00 Foundations** | Python Essentials, SQL Fundamentals, & Programming Basics | ✅ Complete |
| **01 Intro & Orientation** | Environment setup, Ethics, & AI Landscape | ✅ Complete |
| **02 ML Fundamentals** | Regression, Classification, & Evaluation Metrics | ✅ Complete |
| **03 Deep Learning & Transformers** | CNNs, RNNs, LSTMs, & Transformer Architectures | ✅ Complete |
| **04 Generative AI Fundamentals** | VAEs, GANs, Diffusion Models, & Fine-tuning (LoRA) | ✅ Complete |
| **05 Advanced Prompting** | Chain-of-Thought, Reasoning, & Function Calling | ✅ Complete |
| **06 Agentic AI Fundamentals** | Autonomous Agents, Planning, & Tool Use | ✅ Complete |
| **07 RAG & Memory Systems** | Fusion, Re-ranking, & Memory Augmented Models | ✅ Complete |
| **08 Deployment & Integration** | FastAPI, Gradio, & Production Patterns | ✅ Complete |
| **09 Advanced Systems** | Semantic Cache, GraphRAG, & Agentic RAG | ✅ Complete |
| **10 Governance & Capstone** | Industry Trends & Final Project Implementation | ⏳ Upcoming |

---

## 📁 Repository Structure

The labs and projects are organized by module following the course progression:

### 📚 Foundational Materials

- **`AI- Workshop Material`**: Foundation workshops covering essential skills before the main curriculum.
  - **Python Material**: Comprehensive Python training with 7 progressive modules:
    - `0-First Workshop(22-1)` — Python basics and environment setup
    - `1- Variables, loops, Conditions, Functions (29-1-2026)` — Core programming constructs
    - `2- Data Structures (1-2-2026)` — Lists, dictionaries, sets, and tuples
    - `3- Python exercise (3-2-2026)` — Hands-on practice problems
    - `4- OOP (10-2-2026)` — Object-Oriented Programming principles
    - `5 - OOP & APIs (17-2-2026)` — OOP applications and API integration
    - `General Review & Full System Workshop (24-2)` — Comprehensive review of all Python concepts
  - **SQL Material**: Database fundamentals and advanced querying:
    - `1- Basic SQL (Create, Insert, Update, Alter) (21-2-2026)` — SQL data definition and manipulation
    - `2- Select, Joins, Aggregates (Group By, Order By) 28-2` — Advanced querying and data analysis

### 🧠 Core Modules

- **`M2 ML Fundamentals`**:
  - Implementation of core supervised and unsupervised algorithms.
  - Labs covering regression, classification, and clustering.
  - **Files**: `1 ML fundamentals.ipynb`, `2. machine-learning-Hands-On Demo.ipynb`, `3 ML_Session1_Exercises.ipynb`, `students_clean.csv`

- **`M3 Deep Learning & Transformers`**:
  - Journey from basic Neural Networks to Transformers.
  - Sessions on Computer Vision (CNNs) and NLP (RNNs, LSTMs).
  - Deep dive into Transformer architectures (BERT, GPT).
  - **Sessions**:
    - **Practice NLP - Representation**: Text representation and NLP techniques
    - **Session 1**: Neural Network basics, architecture, optimization, and regularization
    - **Session 2**: Computer Vision with CNNs, image representation, and color detection
    - **Session 3**: NLP data preprocessing for text modeling
    - **Session 4**: RNN & LSTM networks for sequence modeling
    - **Session 5**: Transformer architectures and attention mechanisms

- **`M4 Generative AI Fundamentals`**:
  - **VAEs**: Variational Autoencoders for latent space representation
    - `VAE_Code.ipynb`, `VAE_Code_enhanced.ipynb`, `VAE_Code_Keras.ipynb`
  - **GANs**: Generative Adversarial Networks for image generation
  - **Diffusion Models**: Noise prediction and reverse diffusion processes
  - **LoRA**: Parameter-efficient fine-tuning for Stable Diffusion

- **`M5 Prompt Eng`**:
  - **Part 1**: Microsoft Phi-3.5 mini-instruct for instruction tuning and benchmarking
  - **Part 2**: Function calling for implementing decision engines and structured outputs
  - **Advanced Topics**: Chain-of-Thought prompting, multi-step reasoning, and tool integration

- **`M6 Agentic`**:
  - **S1–S2 Core Agent Patterns**: Orchestration, stateful agents, interactive loops, and Code-Acting (LLM-generated pandas)
    - 📓 `Lab1_Safe_Sales_Insights_Agent.ipynb` — Decision-Driven Code-Acting Agent with policy enforcement, AST sandbox, and 6-query test suite
    - 📄 `lab1_walkthrough.md` — Implementation guide and best practices
    - Supporting notebooks: `1)_Core_Agent_Patterns_.ipynb`, `Code_agent.ipynb`, `Telecom_Support_Lab.ipynb`
  - **S3 Self-Correction**: Implementing agents that can reflect and fix their own errors
    - 📓 `Lab2_Policy_Aware_Self_Correcting_Agent.ipynb` — **Advanced production-hardened agent** featuring auto-repair (5 retries), schema mapping (synonyms + fuzzy), ambiguity handling, and a full evaluation harness with 14 red-team attacks
    - 📄 `Lab2_Documentation.md` — Detailed technical documentation on pipeline architecture, security rules, and metrics

- **`M7 RAG_Retrieval Augmented Generation`**:
  - **Context Management & LangChain & API**: LangChain implementation with memory management and conversational systems
    - `Langchain_Chat with Memory/` — Chat applications with persistent memory
  - **RAG - Book Project**: End-to-end RAG system with PDF processing and semantic search
    - Features: PDF processing, chunking strategies, embeddings, retrieval, reranking
    - **Files**: `demo.ipynb`, `fullsystem.ipynb`, supporting modules (`chunking.py`, `embeddings.py`, `retrieval.py`, `rerank.py`, `rag_api.py`)
  - **RAG PART 1 - Policy Decision Agent**: Hybrid RAG architecture for policy-based decision making
    - `RAG_PART_1_Policy_Decision_Agent_–_Hybrid_Architecture.ipynb` — Comprehensive policy decision system
    - Multiple implementation approaches and iterations
  - **RAG PART 2 - RAG Foundation**: Core RAG pipeline implementation
    - `RAG_PART_2_Hybrid_compliance_decision_engine.ipynb` — Compliance-driven RAG engine

### 🌐 Integration & Deployment

- **`Hugging Face & FastAPI`**: Production-ready deployment patterns
  - **Generate Text Using GPT2 with Gradio**:
    - `Generate_text_using_GPT2.ipynb` — Interactive notebook for GPT2 text generation
    - `main.py` — Standalone Python script
    - Gradio web interface for real-time text generation
  - **FastAPI**:
    - `app.py` — FastAPI application for serving ML models
    - `FastApi_Colab.ipynb` — Jupyter notebook for API development in Colab
    - Building scalable APIs with FastAPI to serve ML models
    - Integration with Hugging Face models

---

## 🛠️ Key Learning Outcomes & Features

- **💻 Programming Mastery**: Python fundamentals, SQL databases, and object-oriented design patterns
- **🧠 Foundational Mastery:** Solid understanding of ML/DL architectures from basic regression to complex Transformers.
- **🎨 Generative Excellence:** Hands-on experience with VAEs, GANs, and Diffusion models, including LoRA fine-tuning for specialized image generation.
- **🎯 Advanced Prompting:** Mastering Microsoft Phi-3.5 and modern LLMs for reasoning-driven workflows, Chain-of-Thought prompting, and structured function calling.
- **🤖 Agentic Autonomy:** Designing autonomous agents capable of independent planning, multi-step execution, complex tool interaction, and self-correction.
- **🛡️ Security & Governance:** Implementing deterministic policy enforcement and AST-based Python sandboxes to prevent data leaks and unauthorized system access.
- **🛠️ Self-Correction & Reliability:** Advanced reflection loops for autonomous code repair, schema mapping (synonyms/fuzzy matching) for robust natural language interaction, and comprehensive evaluation harnesses (red-teaming) for production-grade verification.
- **🏗️ RAG Architectures:** Understanding and implementing multi-part Retrieval-Augmented Generation systems with policy decision engines and semantic search.
- **🌐 Production Readiness:** Building scalable APIs with FastAPI, deploying models with Gradio web interfaces, and implementing real-world deployment patterns (Docker, Hugging Face).
- **📊 End-to-End Systems:** Building complete systems from data ingestion through inference, including RAG workflows for compliance and policy-driven decisions.

---

## 🧰 Tech Stack

### 🐍 Core Programming
- **Languages**: Python 3.8+, SQL
- **Environments**: Jupyter Notebooks, Google Colab, Anaconda

### 🤖 Machine Learning & Deep Learning
- **Frameworks**: `TensorFlow`, `PyTorch`, `Keras`, `JAX`
- **Model Libraries**: `Scikit-learn`, `XGBoost`, `LightGBM`

### 🎨 Generative AI
- **Model Hubs**: `Hugging Face (Transformers, Diffusers, Accelerate)`, `OpenAI API`
- **Models**: `Microsoft Phi-3.5`, `GPT-2`, `Stable Diffusion`
- **Optimization**: `BitsAndBytes (4-bit Quantization)`, `LoRA (Low-Rank Adaptation)`

### 🧠 AI Agents & Reasoning
- **Frameworks**: `LangChain`
- **Utilities**: `AST (Abstract Syntax Trees)`, `Difflib (Fuzzy Matching)`, `reflection loops`

### 📊 Data & Analysis
- **Libraries**: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Plotly`
- **Databases**: `SQLite`, `Vector databases (embeddings)`, `Chromadb`

### 🚀 Deployment & Production
- **Web Frameworks**: `FastAPI`, `Uvicorn`
- **UI Frameworks**: `Gradio`
- **Containerization**: `Docker`
- **Model Serving**: `Hugging Face Spaces`, `Hugging Face Models Hub`

---

## 🚀 How to Use

As this is a comprehensive learning journey, follow this recommended path:

### 📋 Recommended Learning Path

1. **Start with Foundations** (if needed):
   - Navigate to `AI- Workshop Material/Python Material/` for Python basics
   - Navigate to `AI- Workshop Material/SQL Material/` for database fundamentals
   - These modules provide essential skills for the main curriculum

2. **Progress Through Core Modules** in order:
   - `M2 ML Fundamentals` — Build statistical and algorithmic foundations
   - `M3 Deep Learning & Transformers` — Master neural networks and modern architectures
   - `M4 Generative AI Fundamentals` — Explore generative models (VAEs, GANs, Diffusion)
   - `M5 Prompt Eng` — Learn advanced prompting and function calling
   - `M6 Agentic` — Build autonomous agents with self-correction
   - `M7 RAG_Retrieval Augmented Generation` — Implement knowledge-grounded systems

3. **Learn Deployment Patterns**:
   - Explore `Hugging Face & FastAPI/` for web UI and API deployment techniques
   - Apply these patterns to containerize and deploy your own models

### 🛠️ Setup Instructions

- **Setup Environment:**
  - General environment with `jupyter`, `torch`/`tensorflow`, and `transformers` is recommended
  - Individual folders may have specific requirements in their respective documentation
  - Run notebooks sequentially within each module to build understanding progressively

- **Run Labs:**
  - Open the `.ipynb` files to see implementations and complete practical assignments
  - Pay attention to dependency files (e.g., CSV data files, configuration files) in each folder
  - Use provided Python scripts (`.py` files) to understand production-grade implementations

---

## 📚 Module Quick Reference

| Folder | Purpose | Best For | Key Files |
| :--- | :--- | :--- | :--- |
| `AI- Workshop Material/Python Material` | Python programming foundations | Beginners, skill refresher | `.ipynb` worksheets per topic |
| `AI- Workshop Material/SQL Material` | Database and SQL essentials | Data manipulation, querying | SQL exercises and scenarios |
| `M2 ML Fundamentals/ML labs` | Classical ML algorithms | Baseline ML understanding | `1 ML fundamentals.ipynb`, exercises |
| `M3 Deep Learning & Transformers` | Neural networks to transformers | Deep learning concepts | Session notebooks on CNNs, RNNs, LSTMs |
| `M4 Generative AI Fundamentals` | VAEs, GANs, Diffusion, LoRA | Generative model design | Multiple VAE implementations, GAN/Diffusion |
| `M5 Prompt Eng` | LLM prompting and function calling | Advanced prompt engineering | Phi-3.5 notebooks, function calling patterns |
| `M6 Agentic` | Agent design and self-correction | Agent building, production patterns | Lab1 & Lab2 with full documentation |
| `M7 RAG_Retrieval Augmented Generation` | Knowledge grounding and retrieval | RAG system implementation | End-to-end RAG, policy agents, book project |
| `Hugging Face & FastAPI` | Model deployment and APIs | Production deployment | Gradio + FastAPI examples, integration patterns |

---

## 🎯 Common Use Cases & Where to Find Examples

| Use Case | Navigate To | Focus On |
| :--- | :--- | :--- |
| **Build a chatbot** | `M7/Context Management & Langchain & API` | LangChain chat with memory |
| **Deploy ML model as API** | `Hugging Face & FastAPI/3) Fast API` | `app.py`, `FastApi_Colab.ipynb` |
| **Create autonomous agent** | `M6/Agentic/S1,S2_ Core Agent Patterns` | `Lab1_Safe_Sales_Insights_Agent.ipynb` |
| **Implement self-correcting agent** | `M6/Agentic/S3 Self-Correction` | `Lab2_Policy_Aware_Self_Correcting_Agent.ipynb` |
| **Build RAG system** | `M7/RAG - Book project` | `fullsystem.ipynb`, supporting modules |
| **Fine-tune for text generation** | `M4/S4 LoRA Fine-Tuning` | Stable Diffusion LoRA implementation |
| **Learn computer vision** | `M3/Session 2 - Computer Vision` | CNN notebooks and color detection |
| **Understand transformers** | `M3/Session 5 - Transformers` | Transformer architecture notebooks |
| **Learn sentiment analysis** | `M3/Session 4 NLP RNN & LSTM` | RNN/LSTM sentiment tasks |
| **Deploy with Gradio UI** | `Hugging Face & FastAPI/2) Generate text using GPT2` | GPT2 text generation demo |

---

## 📖 Acknowledgments

This journey is part of the **DEPI (Digital Egypt Pioneers Initiative)**. Special thanks to the instructors and the community for the continuous support.

---
*Stay tuned as we continue to build more intelligent agents!* 🏗️🤖
