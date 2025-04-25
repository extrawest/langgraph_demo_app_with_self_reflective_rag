# 🧠 Self-Reflective RAG (Retrieval Augmented Generation)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Maintainer](https://img.shields.io/static/v1?label=Yevhen%20Ruban&message=Maintainer&color=red)](mailto:yevhen.ruban@extrawest.com)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub release](https://img.shields.io/badge/release-v1.0.0-blue)

A self-reflective and adaptive RAG system that dynamically routes queries between web search and vector retrieval, assesses document relevance, checks for hallucinations, and ensures answer quality using a graph-based flow architecture.

## 📋 Features

### Dynamic Query Routing
- 🔄 Intelligently routes queries between vectorstore and web search
- 🧠 Uses LLM to determine the best source for each query
- 🌐 Falls back to web search for queries outside the knowledge base

### Contextual Document Assessment
- 📑 Grades document relevance before using them in answers
- 🔍 Filters out irrelevant documents to improve context quality
- 🔄 Can transform queries when relevant documents aren't found

### Self-Critical Evaluation
- 🕵️ Checks generated answers for hallucinations
- ⚖️ Verifies that answers are grounded in retrieved documents
- 🎯 Confirms that answers address the original questions

### Adaptive Response System
- 🔁 Regenerates answers when quality issues are detected
- 🔄 Transforms queries when retrieval fails to find relevant info
- 📈 Creates a feedback loop to improve response quality

## 🏗️ Architecture

The system is built with LangGraph and follows a workflow pattern:

![Screenshot from 2025-04-25 12-26-07](https://github.com/user-attachments/assets/4ef149e0-dc3a-40e7-a07b-5abeef4bbc6e)


## 🛠️ Requirements

- Python 3.9+
- OpenAI API key
- Tavily API key
- LangChain and LangGraph libraries

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

## 🚀 Usage

```python
import asyncio
from langgraph_adaptive_rag_simple import AdaptiveRAG, main

# Run the example
asyncio.run(main())

# Or use the system for your own queries
async def custom_query():
    rag_system = AdaptiveRAG()
    await rag_system.initialize()
    
    result = await rag_system.process_query("What are the pros and cons of RAG systems?")
    print(result)

asyncio.run(custom_query())
```

## 📊 Example Output

```
===== PROCESSING: 'What player are the Bears expected to draft first in the 2024 NFL draft?' =====

---ROUTE QUESTION---
---ROUTE QUESTION TO WEB SEARCH---
---WEB SEARCH---
Node: 'web_search'
---
---GENERATE---
---CHECK HALLUCINATIONS---
---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
---GRADE GENERATION vs QUESTION---
---DECISION: GENERATION ADDRESSES QUESTION---
Node: 'generate'
---

===== ANSWER =====
The Chicago Bears are expected to draft QB Caleb Williams first in the 2024 NFL draft. Williams is seen as the savior at quarterback that the Bears have always needed, with exceptional arm talent, mobility, and leadership skills.
```

## 🏆 Key Benefits

- **Accuracy**: Self-critical evaluation prevents hallucinations
- **Flexibility**: Handles both known topics and up-to-date information
- **Adaptability**: Transforms approaches based on intermediate results
- **Scalability**: Modular design allows easy extension with new components
