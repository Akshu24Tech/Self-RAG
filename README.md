# Universal Self-RAG System

A Self-Reflective Retrieval Augmented Generation (Self-RAG) system that works with **ANY PDF documents** using Google Gemini.

## 🎯 What is Self-RAG?

Self-RAG is an advanced RAG system that:
- ✅ **Decides** whether to retrieve documents or answer directly
- ✅ **Filters** retrieved documents for relevance
- ✅ **Verifies** that answers are supported by source material
- ✅ **Revises** answers that lack proper grounding
- ✅ **Rewrites** queries when retrieval fails
- ✅ **Validates** answer usefulness before returning

## 🚀 Quick Start (3 Steps)

### Step 1: Get FREE Gemini API Key

1. Go to: **https://makersuite.google.com/app/apikey**
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### Step 2: Setup

```bash
# Clone or download this folder
cd universal-self-rag

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### Step 3: Add Your Documents

```bash
# Create documents folder
mkdir documents

# Add your PDF files
# Copy any PDF files into the documents/ folder
```

## 💻 Usage

### Basic Usage

```python
from self_rag import SelfRAG

# Initialize with your documents
rag = SelfRAG(documents_folder="documents")

# Ask questions
rag.ask("What is the main topic of the documents?")
rag.ask("Summarize the key points.")
rag.ask("Explain the concept mentioned in the documents.")
```

## 📊 Workflow Diagram

<img width="718" height="853" alt="8236bb34-c994-49d7-9f3a-302a98289ba6" src="https://github.com/user-attachments/assets/6e34dd67-7ccc-40c0-ba9f-0cedec0d396d" />

### Verbose Mode

```python
# Show detailed metadata
rag.ask("Your question here", verbose=True)
```

Output:
```
================================================================================
Q: Your question here
--------------------------------------------------------------------------------
A: The detailed answer based on your documents...
--------------------------------------------------------------------------------
METADATA:
  Retrieval: True
  Support: fully_supported
  Useful: useful
  Evidence: ['quote 1', 'quote 2', 'quote 3']
================================================================================
```

### Custom Configuration

```python
# Customize chunk size and overlap
rag = SelfRAG(
    documents_folder="my_docs",
    chunk_size=1000,      # Larger chunks for longer content
    chunk_overlap=250     # More overlap for better context
)
```

## 📁 Project Structure

```
universal-self-rag/
├── self_rag.py           # Main Self-RAG implementation
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
├── .env                 # Your API key (create this)
├── README.md            # This file
└── documents/           # Put your PDF files here
    ├── document1.pdf
    ├── document2.pdf
    └── ...
```

## 🎓 Use Cases

### Academic Research
```python
# Add research papers to documents/
rag = SelfRAG(documents_folder="documents")
rag.ask("What are the main findings of the research?")
rag.ask("Compare the methodologies used.")
```

### Business Documents
```python
# Add company policies, reports, etc.
rag = SelfRAG(documents_folder="documents")
rag.ask("What is the refund policy?")
rag.ask("Summarize Q4 financial results.")
```

### Legal Documents
```python
# Add contracts, agreements, etc.
rag = SelfRAG(documents_folder="documents")
rag.ask("What are the termination clauses?")
rag.ask("Explain the liability terms.")
```

### Educational Content
```python
# Add textbooks, lecture notes, etc.
rag = SelfRAG(documents_folder="documents")
rag.ask("Explain the concept of X.")
rag.ask("What are the key formulas?")
```

## 🔧 How It Works

### Self-RAG Workflow

```
Question
   ↓
1. Decide Retrieval → Need documents?
   ↓
2. Retrieve → Get relevant chunks
   ↓
3. Filter Relevance → Keep only relevant docs
   ↓
4. Generate Answer → Create response
   ↓
5. Verify Support (IsSUP) → Is answer grounded?
   ↓
6. Revise (if needed) → Fix unsupported claims
   ↓
7. Check Usefulness (IsUSE) → Does it answer the question?
   ↓
8. Rewrite Query (if needed) → Try better search
   ↓
Final Answer
```

### Key Features

**1. Intelligent Retrieval Decision**
- Automatically determines if documents are needed
- Handles both general knowledge and document-specific questions

**2. Document Relevance Filtering**
- Filters retrieved chunks for topic relevance
- Focuses on content that matches the query

**3. Answer Verification (IsSUP)**
- Verifies answers are grounded in source material
- Provides evidence quotes
- Triggers revision if unsupported

**4. Answer Revision Loop**
- Automatically revises answers to be more factual
- Grounds responses in actual document content
- Limits revisions to prevent infinite loops (max 3)

**5. Usefulness Check (IsUSE)**
- Validates that answers actually address the question
- Triggers query rewriting if answer is off-topic

**6. Query Rewriting**
- Optimizes search queries for better retrieval
- Adds relevant keywords
- Removes filler words

## 🎯 Why Gemini?

| Feature | Gemini 1.5 Flash | Benefit |
|---------|------------------|---------|
| **Free Tier** | 15 requests/min | Perfect for testing & learning |
| **Speed** | Very Fast | Quick responses |
| **Cost** | $0.075/1M tokens | 2x cheaper than GPT-4o-mini |
| **Context** | 1M tokens | Handle many documents |
| **Quality** | Excellent | High-quality answers |

## 📊 Parameters

### Chunk Size
- **Default**: 800 characters
- **Small (400-600)**: For short, precise content
- **Medium (800-1000)**: For general documents
- **Large (1200-1500)**: For long-form content

### Chunk Overlap
- **Default**: 200 characters
- **Low (100-150)**: Less redundancy
- **Medium (200-250)**: Balanced
- **High (300-400)**: Better context preservation

### Retrieval Count (k)
- **Default**: 5 chunks
- **Few (3-4)**: Precise answers
- **Medium (5-7)**: Comprehensive answers
- **Many (8-10)**: Research/exploration

## ⚡ Concurrent Load Testing

A concurrent load-testing tool is included to stress-test the Self-RAG pipeline under concurrent requests and analyze latency degradation, grounding accuracy, and backpressure.

### Running the Load Test

1. **Get a Groq API Key**: Since concurrent testing easily saturates Gemini's free tier limits, the test suite supports swapping to a faster, high-limit LLM like Groq (`llama-3.3-70b-versatile`). Get a key from [console.groq.com](https://console.groq.com).
2. **Configure your `.env`**:
   ```bash
   GROQ_API_KEY=gsk_your_key_here
   ```
3. **Run the test script**:
   ```bash
   python load_test.py
   ```

The script will test escalating concurrency levels: `1 → 5 → 10 → 20 → 30` simultaneous queries, output a performance metrics table, and save raw data to `load_test_results.json`.

Detailed findings, metrics, and bottleneck analysis are documented in the **[Load Test Report](docs/load_test_report.md)**.

## 🐛 Troubleshooting

### Error: "GOOGLE_API_KEY not found"
**Solution**: Create `.env` file with your API key:
```bash
GOOGLE_API_KEY=your-actual-key-here
```

### Error: "No PDF files found"
**Solution**: Add PDF files to the `documents/` folder:
```bash
mkdir documents
# Copy your PDF files into documents/
```

### Error: "No module named 'langchain_google_genai'"
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Slow Performance
**Solution**: Reduce retrieval count:
```python
# Edit self_rag.py, line with search_kwargs
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

### "No relevant content found"
**Solutions**:
1. Rephrase your question with more specific terms
2. Check if your documents actually contain relevant information
3. Try increasing retrieval count (k=7 or k=10)

### Rate Limit Exceeded
**Solution**: Gemini free tier allows 15 requests/minute. Wait a minute or upgrade to paid tier.

## 🔒 Privacy & Security

- ✅ Your documents are processed locally
- ✅ Only embeddings and queries are sent to Gemini API
- ✅ Google doesn't train on your data (by default)
- ✅ API key is stored in `.env` (not committed to git)

## 📈 Performance Tips

1. **Optimize Chunk Size**: Match your document type
2. **Adjust Retrieval Count**: More chunks = more context but slower
3. **Use Specific Questions**: Better questions = better answers
4. **Organize Documents**: Group related documents together

## 🎓 Examples

### Example 1: Research Papers
```python
from self_rag import SelfRAG

rag = SelfRAG(documents_folder="research_papers")

# Ask about methodology
rag.ask("What methodology was used in the study?")

# Compare findings
rag.ask("Compare the results across different papers.")

# Get specific details
rag.ask("What were the limitations mentioned?", verbose=True)
```

### Example 2: Company Documents
```python
from self_rag import SelfRAG

rag = SelfRAG(documents_folder="company_docs")

# Policy questions
rag.ask("What is the vacation policy?")

# Financial questions
rag.ask("What was the revenue growth in Q3?")

# Product questions
rag.ask("What are the key features of Product X?")
```

### Example 3: Educational Content
```python
from self_rag import SelfRAG

rag = SelfRAG(documents_folder="textbooks")

# Concept explanation
rag.ask("Explain the concept of recursion with examples.")

# Problem solving
rag.ask("How do I solve this type of problem?")

# Summary
rag.ask("Summarize Chapter 5.", verbose=True)
```

## 🚀 Advanced Usage

### Multiple Document Sets

```python
# Different RAG instances for different document sets
research_rag = SelfRAG(documents_folder="research")
business_rag = SelfRAG(documents_folder="business")
legal_rag = SelfRAG(documents_folder="legal")

# Ask domain-specific questions
research_rag.ask("What are the findings?")
business_rag.ask("What is the policy?")
legal_rag.ask("What are the terms?")
```

### Batch Processing

```python
from self_rag import SelfRAG

rag = SelfRAG(documents_folder="documents")

questions = [
    "What is the main topic?",
    "Who are the key stakeholders?",
    "What are the recommendations?",
]

for q in questions:
    rag.ask(q)
```

### Custom Prompts

Edit `self_rag.py` to customize prompts for your specific use case. Look for `ChatPromptTemplate.from_messages()` calls.

## 📝 License

This is a universal implementation of Self-RAG. Use it for any purpose.

## 🙏 Acknowledgments

- Based on the Self-RAG paper and methodology
- Uses LangChain and LangGraph frameworks
- Powered by Google Gemini

**Ready to use! Add your PDF documents and start asking questions! 🚀**



