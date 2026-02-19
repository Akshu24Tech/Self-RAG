# GATE Self-RAG: Intelligent Q&A System for GATE Exam Papers

A Self-Reflective Retrieval Augmented Generation (Self-RAG) system designed specifically for answering questions about GATE (Graduate Aptitude Test in Engineering) Discrete Mathematics and Algorithms exam papers.

## 🎯 Powered by Your Choice

Choose your AI provider:
- **Google Gemini** (Recommended) - Free tier, 2x cheaper, faster
- **OpenAI GPT** - Premium quality, established ecosystem

See [GEMINI_VS_OPENAI.md](GEMINI_VS_OPENAI.md) for detailed comparison.

## 🎯 What is Self-RAG?

Self-RAG is an advanced RAG system that:
- **Decides** whether to retrieve documents or answer directly
- **Filters** retrieved documents for relevance
- **Verifies** that answers are supported by source material
- **Revises** answers that lack proper grounding
- **Rewrites** queries when retrieval fails
- **Validates** answer usefulness before returning

## 📁 Project Structure

```
.
├── paper/
│   ├── DA2024.pdf          # GATE DA 2024 exam paper
│   └── DA2025.pdf          # GATE DA 2025 exam paper
├── self-rag-main/
│   ├── documents/          # Original company demo documents
│   └── self_rag_step7.ipynb  # Original Self-RAG implementation
├── gate_self_rag.ipynb     # ✨ NEW: GATE-adapted Self-RAG
└── README.md               # This file
```

## 🚀 Features

### 1. **Intelligent Retrieval Decision**
   - Automatically determines if exam paper content is needed
   - Handles both general CS theory and specific exam questions

### 2. **Document Relevance Filtering**
   - Filters retrieved chunks for topic relevance
   - Focuses on questions and examples that match the query

### 3. **Answer Verification (IsSUP)**
   - Verifies answers are grounded in exam paper content
   - Provides evidence quotes from source material
   - Triggers revision if claims are unsupported

### 4. **Answer Revision Loop**
   - Automatically revises answers to be more factual
   - Grounds responses in actual exam content
   - Limits revisions to prevent infinite loops

### 5. **Usefulness Check (IsUSE)**
   - Validates that answers actually address the question
   - Triggers query rewriting if answer is off-topic

### 6. **Query Rewriting**
   - Optimizes search queries for better retrieval
   - Adds relevant CS keywords and terms
   - Preserves year references (2024, 2025)

### 7. **Visual Workflow Graph**
   - LangGraph visualization of the entire pipeline
   - Shows decision points and flow paths

## 📊 Workflow Diagram

```
START
  ↓
Decide Retrieval
  ↓
  ├─→ [No Retrieval] → Generate Direct → END
  └─→ [Need Retrieval] → Retrieve Documents
                            ↓
                         Filter Relevant
                            ↓
                            ├─→ [No Relevant] → No Answer Found → END
                            └─→ [Has Relevant] → Generate from Context
                                                    ↓
                                                 Verify Support (IsSUP)
                                                    ↓
                                                    ├─→ [Fully Supported] → Accept
                                                    └─→ [Not Supported] → Revise
                                                                            ↓
                                                                         (Loop back to IsSUP)
                                                    ↓
                                                 Check Usefulness (IsUSE)
                                                    ↓
                                                    ├─→ [Useful] → END
                                                    └─→ [Not Useful] → Rewrite Query
                                                                         ↓
                                                                      (Loop back to Retrieve)
```

## 🛠️ Setup

### Prerequisites
```bash
pip install langchain langchain-community langchain-openai langgraph faiss-cpu python-dotenv pydantic
```

### Environment Variables
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## 💻 Usage

### Basic Usage

```python
# Load the notebook
jupyter notebook gate_self_rag.ipynb

# Run all cells to initialize the system

# Ask a question
question = "Explain the Pumping Lemma for Regular Languages using an example from a 2024 exam paper."
result = app.invoke({"question": question})
print(result["answer"])
```

### Using the Helper Function

```python
# Simple query
ask_gate_question("What graph algorithms appeared in the 2025 GATE DA paper?")

# Verbose mode with metadata
ask_gate_question(
    "Explain time complexity analysis with examples from recent papers.",
    verbose=True
)
```

## 📝 Example Questions

1. **Concept with Exam Example**
   ```
   "Explain the Pumping Lemma for Regular Languages using an example from a 2024 exam paper."
   ```

2. **Topic Search**
   ```
   "What graph algorithms appeared in the 2025 GATE DA paper?"
   ```

3. **Concept Explanation**
   ```
   "Explain time complexity analysis with examples from recent papers."
   ```

4. **Specific Problem**
   ```
   "Show me a dynamic programming problem from GATE 2024 and explain the solution."
   ```

5. **Comparison**
   ```
   "Compare the difficulty of automata theory questions between 2024 and 2025 papers."
   ```

## 🎓 Key Differences from Original

| Aspect | Original (Company Demo) | GATE Version |
|--------|------------------------|--------------|
| **Documents** | Company policies, pricing | GATE exam papers |
| **Chunk Size** | 600 chars | 800 chars (longer questions) |
| **Retrieval Count** | 4 chunks | 5 chunks |
| **Prompts** | Business-focused | Educational/academic |
| **Answer Style** | Factual quotes only | Educational with examples |
| **Max Retries** | 10 | 3 (faster iteration) |
| **Rewrite Tries** | 3 | 2 (quicker fallback) |

## 🔧 Customization

### Adjust Chunk Size
```python
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for longer questions
    chunk_overlap=250
).split_documents(docs)
```

### Change Retrieval Count
```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 7}  # Retrieve more chunks
)
```

### Modify Retry Limits
```python
MAX_RETRIES = 5  # More revision attempts
MAX_REWRITE_TRIES = 3  # More query rewrites
```

## 📈 Performance Tips

1. **Add More Papers**: Include more years for better coverage
2. **Tune Chunk Size**: Larger chunks for complex problems
3. **Adjust Temperature**: Set to 0 for consistent answers
4. **Use Better Embeddings**: Try `text-embedding-3-large` for accuracy

## 🐛 Troubleshooting

### "No relevant content found"
- Try rephrasing your question
- Add year references (2024, 2025)
- Use specific CS terminology

### Slow Performance
- Reduce `k` in retriever (fewer chunks)
- Use smaller embedding model
- Decrease MAX_RETRIES

### Inaccurate Answers
- Check if papers contain relevant content
- Increase chunk overlap
- Adjust relevance filtering prompts

## 📚 Adding More Papers

```python
docs = (
    PyPDFLoader("../paper/DA2024.pdf").load()
    + PyPDFLoader("../paper/DA2025.pdf").load()
    + PyPDFLoader("../paper/DA2023.pdf").load()  # Add more
    + PyPDFLoader("../paper/DA2022.pdf").load()
)
```

## 🤝 Contributing

To adapt this for other exam types:
1. Replace PDF files with your exam papers
2. Update prompts to match your domain
3. Adjust chunk size based on question length
4. Modify retrieval count as needed

## 📄 License

This project adapts the Self-RAG pattern for educational purposes.

## 🙏 Acknowledgments

- Based on the Self-RAG paper and implementation
- Uses LangChain and LangGraph frameworks
- Powered by OpenAI embeddings and GPT models

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review example questions
3. Adjust parameters based on your needs

---

**Happy Learning! 🎓**
