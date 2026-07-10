# Self-RAG Concurrent Load Test Analysis Report

**Date:** 2026-07-10  
**Test Configuration:**
- **Concurrency Levels Tested:** 1, 5, 10, 20, 30
- **Base LLM:** Groq `llama-3.3-70b-versatile` (temporary swap to avoid Gemini API limits)
- **Vector Database:** FAISS (constructed using Gemini `models/gemini-embedding-001`)

---

## 1. Latency & Throughput Under Concurrency

The load test fired batches of concurrent queries through the Self-RAG pipeline. Here is the degradation of performance:

| Concurrency | Successful Queries | Failed Queries | Min Latency (s) | Avg Latency (s) | P95 Latency (s) | Max Latency (s) | Throughput (QPS) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 0 | 1.66 | 1.66 | 1.66 | 1.66 | 0.60 |
| **5** | 5 | 0 | 1.50 | 1.74 | 2.12 | 2.12 | 2.31 |
| **10** | 10 | 0 | 0.66 | 1.85 | 2.62 | 2.62 | 3.77 |
| **20** | 3 | 17 | 1.25 | 3.63 | 5.84 | 5.84 | 0.47 |
| **30** | 2 | 28 | 7.09 | 8.59 | 10.09 | 10.09 | 0.20 |

### Latency Degradation Key Findings:
- Under **low concurrency (1-10)**, the system performs well. Average latency stays stable below 2.0 seconds, and throughput scales up to 3.77 QPS.
- Under **high concurrency (20-30)**, throughput collapses and latency spikes dramatically. At 30 concurrent requests, average latency degrades by **517%** (from 1.66s to 8.59s).

---

## 2. Grounding (IsSUP) Accuracy Under Load

Under the test configuration, the grounding check was skipped because **100% of test queries were routed to the `generate_direct` node** (meaning `need_retrieval` was evaluated as `False`).

### Why did this happen?
The decision node uses `should_retrieve` logic to classify queries. Because we swapped the LLM to Groq `llama-3.3-70b-versatile` (a highly capable 70B parameter model), the LLM classified standard quantum computing queries ("What is quantum superposition?", "How do qubits differ from classical bits?") as general knowledge that did not require domain-specific PDF retrieval. As a result, the pipeline successfully bypassed FAISS retrieval and answered from pre-trained weights.
- While this optimizes execution path and latency, it means the grounding accuracy of the retrieval-augmented generation was not stressed.

---

## 3. Bottleneck Analysis: Rate Limit Saturation (HTTP 429)

### The Real Failure Found:
At 20 and 30 concurrent requests, the system experienced a massive failure rate (85%+ failed requests). The exact error logged for the failures was:

```json
"error": "Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` ... on requests per minute (RPM): Limit 30, Used 30, Requested 1. Please try again in 2s.'}}"
```

### Root Cause:
1. **Chat Session Multiplexing**: Every request going through the `need_retrieval=False` path makes **2 synchronous LLM requests** (1 for retrieval routing decision, 1 for direct generation).
2. **API Limit Contention**: 30 concurrent queries instantly spawn **60 LLM API requests** in parallel.
3. **Free Tier Cap**: Groq's RPM limit is capped at **30 RPM**.
4. **Latency Spikes**: The few requests that succeeded under concurrency 30 had to undergo multiple exponential backoff retries managed by the LLM client, resulting in the latency jumping to 10.09s.

### Mitigation Strategies:
- **Rate Limit Queueing**: Implement a client-side rate limiter or semaphore (e.g., using `asyncio.Semaphore(10)`) to ensure the concurrent load never exceeds the API limit (e.g., maximum 10 requests processed at once).
- **Consolidation**: Combine the retrieval decision and initial response into a single LLM call where possible (e.g., using structured output tool calling).
