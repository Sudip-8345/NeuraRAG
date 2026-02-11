# Setup and Usage Guide

## Quick Start Guide

### 1. Prerequisites

Ensure you have:
- Python 3.8 or higher
- pip (Python package manager)
- A Groq API key from [https://console.groq.com/keys](https://console.groq.com/keys)

### 2. Installation Steps

```bash
# Clone the repository
git clone https://github.com/Sudip-8345/NeuraRAG.git
cd NeuraRAG

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Verify Installation

Run the test script to verify everything is set up correctly:

```bash
python test_system.py
```

You should see:
```
Tests passed: 4/4
🎉 All tests PASSED!
```

## Usage Examples

### Example 1: Index Documents

Before asking questions, you need to index the policy documents:

```bash
python main.py index
```

Expected output:
```
================================================================================
INDEXING DOCUMENTS
================================================================================

Initializing RAG System...
Loading embedding model: all-MiniLM-L6-v2...
Processing documents from ./data...
Processed 24 chunks from documents.
Generating embeddings for 24 chunks...
Adding 24 chunks to vector store...
Successfully added 24 chunks to the vector store.

✅ Indexing complete!
```

### Example 2: Ask a Single Question

```bash
python main.py ask "Can I return a product after 30 days?"
```

Expected output:
```
================================================================================
ANSWER:
================================================================================
According to our Refund Policy, physical products can be returned within 
30 days of delivery. However, the products must be in original condition 
with all tags and packaging intact. Electronics must be unopened and 
sealed in original packaging. Custom or personalized items are not 
eligible for refunds.

================================================================================
Sources: refund_policy.md
================================================================================
```

### Example 3: Ask with Verbose Output

To see the retrieval process:

```bash
python main.py ask "What is the cost of overnight shipping?" --verbose
```

Expected output includes:
- The question
- Retrieved chunks with relevance scores
- Generated answer
- Metadata about sources

### Example 4: Interactive Mode

For a conversation-like experience:

```bash
python main.py interactive
```

Example session:
```
================================================================================
RAG SYSTEM - INTERACTIVE MODE
================================================================================

Ask questions about our policies!
Type 'quit' or 'exit' to stop.

Your question: Can I cancel a custom order?

--------------------------------------------------------------------------------
ANSWER:
--------------------------------------------------------------------------------
According to our Cancellation Policy, custom or made-to-order items have a 
2-hour cancellation window. After 2 hours, custom orders cannot be cancelled. 
A 25% restocking fee may apply to cancelled custom orders within the window.

--------------------------------------------------------------------------------
METADATA:
  • Model: llama-3.1-70b-versatile
  • Prompt Version: v2
  • Chunks Retrieved: 3
  • Sources: cancellation_policy.md
--------------------------------------------------------------------------------

Your question: quit

Thank you for using the RAG system!
```

### Example 5: Run Evaluation

To evaluate the system with the test dataset:

```bash
python main.py evaluate
```

This will:
1. Load the evaluation dataset (8 questions)
2. Process each question through the RAG pipeline
3. Score responses for accuracy, hallucination prevention, and clarity
4. Display a summary report

Save results to JSON:
```bash
python main.py evaluate --save evaluation_results.json
```

### Example 6: View Prompts

To see the prompt templates and their iterations:

```bash
python main.py prompts
```

## Common Use Cases

### Use Case 1: Customer Service Bot

Deploy this system as a backend for a customer service chatbot:

```python
from src.rag_system import RAGSystem

# Initialize
rag = RAGSystem()

# In your chatbot handler:
def handle_customer_question(question: str) -> str:
    result = rag.answer_question(question)
    return result['answer']
```

### Use Case 2: Policy Search Engine

Use for internal employees to quickly find policy information:

```python
from src.rag_system import RAGSystem

rag = RAGSystem()

# Search for specific policy info
questions = [
    "What is the refund process for defective items?",
    "How long does standard shipping take?",
    "Can I cancel a subscription?"
]

for q in questions:
    result = rag.answer_question(q, verbose=True)
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")
```

### Use Case 3: Testing Different Prompts

Compare prompt versions:

```python
from src.rag_system import RAGSystem

rag = RAGSystem()
question = "Do you offer warranty on electronics?"

# Test v1 prompt
result_v1 = rag.answer_question(question, use_improved_prompt=False)
print("V1:", result_v1['answer'])

# Test v2 prompt
result_v2 = rag.answer_question(question, use_improved_prompt=True)
print("V2:", result_v2['answer'])
```

## Troubleshooting

### Issue: "GROQ_API_KEY not found"

**Solution:** 
1. Ensure `.env` file exists in the project root
2. Add `GROQ_API_KEY=your_actual_key_here` to the file
3. Get your API key from https://console.groq.com/keys

### Issue: "No documents indexed yet"

**Solution:** Run `python main.py index` before asking questions

### Issue: Embedding model download fails

**Solution:** 
- Check your internet connection
- The first run downloads the sentence-transformers model (~90MB)
- Subsequent runs use the cached model

### Issue: Poor answer quality

**Solutions:**
1. Ensure documents are indexed: `python main.py index --force`
2. Check if the question is within the scope of the policies
3. Try rephrasing the question to be more specific
4. Verify the improved prompt is being used (default)

## Advanced Configuration

### Customize Chunk Size

Edit the RAGSystem initialization:

```python
rag = RAGSystem(
    chunk_size=256,      # Smaller chunks for more precise retrieval
    chunk_overlap=25,    # Adjust overlap
    top_k=5              # Retrieve more chunks
)
```

### Use a Different Embedding Model

Edit `src/retrieval.py`:

```python
self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
```

### Use a Different LLM Model

Edit `src/generation.py`:

```python
self.model = "llama-3.1-8b-instant"  # Faster, less capable
# or
self.model = "mixtral-8x7b-32768"    # Alternative model
```

## Performance Considerations

### Indexing Time

- ~24 chunks: < 1 minute
- Depends on embedding model download (first time only)

### Query Time

- Typical query: 1-3 seconds
- Breakdown:
  - Embedding generation: ~100-200ms
  - Vector search: ~50-100ms
  - LLM generation: ~1-2 seconds

### Memory Usage

- Embedding model: ~100MB
- Vector database: ~10MB for 24 chunks
- Total RAM: ~500MB typical

## Best Practices

1. **Index Once, Query Many**: Only reindex when documents change
2. **Use Improved Prompt**: The v2 prompt prevents hallucinations
3. **Monitor Relevance Scores**: Low scores (<0.5) may indicate question is off-topic
4. **Batch Evaluation**: Use the evaluation dataset to track system performance over time
5. **Log Queries**: Keep track of common questions to improve documentation

## Next Steps

1. Add more policy documents to `./data/`
2. Run evaluation after changes: `python main.py evaluate`
3. Customize prompts for your specific use case
4. Integrate into your application via the Python API
5. Monitor and iterate based on user feedback
