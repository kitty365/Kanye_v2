# Try to understand Kanye

## Project Description

This project aims to interpret the thoughts, lyrics, tweets, and emotional patterns of Kanye West using Retrieval-Augmented Generation (RAG).  
Through semantic search and clustering of various data sources (tweets, lyrics, biography, research), the goal is to philosophically explore Kanye's inner world.

I chose this project because I find Kanye an interesting, controversal cultural figure with complex public perception and layered expression across media.

### Name & URL

| Name          | URL |
|---------------|-----|
|[Github Repository]()| Github Repository|
|[Hugging Face]()| Hugging Face |

## Data Sources
| Data Source   | Description |
|---------------|-----|
|[brittanica](https://www.britannica.com/biography/Kanye-West)|Biographic Overview|
|[kaggle](https://www.kaggle.com/datasets/parthjuneja/kanye-west-tweets)| Tweets|
|[github](https://github.com/babyakja/GA_capstone_project)|Lyrics|
|[Wiley Online Library](https://onlinelibrary.wiley.com/doi/epdf/10.1002/aps.1768)| Research Article|
|[The World According To Kanye West](https://theworldaccordingtokanye.com)|Illustrated Biography|

**Reason for choice**: This selection represents a rich, multimodal corpus covering Kanye’s artistic, emotional, social, and intellectual expressions.

## Preprocessing & Chunking

### Data Cleaning
- Removed retweets, mentions, links, emojis, and special characters from tweets
- Flattened multi-line lyrics into clean lines
- Extracted and cleaned PDF content (biography, study, TWAK book)

### Chunking Strategy
Selected `paraphrase-multilingual-MiniLM-L12-v2` from Sentence-Transformers
because it balances multilingual capabilities, semantic quality, and performance.

| Property           | Value |
|--------------------|-------|
| Model              | `paraphrase-multilingual-MiniLM-L12-v2` |
| Embedding Dim      | 384   |
| Max Tokens/Chunk   | 128   |
| Justification      | Optimized for paraphrase mining, multilingual support (EN/DE), low latency |


## Semantic Embedding & Retrieval

### Embeddings
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- All chunks embedded into 384-dim vectors
- FAISS index created and saved (`IndexFlatL2`)

### Retrieval Function
- Queries encoded & top-k (default: 10) nearest chunks retrieved
- Source-tracked results for analysis
- Example query: **"Was sagt Kanye über North?"**


## RAG Implementation

### Initial Version (Basic RAG)

In the first version of the system:

- A **single user query** was embedded using a SentenceTransformer model  
- The **top 3 most similar chunks** were retrieved from the FAISS index  
- These chunks were used to construct a prompt, which was passed to the LLM (`llama3-70b-8192` by Groq)  
- The response was based solely on this narrow context  

**Limitation**: The system was highly sensitive to the exact wording of the query. If a key term was missing or phrased differently, important content could be missed.

### Improvements

#### Off-Topic Filtering via Semantic Distance

To avoid irrelevant questions (e.g., *"Why can’t humans breathe underwater?"*), the system validates each user query by checking the semantic similarity of its top-k retrieved results.

- If the **maximum distance** exceeds a defined threshold (e.g., `30.0`), the query is classified as **off-topic**
- This ensures that the LLM only answers questions with meaningful semantic grounding in the Kanye-related corpus

#### Improved Version (with Query Expansion)

To address the limited information, the system was enhanced with **semantic query expansion**:

- Each user query is **automatically expanded into three semantically related questions** using an LLM  
- All four queries (original + 3 variants) are run through the FAISS index  
- Retrieved chunks from all queries are **deduplicated and combined** into a unified context  
- This richer context is then passed to the LLM to generate an answer

| Feature                 | Before                         | After Query Expansion                 |
|------------------------|----------------------------------|----------------------------------------|
| Query input            | One phrasing                    | Original + 3 semantically similar ones |
| Retrieval scope        | Narrow                          | Broader semantic recall                |
| Answer quality         | Variable                        | More robust and well-informed          |
| Sensitivity to phrasing| High                            | Significantly reduced                  |

This improvement increased the **robustness**, **coverage**, and **depth** of the generated answers — especially for abstract or interpretive topics like religion, mental health, or family dynamics in Kanye's work.


## RAG Improvements

| Improvement           | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `Off-Topic Filtering` | Discards unrelated queries if FAISS distances exceed a set similarity threshold |
| `Query Expansion`     | Generates 3 semantically related sub-queries per input query using an LLM   |
| `Query Rewriting`     | (Planned) Reformulates vague user questions into clearer ones               |
| `Result Reranking`    | (Planned) Reranks top chunks based on secondary relevance criteria          |





## Choice of LLM

| Name                | Link |
|---------------------|------|
| LLaMA 3 (70B, 8192) | [Groq Platform](https://groq.com) |

## Test Method

Test data consisted of 10 carefully designed user questions related to Kanye West's themes, emotions, and artistic intentions.  
The evaluation focused on answer depth, contextual fit, and semantic precision.

## Results

