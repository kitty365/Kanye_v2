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

## RAG Improvements

| Improvement                     | Description |
|-----------------------------------|-------------|
| `Query Expansion`          | Generate extra queries to expand search |
| `Query Rewriting`              | Rewrite queries to yield better result |
| `Result reranking` | Reranked the top 10 results with another model |



## Choice of LLM

| Name | Link |
|-------|---------------|
| 1     | platformhttps://ai.google.dev/gemini-api/docs/models/gemini?hl=de#gemini-2.0-flash-lite     |

(Add rows if you combine multiple models or compared their performance.)

## Test Method

Detail how you selected or generated the test data and how you evaluated the performance of the model.

## Results

| Model/Method                                                         | Accuracy | Precision | Recall |
|----------------------------------------------------------------------|----------|-----------|--------|
|Retrieved chunks with config xyz |  -    | -         | -      |
| Generated answer with config xyz  | -      | -         | -      |

## References
