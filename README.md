# KanyeRAG

## Project Description

This project aims to interpret the thoughts, lyrics, tweets and emotional patterns of Kanye West using a Retrieval Augmented Generation system
By applying semantic search and clustering across diverse sources such as tweets, lyrics, biographical texts and academic research, the system seeks to offer a philosophical perspective on Kanye’s inner world

I chose this project because I find Kanye an interesting, controversal cultural figure with complex public perception and layered expression across media.

### Name & URL

|    URL       | Name |
|---------------|-----|
|[Github Repository](https://github.com/kitty365/Kanye_v2)| Github Repository|
|[Hugging Face](https://huggingface.co/spaces/kitty365/kanye?logs=container)| Hugging Face |

## Data Sources
| Data Source   | Description |
|---------------|-----|
|[brittanica](https://www.britannica.com/biography/Kanye-West)|Biographic Overview|
|[kaggle](https://www.kaggle.com/datasets/parthjuneja/kanye-west-tweets)| Tweets|
|[github](https://github.com/babyakja/GA_capstone_project)|Lyrics|
|[Wiley Online Library](https://onlinelibrary.wiley.com/doi/epdf/10.1002/aps.1768)| Research Article|
|[The World According To Kanye West](https://theworldaccordingtokanye.com)|Illustrated Biography|

**Reason for choice**: This collection represents a basic but diverse selection covering Kanye’s artistic, emotional, social, and intellectual expressions.

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

I initially used OpenAI’s text-embedding-3-small model for embedding. While the results were good, it was too expensive and not efficient for this setup. I replaced it with MiniLM from SentenceTransformers, which is free, fast, and works well with FAISS. This combination provides a stable and affordable solution for local use.

>OPENAI Embedding:
<pre> ```python load_dotenv() key = os.getenv("OPENAI_API_KEY") # PDFs pdf_paths = { "biography": "Input/Biography.pdf", "psychoanalysis": "Input/Psychoanalytics.pdf", "twak": "Input/TWAK.pdf" } # CSVs lyrics_path = "Input/LyricsWest.csv" tweets_path = "Input/KanyeTweets.csv" # PDF's und CSV's laden def load_pdf_text(file_path): doc = fitz.open(file_path) return "\n".join([page.get_text() for page in doc]) pdf_texts = {name: load_pdf_text(path) for name, path in pdf_paths.items()} lyrics_df = pd.read_csv(lyrics_path) tweets_df = pd.read_csv(tweets_path) ``` </pre>

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

Even if a question cannot be fully answered, for example "How is his relationship with North?", the system still provides a meaningful and context-aware response. It confirms, for instance, that North is his daughter.

## RAG Improvements

| Improvement           | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `Off-Topic Filtering` | Discards unrelated queries if FAISS distances exceed a set similarity threshold |
| `Query Expansion`     | Generates 3 semantically related sub-queries per input query using an LLM   |
| `Query Rewriting`     | (Planned) Reformulates vague user questions into clearer ones               |
| `Result Reranking`    | (Planned) Reranks top chunks based on secondary relevance criteria          |

This refinement makes the system more reliable when facing ambiguous or abstract questions and significantly enhances the user experience. It also aligns with the project's goal of offering deeper insight rather than superficial answers.

## Choice of LLM

| Name                | Link |
|---------------------|------|
| LLaMA 3 (70B, 8192) | [Groq Platform](https://groq.com) |

I chose LLaMA 3 on Groq because it is fast, powerful, and free to use. It handles longer contexts well and fits the needs of this RAG setup better than smaller or more expensive models.

## Test Method

I tested the system with around 10 questions about Kanye’s life, music, emotions, and beliefs. I checked if the answers were relevant, correct, and if off-topic questions were rejected. The testing is still very basic and could be expanded much more in the future.

## Results

This project showed how important good data preparation is for a RAG system. I noticed that cleaning and structuring data, especially tweets, has a big effect on the results. I began with OpenAI models, which performed well but were too costly and prone to hallucinations. Switching to LLaMA 3 on Groq significantly improved speed and stability. The project is not finished and can be improved in many ways, for example by better handling tweets or adding smarter logic, expanding the queries deeper and testing for better results. The Gradio app works well so far. It provides solid answers to Kanye-related questions and effectively blocks off-topic queries.


