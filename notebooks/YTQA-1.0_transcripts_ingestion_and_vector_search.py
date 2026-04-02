# Databricks notebook source
# MAGIC %md
# MAGIC # YTQA-1.0: YouTube Transcripts Ingestion + Vector Search
# MAGIC 
# MAGIC This notebook combines the key ideas from the course notebooks:
# MAGIC - **2.3 Chunking Strategies** (create clean chunks for retrieval)
# MAGIC - **2.4 Embeddings & Vector Search** (create and sync a Vector Search index)
# MAGIC 
# MAGIC ## What this notebook does
# MAGIC 1. Takes a list of YouTube URLs (or video IDs)
# MAGIC 2. Downloads transcript text using `youtube-transcript-api`
# MAGIC 3. Stores transcripts in a Delta table: `youtube_videos`
# MAGIC 4. Creates fixed-length overlapping chunks into: `youtube_chunks_table`
# MAGIC 5. Creates (or retrieves) a Vector Search index: `youtube_index`
# MAGIC 6. Runs a sample similarity search query

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from youtube_qa.config import get_env, load_config
from youtube_qa.data_processor import DataProcessor
from youtube_qa.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

logger.info(f"Env: {env}")
logger.info(f"Catalog: {cfg.catalog}")
logger.info(f"Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Provide YouTube URLs
# MAGIC 
# MAGIC Provide a list of YouTube URLs (or 11-char video IDs).
# MAGIC 
# MAGIC Notes:
# MAGIC - Some videos do not have transcripts available.
# MAGIC - Some videos block transcript access.

# COMMAND ----------

youtube_urls: list[str] = [
    "https://www.youtube.com/watch?v=Us-w_j_4qGo"
    # "https://www.youtube.com/watch?v=VIDEO_ID",
    # "https://youtu.be/VIDEO_ID",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Download transcripts and write tables
# MAGIC 
# MAGIC This writes:
# MAGIC - `catalog.schema.youtube_videos`
# MAGIC - `catalog.schema.youtube_chunks_table` (with Change Data Feed enabled)

# COMMAND ----------
ws_username = dbutils.secrets.get(scope="hosseinh-secrets", key="ws-username")
ws_password = dbutils.secrets.get(scope="hosseinh-secrets", key="ws-password")

# COMMAND ----------

processor = DataProcessor(
    spark=spark,
    config=cfg,
    proxy_username=ws_username,
    proxy_password=ws_password,
)
processor.process_and_save(youtube_urls)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Inspect the stored transcripts

# COMMAND ----------

videos_table = f"{cfg.catalog}.{cfg.schema}.youtube_videos"
chunks_table = f"{cfg.catalog}.{cfg.schema}.youtube_chunks_table"

logger.info(f"Videos table: {videos_table}")
logger.info(f"Chunks table: {chunks_table}")

spark.table(videos_table).orderBy("ingest_ts", ascending=False).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Inspect chunks

# COMMAND ----------

spark.table(chunks_table).orderBy("ingest_ts", ascending=False).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Create or get Vector Search index
# MAGIC 
# MAGIC The `VectorSearchManager` creates:
# MAGIC - a Vector Search endpoint (if missing)
# MAGIC - a delta-sync Vector Search index (`youtube_index`) backed by `youtube_chunks_table`

# COMMAND ----------

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
    usage_policy_id=cfg.usage_policy_id,
)

logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model: {vs_manager.embedding_model}")
logger.info(f"Index Name: {vs_manager.index_name}")

# COMMAND ----------

vs_manager.create_endpoint_if_not_exists()
index = vs_manager.create_or_get_index()

logger.info("✓ Vector search setup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) (Triggered pipeline) Sync the index
# MAGIC 
# MAGIC The index is created with a `TRIGGERED` pipeline.
# MAGIC Run a sync after writing new chunks.

# COMMAND ----------

index.sync()
logger.info("✓ Index sync triggered")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Similarity search
# MAGIC 
# MAGIC Query the index with semantic similarity.

# COMMAND ----------

query = "Summarize the main points of the video."

results = index.similarity_search(
    query_text=query,
    columns=["id", "text", "video_id"],
    num_results=5,
)

results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Optional: parse results into a list of dicts

# COMMAND ----------

def parse_vector_search_results(results: dict) -> list[dict]:
    """Convert Vector Search results to a list of dicts."""
    columns = [c["name"] for c in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row, strict=False)) for row in data_array]


parsed = parse_vector_search_results(results)
parsed[:3]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC - If you add more videos, rerun the ingestion cell and then run `index.sync()` again.
# MAGIC - You can later extend this with metadata (channel, title) or timestamp-aware chunking.
