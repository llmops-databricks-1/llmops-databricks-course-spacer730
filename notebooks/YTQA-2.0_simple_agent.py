# Databricks notebook source
# MAGIC %md
# MAGIC # YTQA-2.0: YouTube Q&A Agent
# MAGIC
# MAGIC This notebook showcases a YouTube agent that can manage its own
# MAGIC transcript index and then answer grounded questions about indexed videos.
# MAGIC
# MAGIC ## Agent capabilities
# MAGIC - ingest a YouTube video from a URL
# MAGIC - list indexed videos
# MAGIC - answer grounded questions from transcript chunks
# MAGIC - remove a video from the index
# MAGIC - reset the index when needed

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from youtube_qa import YouTubeTranscriptAgent
from youtube_qa.config import get_env, load_config
from youtube_qa.data_processor import DataProcessor
from youtube_qa.vector_search import VectorSearchManager

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup
# MAGIC
# MAGIC Load the project config, connect to the transcript pipeline, initialize
# MAGIC the vector index, and construct the agent.

# COMMAND ----------

ws_username = dbutils.secrets.get(scope="hosseinh-secrets", key="ws-username")
ws_password = dbutils.secrets.get(scope="hosseinh-secrets", key="ws-password")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

data_processor = DataProcessor(
    spark=spark,
    config=cfg,
    proxy_username=ws_username,
    proxy_password=ws_password,
)
vector_search_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
    usage_policy_id=cfg.usage_policy_id,
)
vector_search_manager.create_or_get_index()

youtube_qa_agent = YouTubeTranscriptAgent(
    llm_endpoint=cfg.llm_endpoint,
    vector_search_manager=vector_search_manager,
    data_processor=data_processor,
)

logger.info(f"✓ Using LLM endpoint: {cfg.llm_endpoint}")
logger.info(f"✓ Using Vector Search index: {vector_search_manager.index_name}")
logger.info("✓ YouTubeTranscriptAgent initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inspect current indexed state
# MAGIC
# MAGIC The agent can inspect which videos are currently available before you
# MAGIC ask a content question.

# COMMAND ----------

indexed_videos = youtube_qa_agent.execute_tool("list_indexed_videos", {})
logger.info("Currently indexed videos:")
logger.info(indexed_videos)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Add a video through the agent
# MAGIC
# MAGIC If the user provides a YouTube URL, the agent can ingest the transcript,
# MAGIC chunk it, sync the index, and wait until the index is queryable.

# COMMAND ----------

ingest_request = (
    "I want to chat with you about this YouTube video: "
    "https://www.youtube.com/watch?v=3Rc4MlMJMNU"
)
ingest_response = youtube_qa_agent.chat(ingest_request)

logger.info("Ingest response:")
logger.info(ingest_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ask a grounded question
# MAGIC
# MAGIC After ingestion, the agent can answer questions from the transcript index.

# COMMAND ----------

question_1 = "What study system does the speaker use instead of traditional notes?"
answer_1 = youtube_qa_agent.chat(question_1)

logger.info(f"Q: {question_1}")
logger.info(f"A: {answer_1}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ask a follow-up question
# MAGIC
# MAGIC The agent keeps chat history, so follow-up questions can build on prior turns.

# COMMAND ----------

question_2 = "Why does the speaker think this helps them remember what they study better?"
answer_2 = youtube_qa_agent.chat(question_2)

logger.info(f"Q: {question_2}")
logger.info(f"A: {answer_2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Use the agent for index management
# MAGIC
# MAGIC The same agent can manage the transcript index through natural language.

# COMMAND ----------

list_response = youtube_qa_agent.chat("What videos are currently indexed?")
logger.info(f"Agent: {list_response}")

# COMMAND ----------

remove_response = youtube_qa_agent.chat(
    "Remove this video from your index: https://www.youtube.com/watch?v=IL1t_S4mfyc"
)
logger.info(f"Agent: {remove_response}")

# COMMAND ----------

post_remove_response = youtube_qa_agent.chat("What videos are currently indexed now?")
logger.info(f"Agent: {post_remove_response}")

# COMMAND ----------

reset_response = youtube_qa_agent.chat("Reset your YouTube transcript index.")
logger.info(f"Agent: {reset_response}")
