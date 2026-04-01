"""YouTube transcripts ingestion and chunking.

Pipeline:
    list[YouTube URLs]
        -> download_and_store_transcripts
        -> youtube_videos (Delta)
        -> process_chunks
        -> youtube_chunks_table (Delta, CDF enabled)
        -> Vector Search index (see VectorSearchManager)
"""

from __future__ import annotations

import re
import time
from urllib.parse import parse_qs, urlparse

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col, current_timestamp
from youtube_transcript_api import YouTubeTranscriptApi

from youtube_qa.config import ProjectConfig


class DataProcessor:
    """Download YouTube transcripts, store them, and create text chunks."""

    _CHUNK_SIZE_CHARS = 1500
    _CHUNK_OVERLAP_CHARS = 200

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """Initialize the processor.

        Args:
            spark: Spark session.
            config: Project config.
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema

        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())

        self.videos_table = f"{self.catalog}.{self.schema}.youtube_videos"
        self.chunks_table = f"{self.catalog}.{self.schema}.youtube_chunks_table"

    @staticmethod
    def _extract_video_id(url_or_id: str) -> str:
        """Extract a YouTube video id from a URL or return the id as-is."""
        candidate = url_or_id.strip()

        if re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
            return candidate

        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
        path = parsed.path.strip("/")

        if host == "youtu.be":
            vid = path.split("/")[0]
            if vid:
                return vid

        if host.endswith("youtube.com"):
            if path == "watch":
                qs = parse_qs(parsed.query)
                vid = (qs.get("v") or [""])[0]
                if vid:
                    return vid

            if path.startswith("shorts/"):
                parts = path.split("/")
                if len(parts) >= 2 and parts[1]:
                    return parts[1]

            if path.startswith("embed/"):
                parts = path.split("/")
                if len(parts) >= 2 and parts[1]:
                    return parts[1]

        raise ValueError(f"Could not extract video id from: {url_or_id}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize transcript text by collapsing whitespace."""
        return re.sub(r"\s+", " ", text).strip()

    def _fetch_transcript_text(self, video_id: str) -> str:
        """Fetch transcript and return as plain text.

        Notes:
            youtube-transcript-api returns a list of dict segments with keys like
            'text', 'start', 'duration'. This keeps only the combined text.
        """
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        combined = " ".join(seg.get("text", "") for seg in segments)
        return self._normalize_text(combined)

    @classmethod
    def _chunk_text(cls, text: str) -> list[str]:
        """Split text into overlapping fixed-size chunks (character-based)."""
        cleaned = cls._normalize_text(text)
        if not cleaned:
            return []

        size = cls._CHUNK_SIZE_CHARS
        overlap = cls._CHUNK_OVERLAP_CHARS
        step = max(1, size - overlap)

        return [cleaned[i : i + size] for i in range(0, len(cleaned), step)]

    def download_and_store_transcripts(self, urls: list[str]) -> list[dict] | None:
        """Download transcripts and store into `youtube_videos` Delta table."""
        if not urls:
            logger.info("No URLs provided.")
            return None

        records: list[dict] = []

        for url in urls:
            video_id = self._extract_video_id(url)
            try:
                transcript_text = self._fetch_transcript_text(video_id)
                if not transcript_text:
                    logger.warning(f"Empty transcript for video_id={video_id}")
                    continue

                records.append(
                    {
                        "video_id": video_id,
                        "source_url": url,
                        "transcript_text": transcript_text,
                        "processed": int(self.end),
                    }
                )
            except Exception as exc:
                logger.warning(f"Failed to fetch transcript for video_id={video_id}: {exc}")

        if not records:
            logger.info("No transcripts downloaded.")
            return None

        schema = T.StructType(
            [
                T.StructField("video_id", T.StringType(), False),
                T.StructField("source_url", T.StringType(), True),
                T.StructField("transcript_text", T.StringType(), True),
                T.StructField("processed", T.LongType(), True),
            ]
        )

        df = self.spark.createDataFrame(records, schema=schema).withColumn(
            "ingest_ts", current_timestamp()
        )

        df.write.format("delta").mode("ignore").saveAsTable(self.videos_table)
        df.createOrReplaceTempView("new_videos")

        self.spark.sql(f"""
            MERGE INTO {self.videos_table} target
            USING new_videos source
            ON target.video_id = source.video_id
            WHEN NOT MATCHED THEN INSERT (
                video_id, source_url, transcript_text, processed
            ) VALUES (
                source.video_id, source.source_url, source.transcript_text, source.processed
            )
        """)

        logger.info(f"Merged {len(records)} transcript records into {self.videos_table}")
        return records

    def process_chunks(self) -> None:
        """Create chunk rows from transcripts and write them to the chunks table."""
        logger.info(f"Chunking transcripts from {self.videos_table} for processed={self.end}")

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                id STRING,
                video_id STRING,
                chunk_index INT,
                text STRING,
                processed LONG,
                ingest_ts TIMESTAMP
            )
            USING DELTA
        """)

        rows = (
            self.spark.table(self.videos_table)
            .where(col("processed") == int(self.end))
            .select("video_id", "transcript_text")
            .collect()
        )

        chunk_records: list[dict] = []
        for row in rows:
            video_id = row["video_id"]
            transcript_text = row["transcript_text"] or ""

            for idx, chunk in enumerate(self._chunk_text(transcript_text)):
                chunk_records.append(
                    {
                        "id": f"{video_id}_{idx}",
                        "video_id": video_id,
                        "chunk_index": idx,
                        "text": chunk,
                        "processed": int(self.end),
                    }
                )

        if not chunk_records:
            logger.info("No chunks produced.")
            return

        chunk_schema = T.StructType(
            [
                T.StructField("id", T.StringType(), False),
                T.StructField("video_id", T.StringType(), False),
                T.StructField("chunk_index", T.IntegerType(), False),
                T.StructField("text", T.StringType(), False),
                T.StructField("processed", T.LongType(), True),
            ]
        )

        chunks_df = self.spark.createDataFrame(chunk_records, schema=chunk_schema).withColumn(
            "ingest_ts", current_timestamp()
        )

        chunks_df.write.mode("append").saveAsTable(self.chunks_table)
        logger.info(f"Saved {len(chunk_records)} chunks to {self.chunks_table}")

        self.spark.sql(f"""
            ALTER TABLE {self.chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {self.chunks_table}")

    def process_and_save(self, urls: list[str]) -> None:
        """Download transcripts and write transcript chunks."""
        records = self.download_and_store_transcripts(urls)
        if records is None:
            logger.info("No new videos to process. Exiting.")
            return

        self.process_chunks()
        logger.info("Processing complete!")
