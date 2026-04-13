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
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col, current_timestamp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

from youtube_qa.config import ProjectConfig


class DataProcessor:
    """Download YouTube transcripts, store them, and create text chunks."""

    _CHUNK_SIZE_CHARS = 1500
    _CHUNK_OVERLAP_CHARS = 200
    _RECENT_DEDUP_LOOKBACK_HOURS = 24

    def __init__(
        self,
        spark: SparkSession,
        config: ProjectConfig,
        *,
        proxy_username: str | None = None,
        proxy_password: str | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            spark: Spark session.
            config: Project config.
            proxy_username: Webshare proxy username.
            proxy_password: Webshare proxy password.
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema

        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())

        self.videos_table = f"{self.catalog}.{self.schema}.youtube_videos"
        self.chunks_table = f"{self.catalog}.{self.schema}.youtube_chunks_table"

        if proxy_username and proxy_password:
            self.ytt_api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
                    proxy_username=proxy_username,
                    proxy_password=proxy_password,
                    retries_when_blocked=1,
                )
            )
        else:
            self.ytt_api = YouTubeTranscriptApi()

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
        """Fetch transcript and return as plain text."""
        transcript = self.ytt_api.fetch(video_id)
        combined = " ".join(snippet.text for snippet in transcript)
        return self._normalize_text(combined)

    def _recent_processed_cutoff(self) -> int:
        """Return a time-based cutoff for recent-ingest deduplication."""
        run_timestamp = datetime.strptime(self.end, "%Y%m%d%H%M").replace(tzinfo=UTC)
        cutoff = run_timestamp - timedelta(hours=self._RECENT_DEDUP_LOOKBACK_HOURS)
        return int(cutoff.strftime("%Y%m%d%H%M"))

    def _get_recent_video_ids(self, video_ids: list[str]) -> set[str]:
        """Return video ids already ingested within the recent dedup window."""
        if not video_ids or not self.spark.catalog.tableExists(self.videos_table):
            return set()

        cutoff = self._recent_processed_cutoff()
        candidate_df = self.spark.createDataFrame(
            [(video_id,) for video_id in video_ids], ["video_id"]
        )

        rows = (
            self.spark.table(self.videos_table)
            .where(col("processed") >= cutoff)
            .select("video_id")
            .distinct()
            .join(candidate_df, on="video_id", how="inner")
            .collect()
        )
        return {row["video_id"] for row in rows}

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

        unique_urls_by_video_id: dict[str, str] = {}
        for url in urls:
            video_id = self._extract_video_id(url)
            if video_id in unique_urls_by_video_id:
                logger.info(f"Skipping duplicate input URL for video_id={video_id}")
                continue

            unique_urls_by_video_id[video_id] = url

        recent_existing_ids = self._get_recent_video_ids(list(unique_urls_by_video_id))
        if recent_existing_ids:
            logger.info(
                "Skipping {} video(s) already ingested since processed >= {}",
                len(recent_existing_ids),
                self._recent_processed_cutoff(),
            )

        records: list[dict] = []

        for video_id, url in unique_urls_by_video_id.items():
            if video_id in recent_existing_ids:
                continue

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

        if self.spark.catalog.tableExists(self.videos_table):
            df.write.format("delta").mode("append").saveAsTable(self.videos_table)
        else:
            df.write.format("delta").saveAsTable(self.videos_table)

        logger.info(f"Saved {len(records)} transcript records into {self.videos_table}")
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
