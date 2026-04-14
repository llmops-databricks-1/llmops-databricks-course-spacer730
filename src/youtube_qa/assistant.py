"""Multi-turn YouTube Q&A agent with transcript management tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from youtube_qa.data_processor import DataProcessor
from youtube_qa.vector_search import VectorSearchManager


@dataclass
class ToolDefinition:
    """A small container for agent tool metadata and execution."""

    name: str
    spec: dict[str, Any]
    exec_fn: Callable[..., str]


class YouTubeTranscriptAgent:
    """Agent for chatting about YouTube videos and managing transcript state."""

    def __init__(
        self,
        llm_endpoint: str,
        vector_search_manager: VectorSearchManager,
        data_processor: DataProcessor,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.vector_search_manager = vector_search_manager
        self.data_processor = data_processor
        self.conversation_history: list[dict[str, str]] = []

        self.workspace_client = WorkspaceClient()
        self.llm_client = OpenAI(
            api_key=self.workspace_client.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{self.workspace_client.config.host}/serving-endpoints",
        )
        self._tools = {tool.name: tool for tool in self._build_tools()}

    def _build_tools(self) -> list[ToolDefinition]:
        """Create the custom tools exposed to the model."""
        return [
            ToolDefinition(
                name="add_youtube_video",
                spec={
                    "type": "function",
                    "function": {
                        "name": "add_youtube_video",
                        "description": "Download a YouTube transcript, chunk it, and sync it into the searchable index.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The YouTube video URL to ingest.",
                                }
                            },
                            "required": ["url"],
                        },
                    },
                },
                exec_fn=self._add_youtube_video,
            ),
            ToolDefinition(
                name="remove_youtube_video",
                spec={
                    "type": "function",
                    "function": {
                        "name": "remove_youtube_video",
                        "description": "Remove a YouTube video's transcript chunks from the index and sync the index.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The YouTube video URL to remove.",
                                }
                            },
                            "required": ["url"],
                        },
                    },
                },
                exec_fn=self._remove_youtube_video,
            ),
            ToolDefinition(
                name="reset_youtube_index",
                spec={
                    "type": "function",
                    "function": {
                        "name": "reset_youtube_index",
                        "description": "Remove all videos and chunks from the transcript tables and sync the index.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                },
                exec_fn=self._reset_youtube_index,
            ),
            ToolDefinition(
                name="sync_youtube_index",
                spec={
                    "type": "function",
                    "function": {
                        "name": "sync_youtube_index",
                        "description": "Sync the vector index with the latest transcript chunks.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                },
                exec_fn=self._sync_youtube_index,
            ),
            ToolDefinition(
                name="list_indexed_videos",
                spec={
                    "type": "function",
                    "function": {
                        "name": "list_indexed_videos",
                        "description": "List which YouTube videos are currently stored in the transcript tables.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                },
                exec_fn=self._list_indexed_videos,
            ),
        ]

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Return tool specifications in OpenAI function-calling format."""
        return [tool.spec for tool in self._tools.values()]

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        return self._tools[tool_name].exec_fn(**args)

    def _add_youtube_video(self, url: str) -> str:
        """Ingest a new YouTube video and sync the index."""
        self.data_processor.process_and_save([url])
        self.vector_search_manager.sync_index_and_wait()
        return f"Added video and synced the index for URL: {url}"

    def _remove_youtube_video(self, url: str) -> str:
        """Remove a YouTube video and sync the index."""
        self.data_processor.remove_videos_by_url([url])
        self.vector_search_manager.sync_index_and_wait()
        return f"Removed video and synced the index for URL: {url}"

    def _reset_youtube_index(self) -> str:
        """Reset all stored YouTube transcript data and sync the index."""
        self.data_processor.reset_pipeline_tables()
        self.vector_search_manager.sync_index_and_wait()
        return "Reset all indexed YouTube transcript data and synced the index."

    def _sync_youtube_index(self) -> str:
        """Sync the vector index with the chunks table."""
        self.vector_search_manager.sync_index_and_wait()
        return "Synced the YouTube vector index."

    def _list_indexed_videos(self) -> str:
        """List currently indexed videos."""
        videos = self.data_processor.list_indexed_videos()
        if not videos:
            return "No YouTube videos are currently indexed."

        return json.dumps(videos, indent=2)

    def retrieve(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant transcript chunks for the current question."""
        try:
            return self.vector_search_manager.search_transcript_chunks(
                query=query,
                num_results=num_results,
                columns=["text", "video_id", "chunk_index"],
            )
        except Exception as exc:
            logger.info(f"Transcript retrieval unavailable: {exc}")
            return []

    @staticmethod
    def _format_context(transcript_chunks: list[dict[str, Any]]) -> str:
        """Render retrieved transcript chunks into a prompt-friendly string."""
        if not transcript_chunks:
            return "No transcript chunks are currently available in the index."

        return "\n\n".join(
            (
                f"[Chunk {chunk['chunk_index']} from video {chunk['video_id']}] "
                f"{chunk['text']}"
            )
            for chunk in transcript_chunks
        )

    @staticmethod
    def _build_tool_system_prompt() -> str:
        """Instruction prompt for deciding when to call tools."""
        return """
You are a helpful YouTube Q&A agent.

You can manage a transcript index before answering questions.
Use tools when needed.

Rules:
- If the user provides a YouTube URL and wants to chat about that video, call add_youtube_video first.
- If the user asks to remove a video, call remove_youtube_video.
- If the user asks to clear everything, call reset_youtube_index.
- If you need to inspect current indexed state, call list_indexed_videos.
- If you changed transcript data, ensure the index is synced.
- Do not pretend you already know a video's contents unless transcript chunks have been indexed.
""".strip()

    @staticmethod
    def _build_answer_system_prompt(context: str, tool_summary: str) -> str:
        """Instruction prompt for grounded final answers."""
        return f"""
You are a helpful YouTube Q&A agent.

For index-management requests, use the tool results below as the source of truth.
For video-content questions, use only the retrieved transcript context below.
If there is no transcript context yet for a content question, say that clearly and ask the user to provide a YouTube URL for ingestion.

TOOL RESULTS:
{tool_summary or 'No tools were used for this turn.'}

CONTEXT:
{context}

Keep the answer concise, grounded, and explicit about uncertainty.
""".strip()

    def chat(self, user_message: str, num_chunks: int = 5, max_iterations: int = 5) -> str:
        """Chat with the agent, allowing it to call management tools first."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_tool_system_prompt()},
            *self.conversation_history,
            {"role": "user", "content": user_message},
        ]
        tool_results: list[str] = []

        for _ in range(max_iterations):
            response = self.llm_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs(),
                max_tokens=1000,
            )
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in assistant_message.tool_calls
                    ],
                }
            )

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                logger.info(f"Calling tool: {tool_name}({tool_args})")

                try:
                    result = self.execute_tool(tool_name, tool_args)
                except Exception as exc:
                    result = f"Error: {exc}"

                tool_results.append(f"{tool_name}: {result}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        transcript_chunks = self.retrieve(user_message, num_results=num_chunks)
        answer_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_answer_system_prompt(
                    context=self._format_context(transcript_chunks),
                    tool_summary="\n".join(tool_results),
                ),
            },
            *self.conversation_history,
            {"role": "user", "content": user_message},
        ]

        response = self.llm_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=answer_messages,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content or ""
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self) -> None:
        """Clear the previous conversation turns."""
        self.conversation_history = []