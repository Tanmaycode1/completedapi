import asyncio
from typing import List, Dict, Any, Optional
import logging
from openai import OpenAI, AsyncOpenAI
from ..prompts.config import PromptConfig
from ..utils.json_handler import JSONHandler, JSONProcessingError

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize OpenAI client with both sync and async clients.
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model to use (default: "gpt-4")
        """
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.json_handler = JSONHandler()
        self.max_chunks = 10
        self.max_retries = 3
        self.model = model

    async def get_complete_response(
            self,
            messages: List[Dict[str, str]],
            config: PromptConfig,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get complete response with chunking support.
        
        Args:
            messages: List of message dictionaries
            config: Prompt configuration
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
        
        Returns:
            Dict containing merged response
        """
        chunks = []
        partial_content = ""
        retry_count = 0

        max_tokens = max_tokens or config.max_tokens
        temperature = temperature or config.temperature

        while True:
            try:
                # Prepare messages with continuation context
                current_messages = messages.copy()
                if partial_content:
                    current_messages.append({
                        "role": "assistant",  # Changed from "user" to "assistant" for continuation
                        "content": f"Continue from: {partial_content}"
                    })

                # Make API request
                response = await self._make_request(
                    current_messages,
                    max_tokens,
                    temperature
                )

                content = response.choices[0].message.content

                try:
                    # Process chunk using json handler
                    chunk = await self.json_handler.process_chunk(content)
                    if chunk:
                        chunks.append(chunk)

                    # Check if response is complete
                    if any(marker in content for marker in config.completion_markers):
                        break

                    # Get continuation point using chunk handler
                    completed, partial_content = self.json_handler.find_continuation_point(content)
                    if completed or not partial_content:
                        break

                    if len(chunks) >= self.max_chunks:
                        logger.warning("Maximum chunk limit reached")
                        break

                except JSONProcessingError as e:
                    logger.warning(f"JSON processing error: {str(e)}")
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        if chunks:
                            return await self.json_handler.merge_chunks(chunks)
                        return {"content": content}
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    continue

            except Exception as e:
                logger.error(f"Error in get_complete_response: {str(e)}")
                if chunks:
                    return await self.json_handler.merge_chunks(chunks)
                raise OpenAIError(f"Failed to get complete response: {str(e)}")

        try:
            # Merge chunks using json handler
            merged_response = await self.json_handler.merge_chunks(chunks)
            return merged_response
        except JSONProcessingError as e:
            if chunks:
                return chunks[-1]
            raise OpenAIError(f"Failed to merge response chunks: {str(e)}")

    async def _make_request(
            self,
            messages: List[Dict[str, str]],
            max_tokens: int,
            temperature: float
    ):
        """
        Make a request to OpenAI API with error handling.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
        
        Returns:
            OpenAI API response
            
        Raises:
            OpenAIError: If API request fails
        """
        try:
            return await self.async_client.chat.completions.create(
                model=self.model,  # Use instance model instead of hardcoded value
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise OpenAIError(f"API request failed: {str(e)}")