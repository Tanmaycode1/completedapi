import json
import logging
from typing import List, Dict, Any, Tuple
import re

logger = logging.getLogger(__name__)


class JSONProcessingError(Exception):
    """Custom exception for JSON processing errors."""
    pass


class JSONHandler:
    @staticmethod
    async def process_chunk(content: str) -> Dict[str, Any]:
        """Process and validate a JSON chunk."""
        try:
            # Clean the content
            cleaned = content.strip()
            # Remove markdown code blocks and json language identifier
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)

            # Try to find JSON structure
            start_brace = cleaned.find('{')
            start_bracket = cleaned.find('[')

            if start_brace == -1 and start_bracket == -1:
                # If no JSON structure found, try to format as JSON
                try:
                    return {"content": cleaned}
                except Exception:
                    raise JSONProcessingError("No valid JSON structure found")

            # Determine the start position
            start = min(pos for pos in [start_brace, start_bracket] if pos != -1)

            # Find corresponding end bracket/brace
            if cleaned[start] == '{':
                end = cleaned.rindex('}') + 1
            else:
                end = cleaned.rindex(']') + 1

            json_str = cleaned[start:end]

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON is invalid, try to fix common issues
                fixed_json = json_str.replace('\n', ' ').replace('\\', '\\\\')
                fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)  # Remove trailing commas
                fixed_json = re.sub(r'([{\[,])\s*,', r'\1', fixed_json)  # Remove empty elements
                return json.loads(fixed_json)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            # Try to return as plain text if JSON parsing fails
            return {"content": cleaned}
        except Exception as e:
            logger.error(f"Error processing JSON chunk: {str(e)}")
            raise JSONProcessingError(str(e))

    @staticmethod
    async def merge_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple JSON chunks into a single response."""
        if not chunks:
            raise JSONProcessingError("No chunks to merge")

        if len(chunks) == 1:
            return chunks[0]

        def merge_values(v1: Any, v2: Any) -> Any:
            """Merge two values recursively."""
            if isinstance(v1, dict) and isinstance(v2, dict):
                result = v1.copy()
                for key, value in v2.items():
                    if key in result:
                        result[key] = merge_values(result[key], value)
                    else:
                        result[key] = value
                return result
            elif isinstance(v1, list) and isinstance(v2, list):
                # Use string representation to check for duplicates
                seen = {str(item) for item in v1}
                merged = v1.copy()
                for item in v2:
                    if str(item) not in seen:
                        merged.append(item)
                        seen.add(str(item))
                return merged
            # For non-container types, prefer non-None values
            return v2 if v2 is not None else v1

        try:
            result = chunks[0].copy()
            for chunk in chunks[1:]:
                result = merge_values(result, chunk)
            return result
        except Exception as e:
            logger.error(f"Error merging chunks: {str(e)}")
            # If merging fails, concatenate text content
            return {
                "content": " ".join(
                    chunk.get("content", str(chunk))
                    for chunk in chunks
                )
            }

    @staticmethod
    def find_continuation_point(content: str) -> Tuple[str, str]:
        """Find where to continue from in partial JSON."""
        # Find the last complete JSON structure
        stack = []
        last_complete_pos = -1
        in_string = False
        escape_char = False

        for i, char in enumerate(content):
            if not escape_char and char == '\\':
                escape_char = True
                continue

            if not escape_char and char == '"':
                in_string = not in_string

            if not in_string:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if stack:
                        stack.pop()
                        if not stack:
                            last_complete_pos = i + 1

            escape_char = False

        if last_complete_pos == -1:
            # No complete JSON structure found
            # Try to find a logical breaking point
            sentences = content.split('. ')
            if len(sentences) > 1:
                return '. '.join(sentences[:-1]) + '.', sentences[-1]
            return content, ""

        return content[:last_complete_pos], content[last_complete_pos:]
