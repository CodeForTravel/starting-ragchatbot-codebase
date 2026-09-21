from google import genai
from google.genai import types
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Google's Gemini API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to two tools for course information: `search_course_content` and `get_course_outline`.

Tool Usage:
- Use `search_course_content` **only** for questions about specific course content or detailed educational materials
- Use `get_course_outline` for outline-related queries (course outline, syllabus, structure, lesson list, "what lessons does course X have")
- **Up to 2 sequential tool calls per query.** Make a second call only when it depends on the result of the first (e.g. get an outline to learn a lesson title, then search on that title). Otherwise use one call
- For "find another course that covers the same topic as lesson N of course X": get the outline of X first, then search using that lesson's title with **no `course_name`**, so other courses are searched too
- Never repeat a call that already returned results
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Outline Responses:
- Always include the course title, the course link, and every lesson with its number and title (e.g. "Lesson 0: <title>")
- Do not omit or summarize away any lessons

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Search first, then answer
- **Course outline questions**: Get the outline first, then answer
- **Multi-step questions** (comparisons, info from several courses/lessons): outline first if needed, then search, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_ROUNDS = 2  # max sequential tool rounds per query

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _build_config(
        self,
        system_content: str,
        tools: Optional[List] = None,
        allow_calls: bool = True,
    ) -> types.GenerateContentConfig:
        """Build generation config; tool calls are executed by us, not automatically by the SDK.

        allow_calls=False keeps the tools declared (the history contains function calls)
        but forbids the model from calling them, so it must answer in text.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_content,
            temperature=0,
            max_output_tokens=800,
        )
        if tools:
            config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"],
                            description=t["description"],
                            parameters_json_schema=t["input_schema"],
                        )
                        for t in tools
                    ]
                )
            ]
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
                disable=True
            )
            if not allow_calls:
                config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                )
        return config

    def _generate(self, contents: List, config: types.GenerateContentConfig):
        """One Gemini request; passes a copy of contents so later appends don't alter what was sent."""
        return self.client.models.generate_content(
            model=self.model,
            contents=list(contents),
            config=config,
        )

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        The model may make up to MAX_ROUNDS sequential tool rounds; each round is a separate
        API request so it can reason about earlier results. A round is one model response,
        which may contain several parallel function calls. The loop ends when the model
        answers without calling a tool, after MAX_ROUNDS rounds, or when a tool raises; in
        the last two cases one final text-only request produces the answer.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=query)])
        ]
        config = self._build_config(system_content, tools)
        response = self._generate(contents, config)

        rounds = 0
        while response.function_calls and tool_manager:
            rounds += 1
            # Keep the model's function-call turn as-is (preserves any thought signatures)
            contents.append(response.candidates[0].content)
            result_parts, failed = self._execute_calls(
                response.function_calls, tool_manager
            )
            contents.append(types.Content(role="user", parts=result_parts))

            if failed or rounds >= self.MAX_ROUNDS:
                return self._final_text(contents, system_content, tools)
            response = self._generate(contents, config)

        if response.text or rounds == 0:
            return response.text or ""
        return self._final_text(
            contents, system_content, tools
        )  # model went silent after a tool round

    def _execute_calls(self, function_calls: List, tool_manager):
        """Run every call of one round; returns (function_response parts, whether any tool raised).

        Every call gets a response part (an error payload on failure) so the history stays valid.
        """
        parts = []
        failed = False
        for call in function_calls:
            try:
                result = {
                    "result": tool_manager.execute_tool(call.name, **(call.args or {}))
                }
            except Exception as e:
                failed = True
                result = {"error": f"Tool '{call.name}' failed: {e}"}
            parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )
        return parts, failed

    def _final_text(
        self, contents: List, system_content: str, tools: Optional[List]
    ) -> str:
        """Final request: tools stay declared but calling is disabled, so the model must answer in text."""
        final_config = self._build_config(system_content, tools, allow_calls=False)
        for _ in range(2):  # retry once if the model returns no text
            final_response = self._generate(contents, final_config)
            if final_response.text:
                return final_response.text
        return "Sorry, I couldn't generate an answer. Please try asking again."
