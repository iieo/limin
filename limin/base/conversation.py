from pydantic import BaseModel, Field
from .message import Message, SystemMessage, UserMessage, AssistantMessage, ToolMessage
from .base_util import get_last_element
from openai.types.chat import ChatCompletionMessageParam
import json


class Conversation(BaseModel):
    messages: list[Message] = Field(default_factory=list)

    def add_message(self, message: Message):
        last_message = get_last_element(self.messages)

        if last_message is None:
            if message.role == "assistant":
                raise ValueError("The first message must be a system or user message")

            self.messages.append(message)
            return

        if last_message.role == "system" and message.role != "user":
            raise ValueError("System message must be followed by a user message")

        if last_message.role == "assistant" and message.role not in ["user", "tool"]:
            raise ValueError(
                "Assistant message must be followed by a user or tool message"
            )

        if last_message.role == "user" and message.role != "assistant":
            raise ValueError("User message must be followed by an assistant message")

        self.messages.append(message)

    def to_pretty_string(
        self,
        system_color: str = "\033[1;36m",
        user_color: str = "\033[1;32m",
        assistant_color: str = "\033[1;35m",
    ) -> str:
        pretty_lines = []

        for message in self.messages:
            if message.role == "system":
                color_code = system_color
            elif message.role == "user":
                color_code = user_color
            elif message.role == "assistant":
                color_code = assistant_color

            # Reset color code
            reset_code = "\033[0m"

            pretty_lines.append(f"{color_code}{message.role.capitalize()}{reset_code}")

            separator_length = len(message.role) + 2  # +2 for some extra space
            pretty_lines.append("-" * separator_length)

            pretty_lines.append(f"{message.content}\n")

        return "\n".join(pretty_lines).strip()

    def to_markdown(self) -> str:
        markdown_str = ""
        for message in self.messages:
            markdown_str += f"## {message.role.capitalize()} \n"
            markdown_str += f"{message.content}\n\n"
        return markdown_str.strip()

    def to_evaluation_format(self, include_metadata: bool = True) -> str:
        """
        Generate a structured format specifically for LLM evaluation.
        Clearly shows tool calls and their results.
        """
        output = []

        if include_metadata:
            output.append("# Conversation Transcript for Evaluation\n")
            output.append(f"**Total Messages:** {len(self.messages)}\n")
            output.append("---\n")

        for i, message in enumerate(self.messages):
            if isinstance(message, SystemMessage):
                output.append(f"### System Instruction\n")
                output.append(f"{message.content}\n\n")

            elif isinstance(message, UserMessage):
                output.append(f"### User Input\n")
                output.append(f"{message.content}\n\n")

            elif isinstance(message, AssistantMessage):
                output.append(f"### Assistant Response\n")
                if message.content:
                    output.append(f"**Text Response:** {message.content}\n")

                if message.tool_calls:
                    output.append("\n**🔧 TOOL INVOCATIONS:**\n")
                    for tool_call in message.tool_calls:
                        output.append(f"\n**Tool Called:** `{tool_call.name}`\n")
                        output.append(f"- **Call ID:** `{tool_call.id}`\n")
                        output.append(f"- **Arguments Passed:**\n")
                        output.append("```json\n")
                        output.append(json.dumps(tool_call.arguments, indent=2))
                        output.append("\n```\n")
                output.append("\n")

            elif isinstance(message, ToolMessage):
                output.append(f"### Tool Execution Result\n")
                output.append(f"**Response for Call ID:** `{message.tool_call_id}`\n")
                output.append(f"**Result:**\n```\n{message.content}\n```\n\n")

        return "".join(output)

    @property
    def openai_messages(self) -> list[ChatCompletionMessageParam]:
        return [message.openai_message for message in self.messages]

    @staticmethod
    def from_prompts(
        user_prompt: str,
        assistant_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> "Conversation":
        conversation = Conversation()
        if system_prompt is not None:
            conversation.add_message(SystemMessage(content=system_prompt))
        conversation.add_message(UserMessage(content=user_prompt))
        if assistant_prompt is not None:
            conversation.add_message(
                AssistantMessage(content=assistant_prompt, tool_calls=[])
            )
        return conversation
