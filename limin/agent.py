from .base import (
    Tool,
    ToolMessage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    Message,
)
from .base import Conversation, ModelConfiguration
from .completion import generate_completion_for_conversation


def check_tools(tools: list[Tool]) -> None:
    tool_names = [tool.name for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        # Find duplicate names
        seen = set()
        duplicates = set()
        for name in tool_names:
            if name in seen:
                duplicates.add(name)
            else:
                seen.add(name)

        duplicate_names = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Tools must have unique names. Duplicate names: {duplicate_names}"
        )


class Agent:
    def __init__(
        self,
        system_prompt: str,
        model_configuration: ModelConfiguration,
    ):
        self.model_configuration = model_configuration

        self.conversation = Conversation()
        self.conversation.add_message(SystemMessage(content=system_prompt))

        check_tools(model_configuration.tools)
        self.tools = model_configuration.tools

    async def process(self, user_message: str) -> list[Message]:
        messages = []
        self.conversation.add_message(UserMessage(content=user_message))
        messages.append(self.conversation.messages[-1])

        # Keep calling tools until the agent doesn't want to call any more
        while len(self.tools) > 0:
            tool_call_completion = await generate_completion_for_conversation(
                self.conversation,
                model_configuration=self.model_configuration,
            )

            if len(tool_call_completion.tool_calls) == 0:
                # No more tools to call, break out of the loop
                break

            assistant_message = AssistantMessage(
                content=None,
                tool_calls=tool_call_completion.tool_calls,
            )
            self.conversation.add_message(assistant_message)
            messages.append(assistant_message)

            for tool_call in tool_call_completion.tool_calls:
                # Get the tool with the correct name
                tool = next(
                    tool for tool in self.tools if tool.name == tool_call.name
                )

                # Execute the tool
                content = tool.execute(**tool_call.arguments)
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call.id,
                )
                self.conversation.add_message(tool_message)
                messages.append(tool_message)

        # Generate a response from the model
        text_completion = await generate_completion_for_conversation(
            self.conversation,
            model_configuration=self.model_configuration,
        )

        self.conversation.add_message(text_completion.to_assistant_message())
        messages.append(text_completion.to_assistant_message())
        return messages
