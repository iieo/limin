from pydantic import BaseModel, ConfigDict
from openai import NOT_GIVEN, NotGiven
from .tool import Tool


class ModelConfiguration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "gpt-5"
    temperature: float = 1.0
    log_probs: bool = False
    top_log_probs: int | None | NotGiven = NOT_GIVEN
    max_tokens: int | None | NotGiven = NOT_GIVEN
    presence_penalty: float | None | NotGiven = NOT_GIVEN
    frequency_penalty: float | None | NotGiven = NOT_GIVEN
    top_p: float | None | NotGiven = NOT_GIVEN
    seed: int | None | NotGiven = NOT_GIVEN
    api_key: str | None = None
    base_url: str | None = None
    tools: list[Tool] = []


DEFAULT_MODEL_CONFIGURATION = ModelConfiguration()
