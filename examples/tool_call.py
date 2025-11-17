import asyncio

from pydantic import BaseModel, Field

from limin import Tool, generate_completion, ModelConfiguration


class GetWeatherParameters(BaseModel):
    location: str = Field(description="City and country e.g. Bogotá, Colombia")


get_weather_tool = Tool(
    name="get_weather",
    description="Get current temperature for provided location in celsius.",
    parameters=GetWeatherParameters,
)


async def main():
    model_configuration = ModelConfiguration(
        model="gpt-4o", temperature=0.7, tools=[get_weather_tool]
    )
    completion = await generate_completion(
        "What's the weather like in Paris today?",
        model_configuration=model_configuration,
    )
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
