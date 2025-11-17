import asyncio
from limin import generate_completion


async def main():
    completion = await generate_completion("What is the capital of France?")
    print(completion.content)


if __name__ == "__main__":
    asyncio.run(main())
