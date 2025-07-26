import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load Gemini API key securely from environment variable
gemini_api_key = "AIzaSyDKOKL1CHtUhjZpsfuIbYdu11U4FhuW5jI"

if not gemini_api_key:
    raise ValueError(" Missing GEMINI_API_KEY. Set it as an environment variable.")

# Initialize Gemini client (OpenAI-compatible)
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Set run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define your agent with optional instructions
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant."
)

# Run the agent synchronously
try:
    result = Runner.run_sync(
        starting_agent=agent,
        input="Hello, how are you?",
        run_config=config
    )
    print(result.final_output)

except Exception as e:
    print(f"Error occurred: {e}")
