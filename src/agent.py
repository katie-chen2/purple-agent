from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
import os

class Agent:
    def __init__(self):
        load_dotenv()
        # Lazy init: create client only if needed to generate prompts
        self.client = None
        self.config = genai_types.GenerateContentConfig(
            system_instruction="You are a game AI agent designed to maximize memory context usage. \
                Your responses will be submitted as requests to the monitor agent. \
                Your responses should be clear and comprehensive.",
            temperature=0.7,
        )
        self.messenger = Messenger()
        # Optional: URL of another agent to contact. If not provided, skip cross-agent call.
        self.green_agent_url = os.environ.get("GREEN_AGENT_URL")
        self.agent_name = f"MaximizerAgent"
        # Initialize other state here

    def _generate_maximization_prompt(self, input_text: str) -> str:
        prompt_content = f"You are a skilled prompt engineer. \
        Design a prompt request to maximize memory context occupation based on the following strategy: {input_text}\n\n \
        Provide a detailed and comprehensive response that maximizes the information provided."
        if self.client is None:
            self.client = genai.Client()
        response = self.client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[prompt_content],
            config=self.config,
        )
        if response.text is None:
            raise ValueError("No text generated from the model.")
        else:
            return response.text

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        # Replace this example code with your agent logic

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=input_text))],
            name="Echo",
        )
        
        # Optionally contact another agent if configured; otherwise, finish locally.
        if self.green_agent_url:
            input_prompt = self._generate_maximization_prompt(input_text)
            parts = [
                Part(
                    root=TextPart(
                        text=input_prompt,
                        metadata={"sender": self.agent_name}
                    )
                )
            ]
            await self.messenger.talk_to_agent_advanced(
                parts=parts,
                url=self.green_agent_url,
                metadata={"sender": self.agent_name},
            )
        else:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="No GREEN_AGENT_URL configured; completed locally."))],
                name="Info",
            )
