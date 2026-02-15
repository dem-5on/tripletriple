from abc import ABC, abstractmethod
from typing import AsyncGenerator
from ..gateway.session import Session

class Agent(ABC):
    @abstractmethod
    async def process_message(self, session: Session, message: str) -> AsyncGenerator[str, None]:
        """
        Process a message and stream the response.
        """
        pass

class EchoAgent(Agent):
    """
    A simple echo agent for testing.
    """
    async def process_message(self, session: Session, message: str) -> AsyncGenerator[str, None]:
        yield f"Echo: {message}"
