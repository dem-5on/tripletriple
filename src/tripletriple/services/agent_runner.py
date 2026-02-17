import asyncio
import logging
from ..session import Session, SessionManager

logger = logging.getLogger("tripletriple.runner")

# Track rate-limit state to avoid hammering a dead API
_rate_limited_until: float = 0


async def run_session_turn(agent, session: Session, text: str, session_manager: SessionManager):
    """
    Run a single turn of the agent loop for use in background tasks
    (Cron, Heartbeat, etc.) where no WebSocket is attached.
    """
    import time
    global _rate_limited_until

    # If we're currently rate-limited, skip immediately
    if time.time() < _rate_limited_until:
        remaining = int(_rate_limited_until - time.time())
        logger.warning(f"Skipping session turn — rate-limited for {remaining}s more")
        return f"⚠️ Rate-limited. Retry in {remaining}s."

    try:
        # 1. Add user message
        session.add_message("user", text)
        session_manager.write_transcript(session, session.messages[-1])
        
        # 2. Run agent
        full_response = ""
        async for chunk in agent.process_message(session, text):
            full_response += chunk

        # 3. Add assistant response
        session.add_message("assistant", full_response)
        session_manager.write_transcript(session, session.messages[-1])
        session_manager.save()
        
        return full_response

    except Exception as e:
        error_str = str(e)

        # Detect rate-limit (429) errors and back off
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "Too Many Requests" in error_str:
            _rate_limited_until = time.time() + 300  # Back off for 5 minutes
            logger.error(
                f"Rate-limited (429) in session {session.id}. "
                f"Backing off for 5 minutes."
            )
            return "⚠️ API rate limit hit. Backing off for 5 minutes."

        logger.error(f"Error in session turn {session.id}: {e}")
        return f"Error: {e}"
