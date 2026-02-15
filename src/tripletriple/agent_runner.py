import logging
from .gateway.session import Session, SessionManager

logger = logging.getLogger("tripletriple.runner")

async def run_session_turn(agent, session: Session, text: str, session_manager: SessionManager):
    """
    Run a single turn of the agent loop for use in background tasks
    (Cron, Heartbeat, etc.) where no WebSocket is attached.
    """
    try:
        # 1. Add user message
        session.add_message("user", text)
        session_manager.write_transcript(session, session.messages[-1])
        
        # 2. Run agent
        full_response = ""
        async for chunk in agent.process_message(session, text):
            full_response += chunk
            # TODO: If we had a pub/sub system, we'd publish chunks here 
            # so connected clients could see the stream.

        # 3. Add assistant response
        session.add_message("assistant", full_response)
        session_manager.write_transcript(session, session.messages[-1])
        session_manager.save()
        
        return full_response

    except Exception as e:
        logger.error(f"Error in session turn {session.id}: {e}")
        return f"Error: {e}"
