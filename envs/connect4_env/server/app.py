"""FastAPI application for the Connect4 Environment."""

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import Connect4Action, Connect4Observation
    from .connect4_environment import Connect4Environment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import Connect4Action, Connect4Observation
    from server.connect4_environment import Connect4Environment

# Create the FastAPI app
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(
    Connect4Environment, Connect4Action, Connect4Observation, env_name="connect4_env"
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
