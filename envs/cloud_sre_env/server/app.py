"""
FastAPI application for the Cloud SRE Environment.
Uses the OpenEnv create_fastapi_app helper.
"""

from openenv.core.env_server import create_fastapi_app
from ..models import SREAction, SREObservation
from .cloud_sre_environment import CloudSREEnvironment

env = CloudSREEnvironment()
app = create_fastapi_app(env, SREAction, SREObservation)
