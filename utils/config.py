# utils/config.py

import os

from dotenv import load_dotenv

# Load environment variables from .env file with override=True to ensure values are updated
load_dotenv(override=True)


def get_broker_api_key() -> str | None:
    """
    Retrieve the configured broker API key.

    Returns:
        str | None: The broker API key from environment variables, or None if not set.
    """
    return os.getenv("BROKER_API_KEY")


def get_broker_api_secret() -> str | None:
    """
    Retrieve the configured broker API secret.

    Returns:
        str | None: The broker API secret from environment variables, or None if not set.
    """
    return os.getenv("BROKER_API_SECRET")


def get_login_rate_limit_min() -> str:
    """
    Retrieve the rate limit for logins per minute.

    Returns:
        str: The rate limit string (e.g., '5 per minute').
    """
    return os.getenv("LOGIN_RATE_LIMIT_MIN", "5 per minute")


def get_login_rate_limit_hour() -> str:
    """
    Retrieve the rate limit for logins per hour.

    Returns:
        str: The rate limit string (e.g., '25 per hour').
    """
    return os.getenv("LOGIN_RATE_LIMIT_HOUR", "25 per hour")


def get_host_server() -> str:
    """
    Retrieve the host server URL.

    Returns:
        str: The host server URL string.
    """
    return os.getenv("HOST_SERVER", "http://127.0.0.1:5000")


def get_frontend_url() -> str:
    """
    Retrieve the frontend URL for redirects after auth.

    When FRONTEND_URL is set (e.g. Vercel deployment), OAuth callbacks
    redirect there instead of to the backend's own Jinja2 templates.
    Falls back to HOST_SERVER so single-host deployments still work.

    Returns:
        str: The frontend URL string.
    """
    return os.getenv("FRONTEND_URL", get_host_server())
