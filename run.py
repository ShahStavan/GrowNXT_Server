"""GrowNXT Server Main Application Entry Point.

Invokes the Flask Application Factory and starts the API web server.

Google Python Style Guide Compliant.
"""

import argparse
import logging
import sys

from api.app import check_env, create_app

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parses CLI options and launches the GrowNXT Flask API Server."""
    parser = argparse.ArgumentParser(
        description="Run GrowNXT Server Flask Application."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host IP address to bind server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number to listen on (default: 5000)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    args = parser.parse_args()

    if not check_env():
        logger.error("Environment validation failed. Terminating server launch.")
        sys.exit(1)

    app = create_app()
    logger.info(
        "Launching GrowNXT API Server on http://%s:%d (Debug=%s)...",
        args.host,
        args.port,
        args.debug,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
