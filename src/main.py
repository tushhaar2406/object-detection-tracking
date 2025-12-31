import sys
import signal
from pathlib import Path

import yaml

from logging.logger import setup_logger
from pipeline.video_pipeline import VideoPipeline


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to config.yaml

    Returns:
        dict: Parsed configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def handle_shutdown(pipeline: VideoPipeline):
    """
    Graceful shutdown handler.
    Ensures resources are released properly.
    """
    def _shutdown(signum, frame):
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


def main():
    """
    Application entry point.
    """
    try:
        # -------------------------------------------------
        # Load configuration
        # -------------------------------------------------
        config = load_config("configs/config.yaml")

        # -------------------------------------------------
        # Initialize logger
        # -------------------------------------------------
        logger = setup_logger(config)
        logger.info("Application starting...")
        logger.info(f"Environment: {config['app']['environment']}")

        # -------------------------------------------------
        # Initialize pipeline
        # -------------------------------------------------
        pipeline = VideoPipeline(config=config, logger=logger)

        # -------------------------------------------------
        # Register graceful shutdown
        # -------------------------------------------------
        handle_shutdown(pipeline)

        # -------------------------------------------------
        # Start pipeline
        # -------------------------------------------------
        pipeline.run()

    except Exception as exc:
        print(f"[FATAL] Application failed to start: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
