"""Utility functions for SSL certificate configuration."""

import os
from pathlib import Path

import certifi
from structlog import getLogger

logger = getLogger(__name__)


def setup_ssl_environment():
    """Set up environment variables for SSL certificate verification.

    This function sets the necessary environment variables to use the
    certificates from the certifi package installed in the virtual environment.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cert_path = certifi.where()
        if not cert_path or not Path(cert_path).exists():
            logger.error("Certifi certificate path not found")
            return False

        # Set environment variables for certificate path
        os.environ["SSL_CERT_FILE"] = cert_path
        os.environ["REQUESTS_CA_BUNDLE"] = cert_path
        os.environ["CURL_CA_BUNDLE"] = cert_path

        logger.info(
            "SSL environment variables configured successfully", cert_path=cert_path
        )
        return True
    except Exception as e:
        logger.error("Failed to configure SSL environment", error=str(e))
        return False
