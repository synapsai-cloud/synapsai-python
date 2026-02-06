# Copyright 2026 SynapsAI Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Optional

LOGGER_NAME = "synapsai"
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger within the synapsai namespace.

    Example:
        logger = get_logger(__name__)
    """
    if not name:
        return logging.getLogger(LOGGER_NAME)

    # Ensure everything stays under synapsai.*
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)

    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def enable_debug_logging(level: int = logging.DEBUG) -> None:
    """
    Convenience function for developers who want quick SDK logging.

    This adds a simple StreamHandler to the synapsai logger only.
    It does NOT modify global logging configuration.

    Example:
        import synapsai
        from synapsai.logging import enable_debug_logging

        enable_debug_logging()
    """

    logger = logging.getLogger(LOGGER_NAME)

    # Avoid adding duplicate handlers
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.setLevel(level)
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
