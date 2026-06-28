"""Darjeeling core package.

Active code in this package follows docs/design/reboot as the architecture
source of truth.
"""

from darjeeling.model import *  # noqa: F403

__all__ = [name for name in globals() if not name.startswith("_")]
