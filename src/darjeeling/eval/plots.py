from __future__ import annotations


def plotly_available() -> bool:
    try:
        import plotly  # noqa: F401
    except ImportError:
        return False
    return True
