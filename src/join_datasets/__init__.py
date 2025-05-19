"""Joining datasets attribution package."""

__all__ = ["Attribution"]


def __getattr__(name):
    if name == "Attribution":
        from .attribution import Attribution
        return Attribution
    raise AttributeError(f"module {__name__} has no attribute {name}")
