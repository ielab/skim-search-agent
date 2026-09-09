"""Exception types shared across the library."""


class SetupError(RuntimeError):
    """A run cannot start: a required artifact (a persisted index, an embedding cache, a
    served model) is missing or unusable. Raised while building/indexing a retriever, before
    any episode runs, so the harness aborts immediately instead of recording every instance
    as an error — or worse, letting the agent burn its step budget on tool errors."""


__all__ = ["SetupError"]
