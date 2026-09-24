from __future__ import annotations


class NotFoundError(KeyError):
    """A requested customer, dataset or report does not exist (served as HTTP 404)."""

    def __str__(self) -> str:
        # KeyError quotes its message; a client should read it verbatim.
        return str(self.args[0]) if self.args else "Not found"
