from contextlib import contextmanager
from contextvars import ContextVar


class Virtualized:
    """The same loop body can describe dependencies or emit device code."""

    def __init__(self):
        self._ops = ContextVar("demo_ops")

    @property
    def ops(self):
        return self._ops.get()

    @contextmanager
    def set_ops_handler(self, handler):
        token = self._ops.set(handler)
        try:
            yield handler
        finally:
            self._ops.reset(token)


V = Virtualized()
