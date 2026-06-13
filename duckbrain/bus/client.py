"""Async pub/sub client for the duck bus.

A :class:`BusClient` connects to a :class:`~duckbrain.bus.broker.Broker`,
publishes typed :data:`~duckbrain.bus.schemas.BusMessage` values to topics, and
receives messages on subscribed topics.
"""

from __future__ import annotations

from types import TracebackType

from pydantic import TypeAdapter
from websockets.asyncio.client import ClientConnection, connect

from duckbrain.bus.protocol import Frame
from duckbrain.bus.schemas import BusMessage

_FRAME_ADAPTER: TypeAdapter[Frame] = TypeAdapter(Frame)


class BusClient:
    """A single WebSocket connection to the broker.

    Use as an async context manager::

        async with BusClient(host, port) as client:
            await client.subscribe("expression")
            msg = await client.receive()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._uri = f"ws://{host}:{port}"
        self._conn: ClientConnection | None = None

    async def __aenter__(self) -> BusClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        self._conn = await connect(self._uri)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def _live(self) -> ClientConnection:
        if self._conn is None:
            raise RuntimeError("BusClient is not connected; call connect() first.")
        return self._conn

    async def subscribe(self, topic: str) -> None:
        frame = Frame(op="subscribe", topic=topic)
        await self._live.send(frame.model_dump_json())

    async def unsubscribe(self, topic: str) -> None:
        frame = Frame(op="unsubscribe", topic=topic)
        await self._live.send(frame.model_dump_json())

    async def publish(self, topic: str, message: BusMessage) -> None:
        frame = Frame(op="publish", topic=topic, message=message)
        await self._live.send(frame.model_dump_json())

    async def receive(self) -> BusMessage:
        """Block until a message arrives on a subscribed topic and return it."""
        while True:
            raw = await self._live.recv()
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            frame = _FRAME_ADAPTER.validate_json(text)
            if frame.message is not None:
                return frame.message
