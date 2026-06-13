"""Minimal async topic pub/sub broker over WebSocket.

The broker routes frames by topic and is intentionally ignorant of payload
schemas: it reads only ``op`` and ``topic`` and forwards the original frame
text to subscribers. This keeps the brain<->body message contract decoupled
from transport.
"""

from __future__ import annotations

import json
from collections import defaultdict
from types import TracebackType

from websockets.asyncio.server import Server, ServerConnection, serve


class Broker:
    """In-process WebSocket pub/sub broker.

    Use as an async context manager::

        async with Broker(host, port) as broker:
            ...  # broker is serving on broker.host:broker.port
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._subscribers: dict[str, set[ServerConnection]] = defaultdict(set)
        self._server: Server | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    async def __aenter__(self) -> Broker:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        """Begin serving. If port 0 was given, ``self.port`` is updated."""
        self._server = await serve(self._handle, self._host, self._port)
        # Resolve the actual bound port (supports port=0 for tests).
        for sock in self._server.sockets:
            self._port = sock.getsockname()[1]
            break

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle(self, conn: ServerConnection) -> None:
        topics: set[str] = set()
        try:
            async for raw in conn:
                text = raw if isinstance(raw, str) else raw.decode("utf-8")
                envelope = json.loads(text)
                op = envelope.get("op")
                topic = envelope.get("topic")
                if not isinstance(topic, str):
                    continue
                if op == "subscribe":
                    self._subscribers[topic].add(conn)
                    topics.add(topic)
                elif op == "unsubscribe":
                    self._subscribers[topic].discard(conn)
                    topics.discard(topic)
                elif op == "publish":
                    await self._fan_out(topic, text)
        finally:
            for topic in topics:
                self._subscribers[topic].discard(conn)

    async def _fan_out(self, topic: str, text: str) -> None:
        for subscriber in list(self._subscribers.get(topic, set())):
            try:
                await subscriber.send(text)
            except Exception:
                # A dead subscriber should not stop delivery to the rest.
                self._subscribers[topic].discard(subscriber)
