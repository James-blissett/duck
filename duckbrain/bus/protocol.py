"""Wire protocol framing for the bus.

A :class:`Frame` is the envelope sent over the WebSocket. Clients publish and
subscribe by topic; the broker routes frames by topic without needing to
understand the payload, so the message schema can evolve independently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from duckbrain.bus.schemas import BusMessage


class Frame(BaseModel):
    """Envelope exchanged between a :class:`BusClient` and the broker."""

    op: Literal["subscribe", "unsubscribe", "publish"]
    topic: str
    message: BusMessage | None = None
