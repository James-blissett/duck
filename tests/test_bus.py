"""Bus integration tests.

Acceptance criterion (Spec 0): two processes exchange an ExpressionEvent over
the bus. We model the two processes as two independent WebSocket peers (a
publisher and a subscriber) connected to one broker — each ``BusClient`` is a
separate connection exercising the full serialize -> wire -> deserialize path,
exactly as two OS processes would.
"""

from __future__ import annotations

import asyncio

import pytest

from duckbrain.bus import Broker, BusClient, ExpressionEvent, ExpressionKind, UtteranceHeard


async def test_two_peers_exchange_expression_event() -> None:
    async with Broker(host="127.0.0.1", port=0) as broker:
        async with (
            BusClient("127.0.0.1", broker.port) as subscriber,
            BusClient("127.0.0.1", broker.port) as publisher,
        ):
            await subscriber.subscribe("expression")
            # Let the broker register the subscription before we publish.
            await asyncio.sleep(0.05)

            sent = ExpressionEvent(kind=ExpressionKind.HAPPY, intensity=0.7)
            await publisher.publish("expression", sent)

            received = await asyncio.wait_for(subscriber.receive(), timeout=5.0)

    assert isinstance(received, ExpressionEvent)
    assert received.kind is ExpressionKind.HAPPY
    assert received.intensity == pytest.approx(0.7)


async def test_subscriber_only_receives_subscribed_topic() -> None:
    async with Broker(host="127.0.0.1", port=0) as broker:
        async with (
            BusClient("127.0.0.1", broker.port) as subscriber,
            BusClient("127.0.0.1", broker.port) as publisher,
        ):
            await subscriber.subscribe("expression")
            await asyncio.sleep(0.05)

            # Published to a different topic: must not arrive.
            await publisher.publish(
                "utterance",
                UtteranceHeard(session_id="s1", text="ignored", confidence=0.5),
            )
            # The intended message on the subscribed topic.
            await publisher.publish(
                "expression",
                ExpressionEvent(kind=ExpressionKind.LISTENING, intensity=0.3),
            )

            received = await asyncio.wait_for(subscriber.receive(), timeout=5.0)

    assert isinstance(received, ExpressionEvent)
    assert received.kind is ExpressionKind.LISTENING
