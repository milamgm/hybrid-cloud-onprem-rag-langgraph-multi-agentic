"""Thin async publisher bindings for supported cloud and on-premise transports.

SDK clients are injected by the deployment composition root. This keeps the
domain and graph packages independent from broker-specific connection setup,
credentials, retries, and checkpoint stores.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

from src.events.contracts import EventEnvelope
from src.events.transport import EventDelivery, EventSubscription


def _payload(event: EventEnvelope[Any]) -> bytes:
    return event.model_dump_json().encode("utf-8")


class AzureEventHubsPublisher:
    def __init__(self, producer_client: Any) -> None:
        self._producer = producer_client

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        from azure.eventhub import EventData

        batch = await self._producer.create_batch(partition_key=key)
        batch.add(EventData(_payload(event)))
        await self._producer.send_batch(batch)


class KafkaPublisher:
    def __init__(self, producer: Any, topic: str) -> None:
        self._producer = producer
        self._topic = topic

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        await self._producer.send_and_wait(
            self._topic,
            value=_payload(event),
            key=key.encode("utf-8") if key else None,
        )


class AzureServiceBusPublisher:
    def __init__(self, sender: Any) -> None:
        self._sender = sender

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        from azure.servicebus import ServiceBusMessage

        arguments: dict[str, Any] = {
            "message_id": event.id,
            "subject": event.subject,
            "content_type": "application/json",
        }
        if key:
            arguments["session_id"] = key
        message = ServiceBusMessage(_payload(event), **arguments)
        await self._sender.send_messages(message)


class RabbitMQPublisher:
    def __init__(self, exchange: Any, routing_key: str) -> None:
        self._exchange = exchange
        self._routing_key = routing_key

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        del key
        from aio_pika import DeliveryMode, Message

        await self._exchange.publish(
            Message(
                _payload(event),
                message_id=event.id,
                content_type="application/json",
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=self._routing_key,
        )


class SqsPublisher:
    """Uses an injected async ``send_message`` callable to avoid SDK coupling."""

    def __init__(self, send_message: Callable[..., Awaitable[Any]]) -> None:
        self._send_message = send_message

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        arguments: dict[str, Any] = {"MessageBody": _payload(event).decode()}
        if key:
            arguments["MessageDeduplicationId"] = event.id
            arguments["MessageGroupId"] = key
        await self._send_message(**arguments)


class EventBridgePublisher:
    """Binding around an injected AWS ``put_events`` callable."""

    def __init__(self, put_events: Callable[..., Awaitable[Any]], event_bus_name: str) -> None:
        self._put_events = put_events
        self._event_bus_name = event_bus_name

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        del key
        detail = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
        await self._put_events(
            Entries=[
                {
                    "EventBusName": self._event_bus_name,
                    "Source": event.source,
                    "DetailType": event.type,
                    "Detail": detail,
                    "TraceHeader": event.traceparent or "",
                    "Resources": [event.subject],
                }
            ]
        )


class HttpEventPublisher:
    """Binding for Event Grid, EventBridge HTTP ingress, or Knative Broker."""

    def __init__(self, client: Any, endpoint: str) -> None:
        self._client = client
        self._endpoint = endpoint

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        del key
        response = await self._client.post(
            self._endpoint,
            content=_payload(event),
            headers={"content-type": "application/cloudevents+json"},
        )
        response.raise_for_status()


class NatsPublisher:
    def __init__(self, jetstream: Any, subject: str) -> None:
        self._jetstream = jetstream
        self._subject = subject

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        del key
        await self._jetstream.publish(self._subject, _payload(event), headers={"Nats-Msg-Id": event.id})


class IbmMqPublisher:
    """Binding around an injected MQ PUT callable configured with syncpoint."""

    def __init__(self, put_message: Callable[..., Any]) -> None:
        self._put_message = put_message

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        del key
        result = self._put_message(message_id=event.id, body=_payload(event), syncpoint=True)
        if inspect.isawaitable(result):
            await result


DeadLetterHandler = Callable[[EventEnvelope[Any], str], Awaitable[None]]


class _BrokerDelivery:
    def __init__(
        self,
        event: EventEnvelope[Any],
        *,
        ack_callback: Callable[[], Awaitable[None]],
        retry_callback: Callable[[], Awaitable[None]],
        dead_letter_handler: DeadLetterHandler,
        delivery_attempt: int = 1,
    ) -> None:
        self.event = event
        self.delivery_attempt = delivery_attempt
        self._ack_callback = ack_callback
        self._retry_callback = retry_callback
        self._dead_letter_handler = dead_letter_handler

    async def ack(self) -> None:
        await self._ack_callback()

    async def retry(self, reason: str) -> None:
        del reason
        await self._retry_callback()

    async def dead_letter(self, reason: str) -> None:
        await self._dead_letter_handler(self.event, reason)
        await self._ack_callback()


class KafkaSubscription:
    """Manual-commit aiokafka subscription with seek-on-retry semantics."""

    def __init__(self, consumer: Any, dead_letter_handler: DeadLetterHandler) -> None:
        self._consumer = consumer
        self._dead_letter_handler = dead_letter_handler
        self._active = False

    async def __aenter__(self) -> "KafkaSubscription":
        await self._consumer.start()
        self._active = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._active = False
        await self._consumer.stop()

    def __aiter__(self) -> "KafkaSubscription":
        return self

    async def __anext__(self) -> EventDelivery:
        if not self._active:
            raise StopAsyncIteration
        record = await self._consumer.getone()
        event = EventEnvelope[dict[str, Any]].model_validate_json(record.value)

        async def ack() -> None:
            from aiokafka import TopicPartition

            await self._consumer.commit({TopicPartition(record.topic, record.partition): record.offset + 1})

        async def retry() -> None:
            from aiokafka import TopicPartition

            self._consumer.seek(TopicPartition(record.topic, record.partition), record.offset)

        return _BrokerDelivery(
            event,
            ack_callback=ack,
            retry_callback=retry,
            dead_letter_handler=self._dead_letter_handler,
        )


class AzureEventHubsSubscription:
    """Event Processor-compatible source; checkpoint is the acknowledgement."""

    def __init__(self, consumer_client: Any, dead_letter_handler: DeadLetterHandler) -> None:
        self._client = consumer_client
        self._dead_letter_handler = dead_letter_handler
        self._queue: asyncio.Queue[_BrokerDelivery] = asyncio.Queue()
        self._receive_task: asyncio.Task[Any] | None = None
        self._active = False

    async def __aenter__(self) -> "AzureEventHubsSubscription":
        async def on_event(partition_context: Any, event: Any) -> None:
            envelope = EventEnvelope[dict[str, Any]].model_validate_json(event.body_as_str())

            async def ack() -> None:
                await partition_context.update_checkpoint(event)

            async def retry() -> None:
                # No checkpoint means the event remains replayable after a
                # processor restart; the outer consumer applies bounded retry.
                return None

            await self._queue.put(
                _BrokerDelivery(
                    envelope,
                    ack_callback=ack,
                    retry_callback=retry,
                    dead_letter_handler=self._dead_letter_handler,
                )
            )

        self._active = True
        self._receive_task = asyncio.create_task(self._client.receive(on_event=on_event))
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._active = False
        if self._receive_task is not None:
            self._receive_task.cancel()
            await asyncio.gather(self._receive_task, return_exceptions=True)
        await self._client.close()

    def __aiter__(self) -> "AzureEventHubsSubscription":
        return self

    async def __anext__(self) -> EventDelivery:
        if not self._active:
            raise StopAsyncIteration
        return await self._queue.get()
