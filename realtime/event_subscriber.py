"""
Event subscriber for Redis Pub/Sub.

Listens to Redis channels and pushes events to WebSocket clients.
Runs as background async tasks in the FastAPI application.
"""
import asyncio
import redis.asyncio as aioredis
from typing import Dict, Set, Optional, Callable, Awaitable
from doc_analysis.realtime.events import Event
from doc_analysis.logging_config import get_api_logger
from doc_analysis import config

logger = get_api_logger()

# Redis channel prefixes
PDF_CHANNEL_PREFIX = "ws:pdf:"
SUMMARY_CHANNEL_PREFIX = "ws:summary:"


class EventSubscriber:
    """
    Subscribes to Redis Pub/Sub channels and forwards events to callbacks.

    Usage:
        subscriber = EventSubscriber()

        # Register callback for a batch
        await subscriber.subscribe_pdf(batch_id, callback)
        await subscriber.subscribe_summary(batch_id, callback)

        # Unsubscribe when done
        await subscriber.unsubscribe_pdf(batch_id)
    """

    def __init__(self, redis_host: str = None, redis_port: int = None):
        """Initialize Redis connection for subscribing."""
        self.redis_host = redis_host or config.REDIS_HOST
        self.redis_port = redis_port or config.REDIS_PORT
        self._redis: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None

        # Track active subscriptions: channel -> set of callbacks
        self._callbacks: Dict[str, Set[Callable[[Event], Awaitable[None]]]] = {}

        # Background listener task
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False

    async def _get_redis(self) -> aioredis.Redis:
        """Get async Redis connection."""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
        return self._redis

    async def _get_pubsub(self) -> aioredis.client.PubSub:
        """Get PubSub instance."""
        if self._pubsub is None:
            redis = await self._get_redis()
            self._pubsub = redis.pubsub()
        return self._pubsub

    async def _listener_loop(self):
        """Background loop that listens for Redis messages."""
        logger.info("[SUBSCRIBER] Listener loop started")

        pubsub = await self._get_pubsub()

        while self._running:
            try:
                # Only get messages if we have active subscriptions
                if not self._callbacks:
                    await asyncio.sleep(1)
                    continue

                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )

                if message is None:
                    continue

                if message["type"] != "message":
                    continue

                channel = message["channel"]
                data = message["data"]

                # Parse event
                try:
                    event = Event.from_json(data)
                except Exception as e:
                    logger.warning(f"[SUBSCRIBER] Failed to parse event: {e}")
                    continue

                # Get callbacks for this channel
                callbacks = self._callbacks.get(channel, set())

                if not callbacks:
                    logger.debug(f"[SUBSCRIBER] No callbacks for channel: {channel}")
                    continue

                # Call all registered callbacks
                logger.debug(
                    f"[SUBSCRIBER] Dispatching event | channel={channel} | "
                    f"event={event.event_type.value} | callbacks={len(callbacks)}"
                )

                for callback in callbacks.copy():
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"[SUBSCRIBER] Callback error: {e}", exc_info=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SUBSCRIBER] Listener error: {e}", exc_info=True)
                await asyncio.sleep(1)

        logger.info("[SUBSCRIBER] Listener loop stopped")

    async def start(self):
        """Start the background listener."""
        if self._running:
            return

        self._running = True
        self._listener_task = asyncio.create_task(self._listener_loop())
        logger.info("[SUBSCRIBER] Started")

    async def stop(self):
        """Stop the background listener."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

        if self._redis:
            await self._redis.close()
            self._redis = None

        logger.info("[SUBSCRIBER] Stopped")

    async def _subscribe_channel(
        self,
        channel: str,
        callback: Callable[[Event], Awaitable[None]]
    ):
        """Subscribe to a Redis channel."""
        pubsub = await self._get_pubsub()

        # Add callback to tracking
        if channel not in self._callbacks:
            self._callbacks[channel] = set()
            await pubsub.subscribe(channel)
            logger.debug(f"[SUBSCRIBER] Subscribed to channel: {channel}")

        self._callbacks[channel].add(callback)
        logger.debug(f"[SUBSCRIBER] Added callback for channel: {channel}")

    async def _unsubscribe_channel(
        self,
        channel: str,
        callback: Optional[Callable[[Event], Awaitable[None]]] = None
    ):
        """Unsubscribe from a Redis channel."""
        if channel not in self._callbacks:
            return

        if callback:
            self._callbacks[channel].discard(callback)
            logger.debug(f"[SUBSCRIBER] Removed callback from channel: {channel}")

        # If no more callbacks, unsubscribe from Redis
        if not self._callbacks[channel]:
            del self._callbacks[channel]
            pubsub = await self._get_pubsub()
            await pubsub.unsubscribe(channel)
            logger.debug(f"[SUBSCRIBER] Unsubscribed from channel: {channel}")

    async def subscribe_pdf(
        self,
        batch_id: str,
        callback: Callable[[Event], Awaitable[None]]
    ):
        """Subscribe to PDF processing events for a batch."""
        channel = f"{PDF_CHANNEL_PREFIX}{batch_id}"
        await self._subscribe_channel(channel, callback)

    async def unsubscribe_pdf(
        self,
        batch_id: str,
        callback: Optional[Callable[[Event], Awaitable[None]]] = None
    ):
        """Unsubscribe from PDF processing events."""
        channel = f"{PDF_CHANNEL_PREFIX}{batch_id}"
        await self._unsubscribe_channel(channel, callback)

    async def subscribe_summary(
        self,
        batch_id: str,
        callback: Callable[[Event], Awaitable[None]]
    ):
        """Subscribe to summarization events for a batch."""
        channel = f"{SUMMARY_CHANNEL_PREFIX}{batch_id}"
        await self._subscribe_channel(channel, callback)

    async def unsubscribe_summary(
        self,
        batch_id: str,
        callback: Optional[Callable[[Event], Awaitable[None]]] = None
    ):
        """Unsubscribe from summarization events."""
        channel = f"{SUMMARY_CHANNEL_PREFIX}{batch_id}"
        await self._unsubscribe_channel(channel, callback)


# Global subscriber instance
_subscriber: Optional[EventSubscriber] = None


def get_subscriber() -> EventSubscriber:
    """Get the global event subscriber instance."""
    global _subscriber
    if _subscriber is None:
        _subscriber = EventSubscriber()
    return _subscriber


async def start_subscriber():
    """Start the global subscriber."""
    subscriber = get_subscriber()
    await subscriber.start()


async def stop_subscriber():
    """Stop the global subscriber."""
    global _subscriber
    if _subscriber:
        await _subscriber.stop()
        _subscriber = None
