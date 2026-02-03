"""
Event publisher for Redis Pub/Sub.

Used by workers and services to publish real-time events.
"""
import redis
from typing import Optional
from doc_analysis.realtime.events import Event
from doc_analysis.logging_config import get_api_logger

logger = get_api_logger()

# Redis channel prefixes
PDF_CHANNEL_PREFIX = "ws:pdf:"
SUMMARY_CHANNEL_PREFIX = "ws:summary:"


class EventPublisher:
    """
    Publishes events to Redis Pub/Sub channels.

    Usage:
        publisher = EventPublisher()

        # Publish PDF processing event
        publisher.publish_pdf_event(batch_id, event)

        # Publish summarization event
        publisher.publish_summary_event(batch_id, event)
    """

    _instance: Optional["EventPublisher"] = None

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize Redis connection for publishing."""
        self.redis_host = redis_host
        self.redis_port = redis_port
        self._redis: Optional[redis.Redis] = None

    @property
    def redis(self) -> redis.Redis:
        """Lazy Redis connection."""
        if self._redis is None:
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
        return self._redis

    @classmethod
    def get_instance(cls) -> "EventPublisher":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _publish(self, channel: str, event: Event) -> bool:
        """
        Publish event to Redis channel.

        Returns:
            True if published successfully, False otherwise
        """
        try:
            message = event.to_json()
            subscribers = self.redis.publish(channel, message)

            logger.debug(
                f"[PUBLISH] channel={channel} | event={event.event_type.value} | "
                f"batch={event.batch_id} | subscribers={subscribers}"
            )
            return True

        except redis.RedisError as e:
            logger.error(f"[PUBLISH] Redis error: {e}")
            return False
        except Exception as e:
            logger.error(f"[PUBLISH] Unexpected error: {e}", exc_info=True)
            return False

    def publish_pdf_event(self, batch_id: str, event: Event) -> bool:
        """
        Publish PDF processing event.

        Args:
            batch_id: Batch identifier
            event: Event to publish

        Returns:
            True if published successfully
        """
        channel = f"{PDF_CHANNEL_PREFIX}{batch_id}"
        return self._publish(channel, event)

    def publish_summary_event(self, batch_id: str, event: Event) -> bool:
        """
        Publish summarization event.

        Args:
            batch_id: Batch identifier
            event: Event to publish

        Returns:
            True if published successfully
        """
        channel = f"{SUMMARY_CHANNEL_PREFIX}{batch_id}"
        return self._publish(channel, event)

    def close(self):
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None


# Convenience functions for publishing events
_publisher: Optional[EventPublisher] = None


def get_publisher() -> EventPublisher:
    """Get the global event publisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher


def publish_pdf_event(batch_id: str, event: Event) -> bool:
    """Convenience function to publish PDF event."""
    return get_publisher().publish_pdf_event(batch_id, event)


def publish_summary_event(batch_id: str, event: Event) -> bool:
    """Convenience function to publish summary event."""
    return get_publisher().publish_summary_event(batch_id, event)
