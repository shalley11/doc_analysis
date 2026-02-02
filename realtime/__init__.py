"""
Real-time updates module using Redis Pub/Sub and WebSocket.

Architecture:
    Workers/Services → EventPublisher → Redis Pub/Sub → EventSubscriber → WebSocketManager → Clients

Usage:
    # In workers - publish events
    from doc_analysis.realtime import publish_pdf_event, publish_summary_event
    from doc_analysis.realtime.events import pdf_started, summary_completed

    publish_pdf_event(batch_id, pdf_started(batch_id, 3, ["a.pdf", "b.pdf", "c.pdf"]))

    # In API - manage WebSocket connections
    from doc_analysis.realtime import WebSocketManager

    ws_manager = WebSocketManager()
    await ws_manager.start()
    await ws_manager.connect_pdf(batch_id, websocket)
"""

from doc_analysis.realtime.ws_manager import WebSocketManager
from doc_analysis.realtime.event_publisher import (
    EventPublisher,
    get_publisher,
    publish_pdf_event,
    publish_summary_event
)
from doc_analysis.realtime.event_subscriber import (
    EventSubscriber,
    get_subscriber,
    start_subscriber,
    stop_subscriber
)
from doc_analysis.realtime.events import (
    Event,
    EventType,
    # PDF events
    pdf_started,
    pdf_extracting,
    pdf_extracted,
    pdf_chunking,
    pdf_chunked,
    pdf_embedding,
    pdf_embedded,
    pdf_completed,
    pdf_failed,
    # Summary events
    summary_started,
    summary_cache_hit,
    summary_method_selected,
    summary_batch_started,
    summary_batch_completed,
    summary_reduce_started,
    summary_reduce_level,
    summary_llm_call_started,
    summary_llm_call_completed,
    summary_completed,
    summary_failed,
    # Multi-PDF events
    multi_pdf_started,
    multi_pdf_pdf_started,
    multi_pdf_pdf_completed,
    multi_pdf_combining,
    multi_pdf_completed
)

__all__ = [
    # Manager
    "WebSocketManager",

    # Publisher
    "EventPublisher",
    "get_publisher",
    "publish_pdf_event",
    "publish_summary_event",

    # Subscriber
    "EventSubscriber",
    "get_subscriber",
    "start_subscriber",
    "stop_subscriber",

    # Event types
    "Event",
    "EventType",

    # PDF event factories
    "pdf_started",
    "pdf_extracting",
    "pdf_extracted",
    "pdf_chunking",
    "pdf_chunked",
    "pdf_embedding",
    "pdf_embedded",
    "pdf_completed",
    "pdf_failed",

    # Summary event factories
    "summary_started",
    "summary_cache_hit",
    "summary_method_selected",
    "summary_batch_started",
    "summary_batch_completed",
    "summary_reduce_started",
    "summary_reduce_level",
    "summary_llm_call_started",
    "summary_llm_call_completed",
    "summary_completed",
    "summary_failed",

    # Multi-PDF event factories
    "multi_pdf_started",
    "multi_pdf_pdf_started",
    "multi_pdf_pdf_completed",
    "multi_pdf_combining",
    "multi_pdf_completed"
]
