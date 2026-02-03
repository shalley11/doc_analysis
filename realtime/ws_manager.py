"""
Enhanced WebSocket manager with Redis Pub/Sub integration.

Manages WebSocket connections and bridges them to Redis Pub/Sub events.
"""
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from doc_analysis.realtime.events import Event
from doc_analysis.realtime.event_subscriber import get_subscriber
from doc_analysis.logging_config import get_api_logger

logger = get_api_logger()


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.

    Integrates with Redis Pub/Sub to receive events from workers
    and forward them to connected WebSocket clients.

    Usage:
        ws_manager = WebSocketManager()

        # In FastAPI startup
        await ws_manager.start()

        # In WebSocket endpoint
        await ws_manager.connect_pdf(batch_id, websocket)
        # or
        await ws_manager.connect_summary(batch_id, websocket)
    """

    def __init__(self):
        """Initialize WebSocket manager."""
        # Track connections: batch_id -> set of websockets
        self._pdf_connections: Dict[str, Set[WebSocket]] = {}
        self._summary_connections: Dict[str, Set[WebSocket]] = {}

    async def start(self):
        """Start the WebSocket manager (starts Redis subscriber)."""
        subscriber = get_subscriber()
        await subscriber.start()
        logger.info("[WS_MANAGER] Started")

    async def stop(self):
        """Stop the WebSocket manager."""
        from doc_analysis.realtime.event_subscriber import stop_subscriber
        await stop_subscriber()
        logger.info("[WS_MANAGER] Stopped")

    async def _send_to_websocket(self, websocket: WebSocket, event: Event):
        """Send event to a single WebSocket."""
        try:
            await websocket.send_json(event.to_dict())
        except Exception as e:
            logger.warning(f"[WS_MANAGER] Failed to send to WebSocket: {e}")

    def _create_pdf_callback(self, batch_id: str):
        """Create callback for PDF events."""
        async def callback(event: Event):
            connections = self._pdf_connections.get(batch_id, set())
            logger.debug(
                f"[WS_MANAGER] PDF event | batch={batch_id} | "
                f"event={event.event_type.value} | clients={len(connections)}"
            )
            for ws in connections.copy():
                await self._send_to_websocket(ws, event)
        return callback

    def _create_summary_callback(self, batch_id: str):
        """Create callback for summary events."""
        async def callback(event: Event):
            connections = self._summary_connections.get(batch_id, set())
            logger.debug(
                f"[WS_MANAGER] Summary event | batch={batch_id} | "
                f"event={event.event_type.value} | clients={len(connections)}"
            )
            for ws in connections.copy():
                await self._send_to_websocket(ws, event)
        return callback

    async def connect_pdf(self, batch_id: str, websocket: WebSocket):
        """
        Connect a WebSocket for PDF processing updates.

        Args:
            batch_id: Batch identifier
            websocket: WebSocket connection
        """
        await websocket.accept()

        # Track connection
        if batch_id not in self._pdf_connections:
            self._pdf_connections[batch_id] = set()

            # Subscribe to Redis channel for this batch
            subscriber = get_subscriber()
            callback = self._create_pdf_callback(batch_id)
            await subscriber.subscribe_pdf(batch_id, callback)

        self._pdf_connections[batch_id].add(websocket)

        logger.info(
            f"[WS_MANAGER] PDF WebSocket connected | batch={batch_id} | "
            f"total_connections={len(self._pdf_connections[batch_id])}"
        )

    async def disconnect_pdf(self, batch_id: str, websocket: WebSocket):
        """Disconnect a WebSocket from PDF updates."""
        if batch_id in self._pdf_connections:
            self._pdf_connections[batch_id].discard(websocket)

            logger.info(
                f"[WS_MANAGER] PDF WebSocket disconnected | batch={batch_id} | "
                f"remaining={len(self._pdf_connections[batch_id])}"
            )

            # If no more connections, unsubscribe from Redis
            if not self._pdf_connections[batch_id]:
                del self._pdf_connections[batch_id]
                subscriber = get_subscriber()
                await subscriber.unsubscribe_pdf(batch_id)

    async def connect_summary(self, batch_id: str, websocket: WebSocket):
        """
        Connect a WebSocket for summarization updates.

        Args:
            batch_id: Batch identifier
            websocket: WebSocket connection
        """
        await websocket.accept()

        # Track connection
        if batch_id not in self._summary_connections:
            self._summary_connections[batch_id] = set()

            # Subscribe to Redis channel for this batch
            subscriber = get_subscriber()
            callback = self._create_summary_callback(batch_id)
            await subscriber.subscribe_summary(batch_id, callback)

        self._summary_connections[batch_id].add(websocket)

        logger.info(
            f"[WS_MANAGER] Summary WebSocket connected | batch={batch_id} | "
            f"total_connections={len(self._summary_connections[batch_id])}"
        )

    async def disconnect_summary(self, batch_id: str, websocket: WebSocket):
        """Disconnect a WebSocket from summary updates."""
        if batch_id in self._summary_connections:
            self._summary_connections[batch_id].discard(websocket)

            logger.info(
                f"[WS_MANAGER] Summary WebSocket disconnected | batch={batch_id} | "
                f"remaining={len(self._summary_connections[batch_id])}"
            )

            # If no more connections, unsubscribe from Redis
            if not self._summary_connections[batch_id]:
                del self._summary_connections[batch_id]
                subscriber = get_subscriber()
                await subscriber.unsubscribe_summary(batch_id)

    # Legacy methods for backward compatibility
    async def connect(self, batch_id: str, websocket: WebSocket):
        """Legacy method - connects to PDF updates."""
        await self.connect_pdf(batch_id, websocket)

    def disconnect(self, batch_id: str, websocket: WebSocket):
        """Legacy method - use disconnect_pdf instead."""
        import asyncio
        asyncio.create_task(self.disconnect_pdf(batch_id, websocket))

    async def broadcast(self, batch_id: str, message: dict):
        """
        Legacy method - broadcast message directly to PDF connections.

        For new code, use event publishing instead.
        """
        for ws in self._pdf_connections.get(batch_id, set()).copy():
            try:
                await ws.send_json(message)
            except Exception:
                pass

    def get_connection_stats(self) -> dict:
        """Get current connection statistics."""
        pdf_conns = sum(len(c) for c in self._pdf_connections.values())
        summary_conns = sum(len(c) for c in self._summary_connections.values())

        # Get all batch IDs with active connections
        all_batches = set(self._pdf_connections.keys()) | set(self._summary_connections.keys())

        return {
            "pdf_connections": pdf_conns,
            "summary_connections": summary_conns,
            "total_connections": pdf_conns + summary_conns,
            "batches_with_connections": list(all_batches)
        }
