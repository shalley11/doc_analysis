"""
Refinement Store - Redis-based storage for refinement cycle.

Simple approach:
- First request: Generate result, create request_id, store in Redis
- Subsequent requests: Get stored result + user feedback, refine, overwrite
- Only stores the LAST result (no history needed for proofreading/rephrasing)
"""
import json
import uuid
import redis
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REFINEMENT_TTL
from logging_config import get_llm_logger

logger = get_llm_logger()

# Key prefix for refinement sessions
REFINEMENT_KEY_PREFIX = "refine"


@dataclass
class RefinementData:
    """Data stored for each refinement session."""
    request_id: str
    task: str                      # summary, rephrase, translate, etc.
    current_result: str            # Latest result (overwritten each refinement/regeneration)
    original_text: str             # Original input text
    model: str
    user_id: Optional[str] = None
    refinement_count: int = 0      # Count of refinements (without original text)
    regeneration_count: int = 0    # Count of regenerations (with original text)
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefinementData":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "RefinementData":
        return cls.from_dict(json.loads(json_str))


class RefinementStore:
    """
    Redis-based store for refinement sessions.

    Stores only the latest result - overwrites on each refinement.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        ttl: int = REFINEMENT_TTL
    ):
        self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._ttl = ttl
        self._prefix = REFINEMENT_KEY_PREFIX
        logger.info(f"[REFINE_STORE] Initialized | host={host}:{port} | db={db} | ttl={ttl}s")

    def _key(self, request_id: str) -> str:
        """Generate Redis key for a request."""
        return f"{self._prefix}:{request_id}"

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    def create(
        self,
        task: str,
        result: str,
        original_text: str,
        model: str,
        user_id: Optional[str] = None
    ) -> RefinementData:
        """
        Create a new refinement session.

        Args:
            task: Task type (summary, rephrase, etc.)
            result: The generated result
            original_text: Original input text
            model: Model used
            user_id: Optional user identifier

        Returns:
            RefinementData with generated request_id
        """
        request_id = self.generate_request_id()

        data = RefinementData(
            request_id=request_id,
            task=task,
            current_result=result,
            original_text=original_text,
            model=model,
            user_id=user_id,
            refinement_count=0
        )

        key = self._key(request_id)
        self._redis.setex(key, self._ttl, data.to_json())

        user_info = f" | user_id={user_id}" if user_id else ""
        logger.info(f"[REFINE_STORE] Created | request_id={request_id} | task={task}{user_info}")

        return data

    def get(self, request_id: str) -> Optional[RefinementData]:
        """
        Get refinement data by request_id.

        Args:
            request_id: The request identifier

        Returns:
            RefinementData or None if not found/expired
        """
        key = self._key(request_id)
        json_str = self._redis.get(key)

        if json_str:
            data = RefinementData.from_json(json_str)
            logger.debug(f"[REFINE_STORE] Retrieved | request_id={request_id}")
            return data

        logger.debug(f"[REFINE_STORE] Not found | request_id={request_id}")
        return None

    def update(
        self,
        request_id: str,
        new_result: str,
        user_id: Optional[str] = None
    ) -> Optional[RefinementData]:
        """
        Update refinement with new result (overwrites previous).

        Args:
            request_id: The request identifier
            new_result: The new refined result
            user_id: Optional user identifier (for logging)

        Returns:
            Updated RefinementData or None if not found
        """
        data = self.get(request_id)
        if not data:
            return None

        # Update fields
        data.current_result = new_result
        data.refinement_count += 1
        data.updated_at = datetime.now().isoformat()

        # Update user_id if provided and different
        if user_id and data.user_id != user_id:
            data.user_id = user_id

        # Get remaining TTL and preserve it
        key = self._key(request_id)
        ttl = self._redis.ttl(key)
        if ttl > 0:
            self._redis.setex(key, ttl, data.to_json())
        else:
            self._redis.setex(key, self._ttl, data.to_json())

        user_info = f" | user_id={data.user_id}" if data.user_id else ""
        logger.info(
            f"[REFINE_STORE] Updated | request_id={request_id} | "
            f"refinement_count={data.refinement_count}{user_info}"
        )

        return data

    def update_regeneration(
        self,
        request_id: str,
        new_result: str,
        user_id: Optional[str] = None
    ) -> Optional[RefinementData]:
        """
        Update with regenerated result (uses original text).

        Args:
            request_id: The request identifier
            new_result: The new regenerated result
            user_id: Optional user identifier (for logging)

        Returns:
            Updated RefinementData or None if not found
        """
        data = self.get(request_id)
        if not data:
            return None

        # Update fields
        data.current_result = new_result
        data.regeneration_count += 1
        data.updated_at = datetime.now().isoformat()

        # Update user_id if provided and different
        if user_id and data.user_id != user_id:
            data.user_id = user_id

        # Get remaining TTL and preserve it
        key = self._key(request_id)
        ttl = self._redis.ttl(key)
        if ttl > 0:
            self._redis.setex(key, ttl, data.to_json())
        else:
            self._redis.setex(key, self._ttl, data.to_json())

        user_info = f" | user_id={data.user_id}" if data.user_id else ""
        logger.info(
            f"[REFINE_STORE] Regenerated | request_id={request_id} | "
            f"regeneration_count={data.regeneration_count}{user_info}"
        )

        return data

    def delete(self, request_id: str) -> bool:
        """
        Delete a refinement session.

        Args:
            request_id: The request identifier

        Returns:
            True if deleted, False if not found
        """
        key = self._key(request_id)
        result = self._redis.delete(key)

        if result:
            logger.info(f"[REFINE_STORE] Deleted | request_id={request_id}")
            return True

        logger.debug(f"[REFINE_STORE] Delete failed (not found) | request_id={request_id}")
        return False

    def extend_ttl(self, request_id: str, ttl: Optional[int] = None) -> bool:
        """
        Extend the TTL for a refinement session.

        Args:
            request_id: The request identifier
            ttl: New TTL in seconds (uses default if not specified)

        Returns:
            True if extended, False if not found
        """
        key = self._key(request_id)
        ttl = ttl or self._ttl

        if self._redis.exists(key):
            self._redis.expire(key, ttl)
            logger.debug(f"[REFINE_STORE] Extended TTL | request_id={request_id} | ttl={ttl}s")
            return True

        return False

    def get_ttl(self, request_id: str) -> int:
        """Get remaining TTL for a request."""
        key = self._key(request_id)
        return self._redis.ttl(key)

    def exists(self, request_id: str) -> bool:
        """Check if a request exists."""
        key = self._key(request_id)
        return self._redis.exists(key) > 0


# Global store instance
_store: Optional[RefinementStore] = None


def get_refinement_store() -> RefinementStore:
    """Get the global refinement store instance."""
    global _store
    if _store is None:
        _store = RefinementStore()
    return _store


def init_refinement_store(
    host: str = REDIS_HOST,
    port: int = REDIS_PORT,
    db: int = REDIS_DB,
    ttl: int = REFINEMENT_TTL
) -> RefinementStore:
    """Initialize the global refinement store with custom settings."""
    global _store
    _store = RefinementStore(host=host, port=port, db=db, ttl=ttl)
    return _store
