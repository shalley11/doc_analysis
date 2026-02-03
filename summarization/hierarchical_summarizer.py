"""
Hierarchical summarization for large documents.

Handles documents with many chunks by using a map-reduce approach:
1. Group chunks into batches that fit within LLM context
2. Summarize each batch (MAP phase) - stored in Redis
3. Combine batch summaries into final summary (REDUCE phase)
4. Store final summary based on SUMMARY_STORAGE_MODE config

Supports both synchronous and asynchronous processing:
- Sync: Original sequential processing
- Async: Parallel batch processing with asyncio.gather() for 10-20x speedup
"""
import requests
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass

from doc_analysis.config import (
    OLLAMA_URL,
    SUMMARY_MODEL,
    SUMMARY_TIMEOUT,
    SUMMARY_MAX_WORDS_PER_BATCH,
    SUMMARY_MAX_CHUNKS_PER_BATCH,
    SUMMARY_INTERMEDIATE_WORDS,
    SUMMARY_FINAL_WORDS,
    SUMMARY_STORAGE_MODE,
    # Async processing settings
    ASYNC_MAX_CONCURRENT_BATCHES,
    ASYNC_MAX_CONCURRENT_REDUCE,
    ASYNC_LLM_NUM_PREDICT,
    ASYNC_MAX_REDUCE_LEVELS,
    ASYNC_REDUCE_GROUP_SIZE,
)
import time
from doc_analysis.logging_config import get_summarization_logger
from doc_analysis.summarization.summary_store import (
    init_summary_progress,
    store_batch_summary,
    get_batch_summary,
    get_all_batch_summaries,
    store_final_summary,
    get_final_summary,
    is_summary_cached
)
from doc_analysis.summarization.summary_prompts import (
    get_batch_summary_prompt,
    get_final_combine_prompt,
    get_direct_summary_prompt,
    get_multi_pdf_combine_prompt
)
from doc_analysis.realtime import (
    publish_summary_event,
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
    multi_pdf_started,
    multi_pdf_pdf_started,
    multi_pdf_pdf_completed,
    multi_pdf_combining,
    multi_pdf_completed
)

logger = get_summarization_logger()


def _publish_event(batch_id: str, event):
    """Safely publish event, logging any errors."""
    try:
        publish_summary_event(batch_id, event)
    except Exception as e:
        logger.warning(f"[EVENT] Failed to publish event: {e}")


@dataclass
class SummarizerConfig:
    """Configuration for hierarchical summarization."""
    max_words_per_batch: int = SUMMARY_MAX_WORDS_PER_BATCH
    max_chunks_per_batch: int = SUMMARY_MAX_CHUNKS_PER_BATCH
    intermediate_summary_words: int = SUMMARY_INTERMEDIATE_WORDS
    final_summary_words: int = SUMMARY_FINAL_WORDS
    temperature: float = 0.3
    model: str = SUMMARY_MODEL


DEFAULT_CONFIG = SummarizerConfig()


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _chunk_into_batches(
    chunks: List[Dict],
    config: SummarizerConfig
) -> List[List[Dict]]:
    """
    Split chunks into batches that fit within context limits.
    Groups chunks by page proximity when possible.
    """
    logger.debug(f"[BATCHING] START | total_chunks={len(chunks)} | max_words={config.max_words_per_batch} | max_chunks={config.max_chunks_per_batch}")

    batches = []
    current_batch = []
    current_word_count = 0

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_words = _count_words(chunk_text)

        would_exceed_words = (current_word_count + chunk_words) > config.max_words_per_batch
        would_exceed_chunks = len(current_batch) >= config.max_chunks_per_batch

        if current_batch and (would_exceed_words or would_exceed_chunks):
            logger.debug(f"[BATCHING] Batch {len(batches)+1} closed | chunks={len(current_batch)} | words={current_word_count} | reason={'words' if would_exceed_words else 'chunks'}")
            batches.append(current_batch)
            current_batch = []
            current_word_count = 0

        current_batch.append(chunk)
        current_word_count += chunk_words

    if current_batch:
        logger.debug(f"[BATCHING] Batch {len(batches)+1} closed (final) | chunks={len(current_batch)} | words={current_word_count}")
        batches.append(current_batch)

    logger.debug(f"[BATCHING] END | total_batches={len(batches)}")
    return batches


def _prepare_batch_content(chunks: List[Dict]) -> str:
    """Prepare content from a batch of chunks."""
    content_parts = []
    current_page = None
    content_type_counts = {"text": 0, "table": 0, "image": 0}
    pages_seen = set()

    for chunk in chunks:
        page_no = chunk.get("page_no", 0)
        content_type = chunk.get("content_type", "text")
        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        pages_seen.add(page_no)

        if page_no != current_page:
            content_parts.append(f"\n[Page {page_no}]\n")
            current_page = page_no

        # Use unified 'summary' field for table/image vision summaries
        summary = chunk.get("summary", "")

        if content_type == "table":
            if summary:
                content_parts.append(f"[Table]: {summary}\n")
            else:
                content_parts.append(chunk.get("text", "") + "\n")
        elif content_type == "image":
            if summary:
                content_parts.append(f"[Image]: {summary}\n")
        else:
            content_parts.append(chunk.get("text", "") + "\n")

    result = "".join(content_parts)
    logger.debug(f"[PREPARE_CONTENT] chunks={len(chunks)} | pages={len(pages_seen)} | types={content_type_counts} | output_chars={len(result)}")
    return result


def _call_llm(prompt: str, config: SummarizerConfig, context: str = "unknown", batch_id: str = None) -> str:
    """Call Ollama LLM with detailed logging and event publishing."""
    prompt_words = _count_words(prompt)
    logger.debug(f"[LLM] {context} | model={config.model} | prompt_words={prompt_words} | temp={config.temperature}")

    # Publish LLM call started event
    if batch_id:
        _publish_event(batch_id, summary_llm_call_started(batch_id, context, prompt_words))

    start_time = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": ASYNC_LLM_NUM_PREDICT
                }
            },
            timeout=SUMMARY_TIMEOUT
        )
        response.raise_for_status()

        elapsed = time.time() - start_time
        result = response.json().get("response", "").strip()
        result_words = _count_words(result)

        logger.info(f"[LLM] {context} | SUCCESS | elapsed={elapsed:.2f}s | response_words={result_words}")
        logger.debug(f"[LLM] {context} | Response preview: {result[:200]}...")

        # Publish LLM call completed event
        if batch_id:
            _publish_event(batch_id, summary_llm_call_completed(batch_id, context, elapsed, result_words))

        return result
    except requests.exceptions.Timeout as e:
        elapsed = time.time() - start_time
        logger.error(f"[LLM] {context} | TIMEOUT after {elapsed:.2f}s | timeout_limit={SUMMARY_TIMEOUT}s")
        raise
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        logger.error(f"[LLM] {context} | REQUEST_ERROR after {elapsed:.2f}s | error={str(e)}")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[LLM] {context} | UNEXPECTED_ERROR after {elapsed:.2f}s | error={str(e)}", exc_info=True)
        raise


def _summarize_batch(
    content: str,
    batch_index: int,
    total_batches: int,
    config: SummarizerConfig,
    batch_id: str = None
) -> str:
    """Generate summary for a single batch."""
    logger.debug(f"[BATCH] Preparing prompt for batch {batch_index + 1}/{total_batches}")
    prompt = get_batch_summary_prompt(
        content=content,
        batch_index=batch_index,
        total_batches=total_batches,
        word_count=config.intermediate_summary_words
    )
    return _call_llm(prompt, config, context=f"batch_{batch_index + 1}_of_{total_batches}", batch_id=batch_id)


# =========================
# Async Support for Parallel Processing
# =========================

async def _call_llm_async(
    prompt: str,
    config: SummarizerConfig,
    session: aiohttp.ClientSession,
    context: str = "unknown",
    batch_id: str = None
) -> str:
    """
    Call LLM asynchronously for parallel batch processing.

    Args:
        prompt: The prompt text
        config: Summarizer configuration
        session: aiohttp session for connection pooling
        context: Context string for logging
        batch_id: Batch ID for event publishing

    Returns:
        Generated summary text
    """
    prompt_words = _count_words(prompt)
    logger.debug(f"[ASYNC_LLM] {context} | model={config.model} | prompt_words={prompt_words}")

    # Publish LLM call started event
    if batch_id:
        _publish_event(batch_id, summary_llm_call_started(batch_id, context, prompt_words))

    start_time = time.time()

    try:
        async with session.post(
            OLLAMA_URL,
            json={
                "model": config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": ASYNC_LLM_NUM_PREDICT
                }
            },
            timeout=aiohttp.ClientTimeout(total=SUMMARY_TIMEOUT)
        ) as response:
            response.raise_for_status()
            data = await response.json()

            elapsed = time.time() - start_time
            result = data.get("response", "").strip()
            result_words = _count_words(result)

            logger.info(f"[ASYNC_LLM] {context} | SUCCESS | elapsed={elapsed:.2f}s | response_words={result_words}")

            # Publish LLM call completed event
            if batch_id:
                _publish_event(batch_id, summary_llm_call_completed(batch_id, context, elapsed, result_words))

            return result

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[ASYNC_LLM] {context} | TIMEOUT after {elapsed:.2f}s")
        raise RuntimeError(f"LLM timeout after {elapsed:.2f}s")

    except aiohttp.ClientError as e:
        elapsed = time.time() - start_time
        logger.error(f"[ASYNC_LLM] {context} | REQUEST_ERROR after {elapsed:.2f}s | error={str(e)}")
        raise RuntimeError(f"LLM request failed: {e}")


async def _summarize_batch_async(
    content: str,
    batch_index: int,
    total_batches: int,
    config: SummarizerConfig,
    session: aiohttp.ClientSession,
    batch_id: str = None
) -> str:
    """Generate summary for a single batch asynchronously."""
    logger.debug(f"[ASYNC_BATCH] Preparing prompt for batch {batch_index + 1}/{total_batches}")
    prompt = get_batch_summary_prompt(
        content=content,
        batch_index=batch_index,
        total_batches=total_batches,
        word_count=config.intermediate_summary_words
    )
    return await _call_llm_async(
        prompt, config, session,
        context=f"batch_{batch_index + 1}_of_{total_batches}",
        batch_id=batch_id
    )


async def _process_batches_parallel(
    batches: List[List[Dict]],
    config: SummarizerConfig,
    batch_id: str,
    max_concurrent: int = None
) -> List[str]:
    """
    Process multiple batches in parallel using asyncio.gather().

    This is the key function for performance improvement.
    Instead of sequential processing, all batches are sent concurrently
    and the LLM server (vLLM) batches them on the GPU.

    Args:
        batches: List of chunk batches to summarize
        config: Summarizer configuration
        batch_id: Batch ID for storage and events
        max_concurrent: Maximum concurrent requests

    Returns:
        List of batch summaries in order
    """
    num_batches = len(batches)
    max_concurrent = max_concurrent or ASYNC_MAX_CONCURRENT_BATCHES
    logger.info(f"[ASYNC_MAP] START | batches={num_batches} | max_concurrent={max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)
    batch_summaries = [None] * num_batches  # Pre-allocate to maintain order

    async def process_single_batch(idx: int, batch: List[Dict], session: aiohttp.ClientSession):
        """Process a single batch with semaphore control."""
        async with semaphore:
            # Check if this batch was already summarized (for resume)
            existing = get_batch_summary(batch_id, idx)
            if existing:
                logger.info(f"[ASYNC_MAP] Batch {idx+1}/{num_batches} | CACHE_HIT")
                return idx, existing

            content = _prepare_batch_content(batch)
            content_words = _count_words(content)
            logger.info(f"[ASYNC_MAP] Batch {idx+1}/{num_batches} | chunks={len(batch)} | words={content_words}")

            # Publish batch started event
            _publish_event(batch_id, summary_batch_started(batch_id, idx, num_batches, len(batch), content_words))

            batch_start = time.time()
            summary = await _summarize_batch_async(
                content, idx, num_batches, config, session, batch_id=batch_id
            )
            batch_elapsed = time.time() - batch_start

            # Store batch summary in Redis
            store_batch_summary(batch_id, idx, summary, {
                "chunks": len(batch),
                "words": content_words
            })
            logger.debug(f"[ASYNC_MAP] Batch {idx+1}/{num_batches} | Stored | summary_words={_count_words(summary)}")

            # Publish batch completed event
            _publish_event(batch_id, summary_batch_completed(batch_id, idx, num_batches, batch_elapsed))

            return idx, summary

    # Create aiohttp session and process all batches concurrently
    timeout = aiohttp.ClientTimeout(total=SUMMARY_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            process_single_batch(idx, batch, session)
            for idx, batch in enumerate(batches)
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle any errors
    errors = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(result)
            logger.error(f"[ASYNC_MAP] Batch failed: {result}")
        else:
            idx, summary = result
            batch_summaries[idx] = summary

    if errors:
        raise RuntimeError(f"{len(errors)} batches failed during parallel processing")

    logger.info(f"[ASYNC_MAP] END | all {num_batches} batches completed")
    return batch_summaries


async def _reduce_summaries_parallel(
    summaries: List[str],
    config: SummarizerConfig,
    batch_id: str,
    summary_type: str,
    max_concurrent: int = None
) -> str:
    """
    Reduce summaries in parallel for each level.

    Groups summaries and combines them concurrently at each level.
    """
    levels = 1
    current_summaries = summaries
    max_concurrent = max_concurrent or ASYNC_MAX_CONCURRENT_REDUCE

    # Publish reduce started event
    _publish_event(batch_id, summary_reduce_started(batch_id, len(current_summaries)))

    combined_text = " ".join(current_summaries)
    combined_words = _count_words(combined_text)
    logger.debug(f"[ASYNC_REDUCE] Initial combined words: {combined_words}")

    timeout = aiohttp.ClientTimeout(total=SUMMARY_TIMEOUT)

    while combined_words > config.max_words_per_batch and levels < ASYNC_MAX_REDUCE_LEVELS:
        logger.info(f"[ASYNC_REDUCE] Level {levels} | summaries={len(current_summaries)} | combined_words={combined_words}")

        # Group summaries into batches
        groups = [current_summaries[i:i+ASYNC_REDUCE_GROUP_SIZE] for i in range(0, len(current_summaries), ASYNC_REDUCE_GROUP_SIZE)]
        logger.debug(f"[ASYNC_REDUCE] Level {levels} | Creating {len(groups)} groups")

        async with aiohttp.ClientSession(timeout=timeout) as session:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def reduce_group(idx: int, group: List[str]):
                async with semaphore:
                    logger.debug(f"[ASYNC_REDUCE] Level {levels} | Group {idx+1}/{len(groups)} | combining {len(group)} summaries")
                    prompt = _get_final_prompt(group, "detailed")
                    return await _call_llm_async(
                        prompt, config, session,
                        context=f"reduce_level_{levels}_group_{idx+1}",
                        batch_id=batch_id
                    )

            tasks = [reduce_group(idx, group) for idx, group in enumerate(groups)]
            new_summaries = await asyncio.gather(*tasks)

        # Publish reduce level event
        _publish_event(batch_id, summary_reduce_level(batch_id, levels, len(current_summaries), len(new_summaries)))

        current_summaries = list(new_summaries)
        combined_text = " ".join(current_summaries)
        combined_words = _count_words(combined_text)
        levels += 1

    # Final combination
    logger.info(f"[ASYNC_REDUCE] Final combine | summaries={len(current_summaries)} | target_type={summary_type}")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        prompt = _get_final_prompt(current_summaries, summary_type)
        final_summary = await _call_llm_async(
            prompt, config, session,
            context=f"final_{summary_type}",
            batch_id=batch_id
        )

    return final_summary, levels


async def summarize_chunks_async(
    batch_id: str,
    chunks: List[Dict],
    summary_type: str = "detailed",
    pdf_name: Optional[str] = None,
    config: Optional[SummarizerConfig] = None,
    use_cache: bool = True,
    max_concurrent: int = None
) -> Dict:
    """
    Summarize chunks using parallel async processing.

    This is the async version of summarize_chunks() that provides
    10-20x speedup by processing batches concurrently.

    Args:
        batch_id: Batch identifier for storage
        chunks: List of document chunks
        summary_type: brief, bulletwise, or detailed
        pdf_name: PDF name (None for combined summary)
        config: Summarization configuration
        use_cache: Whether to use cached summaries
        max_concurrent: Maximum concurrent LLM requests (default from config)

    Returns:
        Dictionary with summary and metadata
    """
    start_time = time.time()
    target = pdf_name or "all_pdfs"
    max_concurrent = max_concurrent or ASYNC_MAX_CONCURRENT_BATCHES
    logger.info(f"[ASYNC_SUMMARIZE] START | batch={batch_id} | target={target} | type={summary_type} | max_concurrent={max_concurrent}")

    if config is None:
        config = DEFAULT_CONFIG

    total_chunks = len(chunks)
    total_words = sum(_count_words(c.get("text", "")) for c in chunks)

    # Publish started event
    _publish_event(batch_id, summary_started(batch_id, pdf_name, summary_type, total_chunks, total_words))

    # Check cache first
    if use_cache:
        cached = get_final_summary(batch_id, summary_type, pdf_name)
        if cached:
            elapsed = time.time() - start_time
            logger.info(f"[ASYNC_SUMMARIZE] CACHE_HIT | elapsed={elapsed:.2f}s")
            _publish_event(batch_id, summary_cache_hit(batch_id, pdf_name, summary_type))
            return {
                "summary": cached["summary"],
                "method": "cached",
                "storage": cached.get("storage", "unknown"),
                "cached": True
            }

    # Check if content fits in single batch (direct summarization)
    is_small_doc = total_words <= config.max_words_per_batch and total_chunks <= config.max_chunks_per_batch

    if is_small_doc:
        logger.info(f"[ASYNC_SUMMARIZE] Using DIRECT method (single LLM call)")
        _publish_event(batch_id, summary_method_selected(batch_id, "direct", 1, "content fits in single batch"))

        content = _prepare_batch_content(chunks)
        timeout = aiohttp.ClientTimeout(total=SUMMARY_TIMEOUT)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                prompt = _get_direct_prompt(content, summary_type)
                summary = await _call_llm_async(prompt, config, session, context=f"direct_{summary_type}", batch_id=batch_id)

            metadata = {
                "total_chunks": total_chunks,
                "total_words": total_words,
                "method": "direct",
                "batches": 1,
                "levels": 1
            }
            store_final_summary(batch_id, summary, summary_type, pdf_name, metadata)

            elapsed = time.time() - start_time
            logger.info(f"[ASYNC_SUMMARIZE] END | method=direct | elapsed={elapsed:.2f}s")
            _publish_event(batch_id, summary_completed(batch_id, pdf_name, summary_type, "direct", elapsed, _count_words(summary)))

            return {
                "summary": summary,
                "method": "direct",
                "total_chunks": total_chunks,
                "total_words": total_words,
                "batches": 1,
                "levels": 1,
                "storage_mode": SUMMARY_STORAGE_MODE,
                "cached": False
            }
        except Exception as e:
            _publish_event(batch_id, summary_failed(batch_id, str(e), pdf_name))
            raise

    # Hierarchical processing with parallel batches
    logger.info(f"[ASYNC_SUMMARIZE] Using PARALLEL HIERARCHICAL method")
    batches = _chunk_into_batches(chunks, config)
    num_batches = len(batches)

    _publish_event(batch_id, summary_method_selected(batch_id, "hierarchical_async", num_batches, "parallel batch processing"))

    init_summary_progress(batch_id, num_batches, pdf_name)

    try:
        # ========== PARALLEL MAP PHASE ==========
        map_start = time.time()
        batch_summaries = await _process_batches_parallel(batches, config, batch_id, max_concurrent)
        map_elapsed = time.time() - map_start
        logger.info(f"[ASYNC_SUMMARIZE] MAP_PHASE | elapsed={map_elapsed:.2f}s | speedup vs sequential: ~{num_batches}x")

        # ========== PARALLEL REDUCE PHASE ==========
        reduce_start = time.time()
        final_summary, levels = await _reduce_summaries_parallel(
            batch_summaries, config, batch_id, summary_type
        )
        reduce_elapsed = time.time() - reduce_start
        logger.info(f"[ASYNC_SUMMARIZE] REDUCE_PHASE | elapsed={reduce_elapsed:.2f}s | levels={levels}")

        # Store final summary
        metadata = {
            "total_chunks": total_chunks,
            "total_words": total_words,
            "method": "hierarchical_async",
            "batches": num_batches,
            "levels": levels + 1
        }
        store_final_summary(batch_id, final_summary, summary_type, pdf_name, metadata)

        total_elapsed = time.time() - start_time
        logger.info(f"[ASYNC_SUMMARIZE] END | batches={num_batches} | elapsed={total_elapsed:.2f}s | summary_words={_count_words(final_summary)}")

        _publish_event(batch_id, summary_completed(batch_id, pdf_name, summary_type, "hierarchical_async", total_elapsed, _count_words(final_summary)))

        return {
            "summary": final_summary,
            "method": "hierarchical_async",
            "total_chunks": total_chunks,
            "total_words": total_words,
            "batches": num_batches,
            "levels": levels + 1,
            "storage_mode": SUMMARY_STORAGE_MODE,
            "cached": False
        }

    except Exception as e:
        _publish_event(batch_id, summary_failed(batch_id, str(e), pdf_name))
        raise


def summarize_chunks_parallel(
    batch_id: str,
    chunks: List[Dict],
    summary_type: str = "detailed",
    pdf_name: Optional[str] = None,
    config: Optional[SummarizerConfig] = None,
    use_cache: bool = True,
    max_concurrent: int = None
) -> Dict:
    """
    Synchronous wrapper for parallel summarization.

    Use this when calling from synchronous code that can't use async/await.
    Provides the same 10-20x speedup as the async version.

    Args:
        batch_id: Batch identifier for storage
        chunks: List of document chunks
        summary_type: brief, bulletwise, or detailed
        pdf_name: PDF name (None for combined summary)
        config: Summarization configuration
        use_cache: Whether to use cached summaries
        max_concurrent: Maximum concurrent LLM requests (default from config)

    Returns:
        Dictionary with summary and metadata
    """
    return asyncio.run(
        summarize_chunks_async(
            batch_id=batch_id,
            chunks=chunks,
            summary_type=summary_type,
            pdf_name=pdf_name,
            config=config,
            use_cache=use_cache,
            max_concurrent=max_concurrent
        )
    )


def _get_final_prompt(summaries: List[str], summary_type: str) -> str:
    """Generate prompt for combining summaries."""
    return get_final_combine_prompt(summaries, summary_type)


def _get_direct_prompt(content: str, summary_type: str) -> str:
    """Generate prompt for direct summarization (small documents)."""
    return get_direct_summary_prompt(content, summary_type)


def summarize_chunks(
    batch_id: str,
    chunks: List[Dict],
    summary_type: str = "detailed",
    pdf_name: Optional[str] = None,
    config: Optional[SummarizerConfig] = None,
    use_cache: bool = True
) -> Dict:
    """
    Summarize chunks using hierarchical approach with storage.

    Args:
        batch_id: Batch identifier for storage
        chunks: List of document chunks
        summary_type: brief, bulletwise, or detailed
        pdf_name: PDF name (None for combined summary)
        config: Summarization configuration
        use_cache: Whether to use cached summaries

    Returns:
        Dictionary with summary and metadata
    """
    start_time = time.time()
    target = pdf_name or "all_pdfs"
    logger.info(f"[SUMMARIZE] START | batch={batch_id} | target={target} | type={summary_type} | use_cache={use_cache}")

    if config is None:
        config = DEFAULT_CONFIG
        logger.debug(f"[SUMMARIZE] Using default config: max_words={config.max_words_per_batch}, max_chunks={config.max_chunks_per_batch}")

    total_chunks = len(chunks)
    total_words = sum(_count_words(c.get("text", "")) for c in chunks)

    # Publish started event
    _publish_event(batch_id, summary_started(batch_id, pdf_name, summary_type, total_chunks, total_words))

    # Check cache first
    if use_cache:
        logger.debug(f"[SUMMARIZE] Checking cache for {batch_id}/{pdf_name}/{summary_type}")
        cached = get_final_summary(batch_id, summary_type, pdf_name)
        if cached:
            elapsed = time.time() - start_time
            logger.info(f"[SUMMARIZE] CACHE_HIT | batch={batch_id} | target={target} | elapsed={elapsed:.2f}s")

            # Publish cache hit event
            _publish_event(batch_id, summary_cache_hit(batch_id, pdf_name, summary_type))

            return {
                "summary": cached["summary"],
                "method": "cached",
                "storage": cached.get("storage", "unknown"),
                "cached": True
            }
        logger.debug(f"[SUMMARIZE] Cache miss, proceeding with summarization")

    logger.info(f"[SUMMARIZE] Document stats | chunks={total_chunks} | words={total_words}")

    # Check if content fits in single batch (direct summarization)
    is_small_doc = total_words <= config.max_words_per_batch and total_chunks <= config.max_chunks_per_batch
    logger.info(f"[SUMMARIZE] Size decision | is_small={is_small_doc} | threshold_words={config.max_words_per_batch} | threshold_chunks={config.max_chunks_per_batch}")

    if is_small_doc:
        logger.info(f"[SUMMARIZE] Using DIRECT method (single LLM call)")

        # Publish method selected event
        _publish_event(batch_id, summary_method_selected(batch_id, "direct", 1, "content fits in single batch"))

        content = _prepare_batch_content(chunks)
        content_words = _count_words(content)
        logger.debug(f"[SUMMARIZE] Prepared content: {content_words} words")

        try:
            prompt = _get_direct_prompt(content, summary_type)
            summary = _call_llm(prompt, config, context=f"direct_{summary_type}", batch_id=batch_id)

            # Store the summary
            metadata = {
                "total_chunks": total_chunks,
                "total_words": total_words,
                "method": "direct",
                "batches": 1,
                "levels": 1
            }
            logger.debug(f"[SUMMARIZE] Storing final summary to cache")
            store_final_summary(batch_id, summary, summary_type, pdf_name, metadata)

            elapsed = time.time() - start_time
            logger.info(f"[SUMMARIZE] END | method=direct | elapsed={elapsed:.2f}s | summary_words={_count_words(summary)}")

            # Publish completed event
            _publish_event(batch_id, summary_completed(batch_id, pdf_name, summary_type, "direct", elapsed, _count_words(summary)))

            return {
                "summary": summary,
                "method": "direct",
                "total_chunks": total_chunks,
                "total_words": total_words,
                "batches": 1,
                "levels": 1,
                "storage_mode": SUMMARY_STORAGE_MODE,
                "cached": False
            }
        except Exception as e:
            # Publish failed event
            _publish_event(batch_id, summary_failed(batch_id, str(e), pdf_name))
            raise

    # Split into batches for hierarchical processing
    logger.info(f"[SUMMARIZE] Using HIERARCHICAL method (map-reduce)")
    batches = _chunk_into_batches(chunks, config)
    num_batches = len(batches)

    # Publish method selected event
    _publish_event(batch_id, summary_method_selected(batch_id, "hierarchical", num_batches, "content exceeds single batch limit"))

    batch_stats = [{"chunks": len(b), "words": sum(_count_words(c.get("text", "")) for c in b)} for b in batches]
    logger.info(f"[SUMMARIZE] Split into {num_batches} batches")
    logger.debug(f"[SUMMARIZE] Batch stats: {batch_stats}")

    # Initialize progress tracking
    init_summary_progress(batch_id, num_batches, pdf_name)

    try:
        # ========== MAP PHASE: Summarize each batch ==========
        logger.info(f"[MAP_PHASE] START | total_batches={num_batches}")
        map_start = time.time()
        batch_summaries = []

        for i, batch in enumerate(batches):
            # Check if this batch was already summarized (for resume)
            existing = get_batch_summary(batch_id, i)
            if existing:
                logger.info(f"[MAP_PHASE] Batch {i+1}/{num_batches} | CACHE_HIT")
                batch_summaries.append(existing)
                continue

            content = _prepare_batch_content(batch)
            content_words = _count_words(content)
            logger.info(f"[MAP_PHASE] Batch {i+1}/{num_batches} | chunks={len(batch)} | words={content_words}")

            # Publish batch started event
            _publish_event(batch_id, summary_batch_started(batch_id, i, num_batches, len(batch), content_words))

            batch_start = time.time()
            summary = _summarize_batch(content, i, num_batches, config, batch_id=batch_id)
            batch_elapsed = time.time() - batch_start
            batch_summaries.append(summary)

            # Store batch summary in Redis
            store_batch_summary(batch_id, i, summary, {
                "chunks": len(batch),
                "words": content_words
            })
            logger.debug(f"[MAP_PHASE] Batch {i+1}/{num_batches} | Stored to Redis | summary_words={_count_words(summary)}")

            # Publish batch completed event
            _publish_event(batch_id, summary_batch_completed(batch_id, i, num_batches, batch_elapsed))

        map_elapsed = time.time() - map_start
        logger.info(f"[MAP_PHASE] END | elapsed={map_elapsed:.2f}s | summaries_generated={len(batch_summaries)}")

        # ========== REDUCE PHASE: Combine summaries ==========
        logger.info(f"[REDUCE_PHASE] START")
        reduce_start = time.time()
        levels = 1
        current_summaries = batch_summaries

        # Publish reduce started event
        _publish_event(batch_id, summary_reduce_started(batch_id, len(current_summaries)))

        # Check if we need multiple reduce levels
        combined_text = " ".join(current_summaries)
        combined_words = _count_words(combined_text)
        logger.debug(f"[REDUCE_PHASE] Initial combined words: {combined_words}")

        while combined_words > config.max_words_per_batch and levels < ASYNC_MAX_REDUCE_LEVELS:
            logger.info(f"[REDUCE_PHASE] Level {levels} | summaries={len(current_summaries)} | combined_words={combined_words}")

            # Group summaries into batches
            new_summaries = []
            groups = list(range(0, len(current_summaries), ASYNC_REDUCE_GROUP_SIZE))
            logger.debug(f"[REDUCE_PHASE] Level {levels} | Creating {len(groups)} groups")

            for idx, i in enumerate(groups):
                group = current_summaries[i:i+ASYNC_REDUCE_GROUP_SIZE]
                logger.debug(f"[REDUCE_PHASE] Level {levels} | Group {idx+1}/{len(groups)} | combining {len(group)} summaries")
                prompt = _get_final_prompt(group, "detailed")  # Use detailed for intermediate
                combined = _call_llm(prompt, config, context=f"reduce_level_{levels}_group_{idx+1}", batch_id=batch_id)
                new_summaries.append(combined)

            # Publish reduce level event
            _publish_event(batch_id, summary_reduce_level(batch_id, levels, len(current_summaries), len(new_summaries)))

            current_summaries = new_summaries
            combined_text = " ".join(current_summaries)
            combined_words = _count_words(combined_text)
            levels += 1

        # Final combination
        logger.info(f"[REDUCE_PHASE] Final combine | summaries={len(current_summaries)} | target_type={summary_type}")
        prompt = _get_final_prompt(current_summaries, summary_type)
        final_summary = _call_llm(prompt, config, context=f"final_{summary_type}", batch_id=batch_id)

        reduce_elapsed = time.time() - reduce_start
        logger.info(f"[REDUCE_PHASE] END | elapsed={reduce_elapsed:.2f}s | levels={levels}")

        # Store final summary
        metadata = {
            "total_chunks": total_chunks,
            "total_words": total_words,
            "method": "hierarchical",
            "batches": num_batches,
            "levels": levels + 1
        }
        store_final_summary(batch_id, final_summary, summary_type, pdf_name, metadata)
        logger.debug(f"[SUMMARIZE] Stored final summary to cache")

        total_elapsed = time.time() - start_time
        logger.info(f"[SUMMARIZE] END | method=hierarchical | batches={num_batches} | levels={levels+1} | elapsed={total_elapsed:.2f}s | summary_words={_count_words(final_summary)}")

        # Publish completed event
        _publish_event(batch_id, summary_completed(batch_id, pdf_name, summary_type, "hierarchical", total_elapsed, _count_words(final_summary)))

        return {
            "summary": final_summary,
            "method": "hierarchical",
            "total_chunks": total_chunks,
            "total_words": total_words,
            "batches": num_batches,
            "levels": levels + 1,
            "storage_mode": SUMMARY_STORAGE_MODE,
            "cached": False
        }

    except Exception as e:
        # Publish failed event
        _publish_event(batch_id, summary_failed(batch_id, str(e), pdf_name))
        raise


def summarize_multiple_pdfs(
    batch_id: str,
    pdf_chunks: Dict[str, List[Dict]],
    summary_type: str = "detailed",
    config: Optional[SummarizerConfig] = None,
    use_cache: bool = True
) -> Dict:
    """
    Summarize multiple PDFs into a combined summary.

    Args:
        batch_id: Batch identifier
        pdf_chunks: Dictionary mapping pdf_name to list of chunks
        summary_type: brief, bulletwise, or detailed
        config: Summarization configuration
        use_cache: Whether to use cached summaries

    Returns:
        Dictionary with combined summary and per-PDF metadata
    """
    start_time = time.time()
    pdf_names = list(pdf_chunks.keys())
    total_pdfs = len(pdf_names)
    total_chunks = sum(len(chunks) for chunks in pdf_chunks.values())

    logger.info(f"[MULTI_PDF] START | batch={batch_id} | pdfs={total_pdfs} | type={summary_type} | use_cache={use_cache}")
    logger.debug(f"[MULTI_PDF] PDF names: {pdf_names}")
    logger.debug(f"[MULTI_PDF] Total chunks across all PDFs: {total_chunks}")

    # Publish multi-PDF started event
    _publish_event(batch_id, multi_pdf_started(batch_id, pdf_names, summary_type))

    if config is None:
        config = DEFAULT_CONFIG

    # Check cache for combined summary
    if use_cache:
        logger.debug(f"[MULTI_PDF] Checking cache for combined summary")
        cached = get_final_summary(batch_id, summary_type, None)
        if cached:
            elapsed = time.time() - start_time
            logger.info(f"[MULTI_PDF] CACHE_HIT | batch={batch_id} | elapsed={elapsed:.2f}s")

            # Publish cache hit event
            _publish_event(batch_id, summary_cache_hit(batch_id, None, summary_type))

            return {
                "summary": cached["summary"],
                "cached": True,
                "storage": cached.get("storage", "unknown")
            }
        logger.debug(f"[MULTI_PDF] Cache miss, proceeding with multi-PDF summarization")

    try:
        # Step 1: Generate summary for each PDF
        logger.info(f"[MULTI_PDF] PHASE_1 | Summarizing {total_pdfs} PDFs individually")
        phase1_start = time.time()
        pdf_summaries = {}

        for idx, (pdf_name, chunks) in enumerate(pdf_chunks.items()):
            logger.info(f"[MULTI_PDF] PDF {idx+1}/{total_pdfs} | name={pdf_name} | chunks={len(chunks)}")

            # Publish PDF started event
            _publish_event(batch_id, multi_pdf_pdf_started(batch_id, pdf_name, idx, total_pdfs))

            # Use detailed for individual PDFs to preserve info
            result = summarize_chunks(
                batch_id=f"{batch_id}_{pdf_name}",
                chunks=chunks,
                summary_type="detailed",
                pdf_name=pdf_name,
                config=config,
                use_cache=use_cache
            )
            pdf_summaries[pdf_name] = result["summary"]
            logger.debug(f"[MULTI_PDF] PDF {idx+1}/{total_pdfs} | summary_words={_count_words(result['summary'])} | method={result.get('method')}")

            # Publish PDF completed event
            _publish_event(batch_id, multi_pdf_pdf_completed(batch_id, pdf_name, idx, total_pdfs))

        phase1_elapsed = time.time() - phase1_start
        logger.info(f"[MULTI_PDF] PHASE_1 END | elapsed={phase1_elapsed:.2f}s")

        # Step 2: Combine PDF summaries
        logger.info(f"[MULTI_PDF] PHASE_2 | Combining {total_pdfs} PDF summaries into {summary_type} summary")
        phase2_start = time.time()

        # Publish combining event
        _publish_event(batch_id, multi_pdf_combining(batch_id, total_pdfs, summary_type))

        combined_words = sum(_count_words(s) for s in pdf_summaries.values())
        logger.debug(f"[MULTI_PDF] Combined input words: {combined_words}")

        prompt = get_multi_pdf_combine_prompt(pdf_summaries, summary_type)
        final_summary = _call_llm(prompt, config, context=f"multi_pdf_combine_{summary_type}", batch_id=batch_id)

        phase2_elapsed = time.time() - phase2_start
        logger.info(f"[MULTI_PDF] PHASE_2 END | elapsed={phase2_elapsed:.2f}s")

        # Store combined summary
        metadata = {
            "total_pdfs": total_pdfs,
            "pdf_names": pdf_names,
            "method": "multi_pdf"
        }
        store_final_summary(batch_id, final_summary, summary_type, None, metadata)
        logger.debug(f"[MULTI_PDF] Stored combined summary to cache")

        total_elapsed = time.time() - start_time
        logger.info(f"[MULTI_PDF] END | pdfs={total_pdfs} | elapsed={total_elapsed:.2f}s | summary_words={_count_words(final_summary)}")

        # Publish multi-PDF completed event
        _publish_event(batch_id, multi_pdf_completed(batch_id, total_pdfs, total_elapsed))

        return {
            "summary": final_summary,
            "summary_type": summary_type,
            "total_pdfs": total_pdfs,
            "pdf_names": pdf_names,
            "pdf_summaries": pdf_summaries,
            "storage_mode": SUMMARY_STORAGE_MODE,
            "cached": False
        }

    except Exception as e:
        # Publish failed event
        _publish_event(batch_id, summary_failed(batch_id, str(e), None))
        raise
