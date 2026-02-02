"""
LLM Client with comprehensive logging support.

Supports:
- Ollama backend
- VLLM backend
- Streaming responses
- Automatic request/response logging
- Metrics collection
"""
import time
import json
import requests
from typing import Generator, Optional

from config import (
    LLM_BACKEND,
    VLLM_URL,
    OLLAMA_URL,
    DEFAULT_MODEL,
    get_model_context_length
)
from logging_config import (
    get_llm_logger,
    log_llm_request,
    log_llm_response,
    log_metrics,
    log_llm_call,
    log_context_usage,
    RequestContext
)

logger = get_llm_logger()


@log_llm_call(task="generate")
def generate_text(prompt: str, model: str | None = None) -> str:
    """
    Generate text from either Ollama or VLLM based on LLM_BACKEND.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (uses DEFAULT_MODEL if not specified)

    Returns:
        Generated text response
    """
    model_name = model or DEFAULT_MODEL
    logger.debug(f"generate_text called | model={model_name} | backend={LLM_BACKEND}")

    if LLM_BACKEND == "vllm":
        return _call_vllm(prompt, model_name)

    return _call_ollama(prompt, model_name)


def generate_text_with_logging(
    prompt: str,
    model: str | None = None,
    task: str = "generate",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> str:
    """
    Generate text with explicit logging control.

    Args:
        prompt: The prompt to send
        model: Model name
        task: Task identifier for logging
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response
    """
    model_name = model or DEFAULT_MODEL

    # Get context limit for model
    context_limit = get_model_context_length(model_name)

    # Log request
    request_id = log_llm_request(
        model=model_name,
        backend=LLM_BACKEND,
        task=task,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Log context usage and get stats
    context_stats = log_context_usage(
        request_id=request_id,
        model=model_name,
        prompt=prompt,
        context_limit=context_limit
    )

    start_time = time.time()

    try:
        if LLM_BACKEND == "vllm":
            response = _call_vllm_internal(prompt, model_name, temperature, max_tokens)
        else:
            response = _call_ollama_internal(prompt, model_name, temperature)

        latency_ms = (time.time() - start_time) * 1000

        # Log successful response
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            response=response,
            latency_ms=latency_ms,
            status="success"
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=len(response),
            status="success",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        return response

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        # Log error
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            response="",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e)
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend=LLM_BACKEND,
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=0,
            status="error",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        raise


def _call_ollama(prompt: str, model: str) -> str:
    """Call Ollama API (simple version for decorator)."""
    return _call_ollama_internal(prompt, model)


def _call_ollama_internal(
    prompt: str,
    model: str,
    temperature: float = 0.3,
    timeout: int = 300
) -> str:
    """
    Call Ollama API with full parameters.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature
        timeout: Request timeout in seconds

    Returns:
        Generated text
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    logger.debug(f"Calling Ollama | url={OLLAMA_URL}/api/generate | model={model}")

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=timeout
        )
        r.raise_for_status()

        response_data = r.json()
        response_text = response_data.get("response", "").strip()

        # Log token info if available
        if "eval_count" in response_data:
            logger.debug(
                f"Ollama tokens | eval_count={response_data.get('eval_count')} | "
                f"eval_duration={response_data.get('eval_duration')}"
            )

        return response_text

    except requests.exceptions.Timeout:
        logger.error(f"Ollama timeout after {timeout}s | model={model}")
        raise RuntimeError(f"Ollama API timeout after {timeout}s")

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed | model={model} | error={e}")
        raise RuntimeError(f"Ollama API failed: {e}")


def _call_vllm(prompt: str, model: str) -> str:
    """Call VLLM API (simple version for decorator)."""
    return _call_vllm_internal(prompt, model)


def _call_vllm_internal(
    prompt: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: int = 300
) -> str:
    """
    Call VLLM API with full parameters.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        Generated text
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    logger.debug(f"Calling VLLM | url={VLLM_URL}/v1/chat/completions | model={model}")

    try:
        r = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        r.raise_for_status()

        response_data = r.json()
        response_text = response_data["choices"][0]["message"]["content"].strip()

        # Log usage info if available
        usage = response_data.get("usage", {})
        if usage:
            logger.debug(
                f"VLLM tokens | prompt={usage.get('prompt_tokens')} | "
                f"completion={usage.get('completion_tokens')} | "
                f"total={usage.get('total_tokens')}"
            )

        return response_text

    except requests.exceptions.Timeout:
        logger.error(f"VLLM timeout after {timeout}s | model={model}")
        raise RuntimeError(f"VLLM API timeout after {timeout}s")

    except requests.exceptions.RequestException as e:
        logger.error(f"VLLM request failed | model={model} | error={e}")
        raise RuntimeError(f"VLLM API failed: {e}")


# =========================
# Streaming Support
# =========================

def stream_ollama(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    task: str = "stream"
) -> Generator[str, None, None]:
    """
    Stream response from Ollama token by token.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature
        task: Task identifier for logging

    Yields:
        Individual tokens as they're generated
    """
    model_name = model or DEFAULT_MODEL

    # Get context limit for model
    context_limit = get_model_context_length(model_name)

    # Log request
    request_id = log_llm_request(
        model=model_name,
        backend="ollama",
        task=task,
        prompt=prompt,
        temperature=temperature
    )

    # Log context usage
    context_stats = log_context_usage(
        request_id=request_id,
        model=model_name,
        prompt=prompt,
        context_limit=context_limit
    )

    start_time = time.time()
    full_response = []

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature}
            },
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        full_response.append(token)
                        yield token

                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        latency_ms = (time.time() - start_time) * 1000
        complete_response = "".join(full_response)

        # Log response
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            response=complete_response,
            latency_ms=latency_ms,
            status="success"
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=len(complete_response),
            status="success",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            response="",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e)
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="ollama",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=0,
            status="error",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        logger.error(f"Stream error | {e}")
        yield f"\n[Error: {str(e)}]"


def stream_vllm(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    task: str = "stream"
) -> Generator[str, None, None]:
    """
    Stream response from VLLM token by token.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens
        task: Task identifier for logging

    Yields:
        Individual tokens as they're generated
    """
    model_name = model or DEFAULT_MODEL

    # Get context limit for model
    context_limit = get_model_context_length(model_name)

    # Log request
    request_id = log_llm_request(
        model=model_name,
        backend="vllm",
        task=task,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Log context usage
    context_stats = log_context_usage(
        request_id=request_id,
        model=model_name,
        prompt=prompt,
        context_limit=context_limit
    )

    start_time = time.time()
    full_response = []

    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            },
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            full_response.append(token)
                            yield token
                    except json.JSONDecodeError:
                        continue

        latency_ms = (time.time() - start_time) * 1000
        complete_response = "".join(full_response)

        # Log response
        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            response=complete_response,
            latency_ms=latency_ms,
            status="success"
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=len(complete_response),
            status="success",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        log_llm_response(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            response="",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e)
        )

        # Log metrics with context stats
        log_metrics(
            request_id=request_id,
            model=model_name,
            backend="vllm",
            task=task,
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            response_chars=0,
            status="error",
            context_limit=context_stats["context_limit"],
            estimated_tokens=context_stats["estimated_tokens"],
            context_usage_percent=context_stats["usage_percent"]
        )

        logger.error(f"Stream error | {e}")
        yield f"\n[Error: {str(e)}]"


def stream_text(
    prompt: str,
    model: str | None = None,
    task: str = "stream"
) -> Generator[str, None, None]:
    """
    Stream text from the configured backend.

    Args:
        prompt: The prompt text
        model: Model name
        task: Task identifier for logging

    Yields:
        Individual tokens
    """
    if LLM_BACKEND == "vllm":
        yield from stream_vllm(prompt, model, task=task)
    else:
        yield from stream_ollama(prompt, model, task=task)
