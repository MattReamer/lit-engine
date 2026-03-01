# lit-engine

A from-scratch LLM inference server with continuous batching, KV cache management, and streaming token generation.

No inference frameworks. The scheduling, batching, and cache management are implemented from scratch to demonstrate how LLM serving works under the hood.

## Why This Exists

Production inference servers like vLLM and TGI are complex systems that abstract away the core mechanics of LLM serving. This project strips that back and implements the key ideas from first principles: autoregressive generation with a hand-managed KV cache, continuous batching that admits and evicts requests between generation steps, and async streaming that bridges a blocking GPU thread back to HTTP clients.

The serving and orchestration layer is built from scratch. Model loading and tokenization use HuggingFace `transformers` — reimplementing those would be engineering theater, not engineering.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  HTTP/SSE   │────▶│   Request    │────▶│     Batch         │────▶│    Model     │
│  API Layer  │◀────│   Queue      │◀────│     Scheduler     │◀────│    Engine    │
│  (FastAPI)  │     │  (asyncio)   │     │  (continuous)     │     │  (PyTorch)   │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────────────┘
                                                                         │
                                                                   ┌─────▼────────┐
                                                                   │   KV Cache   │
                                                                   │   Manager    │
                                                                   └──────────────┘
```

**Request lifecycle:** Client sends a prompt via `POST /v1/completions` → API tokenizes and queues a `GenerationRequest` → Scheduler assigns a KV cache slot and adds it to the active batch → Engine runs prefill (full prompt forward pass) then decode steps (one token at a time using cached context) → Tokens are pushed into an `asyncio.Queue` and streamed back to the client via SSE → When the request hits a stop condition, it's evicted from the batch and its cache slot is released → New waiting requests join the batch mid-generation.

## Key Concepts Demonstrated

**Continuous batching** — Requests enter and leave the batch between generation steps rather than waiting for the entire batch to finish. This is the core scheduling technique used by production inference servers and the main throughput advantage over naive static batching.

**Pre-allocated KV cache pool** — GPU memory for key/value caches is allocated at startup as a fixed pool of slots rather than dynamically per-request. This avoids CUDA malloc overhead during generation and makes memory usage predictable. The tradeoff is wasted memory for short sequences — production systems solve this with paged allocation (PagedAttention).

**Prefill/decode separation** — New requests go through a prefill pass (all prompt tokens processed at once, compute-bound) before entering the decode loop (one token per step using cached context, memory-bound). These have fundamentally different performance characteristics and the engine handles them separately.

**Async/threaded bridge** — The FastAPI event loop handles HTTP connections and SSE streaming asynchronously. GPU inference runs in a dedicated thread via `run_in_executor` since forward passes are blocking. An `asyncio.Queue` per request bridges the two worlds.

## Quick Start

```bash
pip install -r requirements.txt

# Download model weights (~3 GB)
python scripts/download_model.py

# Start the server
python main.py

# Generate text
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The key insight about transformer architectures is", "max_tokens": 128}'
```

Default model is Qwen 2.5 1.5B (fp16).

## API

### `POST /v1/completions`

```json
{
  "prompt": "The key insight about transformer architectures is",
  "max_tokens": 128,
  "temperature": 0.8,
  "top_p": 0.95,
  "stop": ["\n\n"],
  "stream": true
}
```

Returns an SSE stream:

```
data: {"request_id": "abc-123", "token": " that", "finish_reason": null}
data: {"request_id": "abc-123", "token": " they", "finish_reason": null}
...
data: {"request_id": "abc-123", "token": "", "finish_reason": "stop"}
data: [DONE]
```

### `GET /v1/health`

```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-1.5B",
  "cache_slots_free": 12,
  "active_batch_size": 4,
  "queue_depth": 2
}
```

### `GET /metrics`

Prometheus-format metrics: request latency histograms, time-to-first-token, tokens/sec throughput, batch size, queue depth, cache utilization.

## Configuration

All settings are configurable via environment variables with a `LIT_` prefix:

| Variable | Default | Description |
|---|---|---|
| `LIT_MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-1.5B` | HuggingFace model identifier or local path |
| `LIT_DEVICE` | `cuda` | Device for inference |
| `LIT_DTYPE` | `float16` | Model precision (`float16`, `bfloat16`, `float32`) |
| `LIT_MAX_BATCH_SIZE` | `8` | Maximum concurrent requests in a batch |
| `LIT_MAX_SEQUENCE_LENGTH` | `2048` | Maximum total sequence length (prompt + generation) |
| `LIT_MAX_CACHE_SLOTS` | `16` | Number of pre-allocated KV cache slots |
| `LIT_MAX_WAITING_TIME_S` | `0.05` | Max time before running an undersized batch |

## Project Structure

```
lit-engine/
├── src/
│   ├── api/app.py             # FastAPI app, SSE streaming, request validation
│   ├── engine/engine.py       # Forward passes, prefill, batched decode, token sampling
│   ├── scheduler/scheduler.py # Request queue, batch admission/eviction, continuous batching loop
│   ├── cache/kv_cache.py      # Pre-allocated GPU KV cache pool, slot management
│   ├── models/loader.py       # Model + tokenizer loading, config extraction
│   ├── metrics/metrics.py     # Prometheus metrics
│   └── types.py               # GenerationRequest, RequestStatus, shared types
├── tests/
├── scripts/
├── config.py
└── main.py
```

## Design Decisions

### Why not use `model.generate()`?

The HuggingFace `generate()` method handles a single request synchronously. It doesn't support continuous batching, external KV cache management, or streaming tokens mid-generation in a way that integrates with an async serving layer. Writing the generation loop from scratch is the entire point — it's where the interesting engineering lives.

### Why pre-allocated cache slots instead of paged allocation?

Simplicity. A fixed pool with slot-level allocation is straightforward to reason about and debug. It wastes memory for short sequences because each slot reserves space for `max_sequence_length` tokens regardless of actual usage. Production systems like vLLM solve this with PagedAttention, which manages cache memory in fixed-size blocks and allocates them on demand. That's a meaningful improvement but significantly more complex — noted as a stretch goal.

### Why `run_in_executor` instead of a separate process?

GPU forward passes block the calling thread, so they can't run directly on the asyncio event loop. Using `run_in_executor` with the default thread pool is the simplest approach that works. A dedicated process with IPC would be more robust for production (isolates GPU crashes from the API process) but adds complexity that doesn't serve the goals of this project.

## Tests

```bash
pytest tests/ -v
```

The KV cache manager is fully testable on CPU with no model dependency. The scheduler can be tested with a mock engine. Integration tests run a full request lifecycle against the real server using `httpx`.

## Limitations

This is a learning project, not a production inference server. Known limitations:

- Single GPU only — no tensor parallelism or pipeline parallelism
- No paged KV cache — memory waste for short sequences
- Prefills are processed one at a time rather than batched
- No request cancellation on client disconnect
- No model quantization support
- No CUDA graph optimization for the decode path

## Built With

PyTorch, HuggingFace Transformers, FastAPI, Uvicorn, prometheus-client

## License

MIT
