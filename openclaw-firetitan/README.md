# OpenClaw-RL with Firetitan Backend

Replaces the local Slime/Megatron GPU training stack with [Fireworks Firetitan](https://fireworks.ai) for fully remote RL training and inference. No local GPUs are required.

## Architecture

```
OpenClaw Client
    |
    v
[OpenClaw Firetitan Proxy]  (local FastAPI server, port 30000)
    |           |
    v           v
[Fireworks     [Firetitan
 Deployment]    Trainer]
 (inference     (remote training,
  + PRM)        teacher logprobs,
                weight sync)
```

Three components work together:

1. **Fireworks Deployment** -- remote inference endpoint for policy rollouts and PRM judge queries.
2. **Firetitan Trainer** -- remote training backend that runs forward/backward passes, optimizer steps, and syncs updated weights back to the deployment.
3. **OpenClaw Firetitan Proxy** -- local FastAPI server that acts as an OpenAI-compatible API proxy. It collects training data from live conversations, computes PRM evaluations, obtains teacher log-probs, and feeds samples to the training loop.

## Prerequisites

- A Fireworks API key with access to Firetitan Training SDK.
- Python 3.12+
- A HuggingFace-compatible tokenizer for your base model (downloaded automatically).
- The Firetitan Training SDK is in private preview. The `--pre` flag is required when installing `fireworks-ai[training]` to get the alpha release that includes `fireworks.training.sdk` and `tinker`.

## Setup

### 1. Create virtual environment and install dependencies

```bash
cd openclaw-firetitan

python3 -m venv .venv
source .venv/bin/activate

pip install --pre -r requirements.txt
```

### 2. Set your Fireworks API key

```bash
export FIREWORKS_API_KEY="your-fireworks-api-key"
```

## Launch

```bash
cd openclaw-firetitan
source .venv/bin/activate

FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
BASE_MODEL="accounts/fireworks/models/qwen3-8b" \
TRAINING_SHAPE_ID="accounts/fireworks/trainingShapes/qwen3-8b-128k" \
DEPLOYMENT_ID="openclaw-serving" \
TOKENIZER_MODEL="Qwen/Qwen3-8B" \
ROLLOUT_BATCH_SIZE=16 \
LEARNING_RATE=1e-5 \
W_OPD=1.0 \
W_RL=1.0 \
PRM_ENABLED=1 \
PRM_M=3 \
SERVER_PORT=30000 \
python run_openclaw_firetitan.py
```

The script will:

1. Resolve the training shape from Fireworks.
2. Create (or reuse) a Fireworks Deployment for inference.
3. Create a Firetitan Trainer job for remote training.
4. Start the local OpenClaw proxy server on the configured port.
5. Enter the training loop: drain samples from the queue, run `forward_backward_custom` + `optim_step` on the remote trainer, and periodically sync weights to the deployment.

Once the proxy is ready, it prints a banner:

```
======================================================================
  [Firetitan] OpenClaw proxy ready
  proxy 0.0.0.0:30000 -> <deployment_chat_url>
  PRM enabled: True (m=3)
======================================================================
```

The model is then served as an OpenAI-compatible API at `http://<HOST_IP>:30000/v1`.

## OpenClaw Configuration

To route OpenClaw requests to your Firetitan-backed RL server, add a provider entry in your `openclaw.json`:

```json
{
  "models": {
    "providers": {
      "openclaw-rl": {
        "baseUrl": "http://<HOST_IP>:30000/v1",
        "apiKey": "no-auth-needed",
        "api": "openai-completions",
        "models": [
          {
            "id": "default",
            "name": "Qwen3 8B (OpenClaw-RL Firetitan)",
            "reasoning": true,
            "input": ["text"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

Replace `<HOST_IP>` with the IP address of the machine running the Firetitan proxy.

If you set `EXPECTED_API_KEY` when launching, use that value as the `apiKey` in the config above.

If you set `SERVED_MODEL_NAME` to something other than `"default"`, use that as the model `id`.

## Environment Variables

All configuration is via environment variables. Only `FIREWORKS_API_KEY` is required; everything else has defaults.

### Required

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | Fireworks platform API key |

### Model and Training Shape

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL` | `accounts/fireworks/models/qwen3-8b` | Fireworks model identifier |
| `TRAINING_SHAPE_ID` | `accounts/fireworks/trainingShapes/qwen3-8b-128k` | Training shape for the Firetitan trainer |
| `DEPLOYMENT_ID` | `openclaw-serving` | Fireworks deployment name |
| `TOKENIZER_MODEL` | `Qwen/Qwen3-8B` | HuggingFace tokenizer model name |

### Training Hyperparameters

| Variable | Default | Description |
|---|---|---|
| `ROLLOUT_BATCH_SIZE` | `16` | Number of samples per training step |
| `LEARNING_RATE` | `1e-5` | Adam learning rate |
| `W_OPD` | `1.0` | Weight for OPD (teacher distillation) advantage |
| `W_RL` | `1.0` | Weight for GRPO (reward) advantage |
| `EPS_CLIP` | `0.2` | Lower PPO clipping bound |
| `EPS_CLIP_HIGH` | `0.28` | Upper PPO clipping bound |
| `LORA_RANK` | `0` | LoRA rank (0 = full-parameter training) |
| `GRADIENT_ACCUMULATION` | `1` | Gradient accumulation steps |
| `MAX_SEQ_LEN` | `32768` | Maximum sequence length for training datums |
| `MAX_STEPS` | `0` | Stop after N steps (0 = run forever) |
| `WEIGHT_SYNC_INTERVAL` | `1` | Steps between weight syncs to deployment |

### PRM (Process Reward Model)

| Variable | Default | Description |
|---|---|---|
| `PRM_ENABLED` | `1` | Enable PRM judge evaluation (1/0) |
| `PRM_M` | `3` | Number of PRM judge votes (majority voting) |
| `PRM_TEMPERATURE` | `0.6` | PRM sampling temperature |
| `PRM_MAX_TOKENS` | `4096` | PRM max generation tokens |

### Server

| Variable | Default | Description |
|---|---|---|
| `SERVER_PORT` | `30000` | Local proxy server port |
| `SERVED_MODEL_NAME` | `default` | Model name returned by the proxy API |
| `EXPECTED_API_KEY` | _(empty)_ | Proxy auth key (empty = no auth) |
| `RECORD_DIR` | _(empty)_ | Directory for conversation/PRM record files |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai` | Fireworks API base URL |

## File Structure

```
openclaw-firetitan/
  run_openclaw_firetitan.py    -- entry point: bootstraps deployment, trainer, proxy, and training loop
  firetitan_server.py          -- FastAPI proxy server (OpenAI-compatible, collects training data)
  firetitan_training_loop.py   -- training loop: drain queue -> build batch -> train step -> weight sync
  firetitan_loss.py            -- combined RL (GRPO) + OPD (distillation) loss function
  requirements.txt             -- Python dependencies
```
