# ADVANCE-FER: Production-Grade Facial Expression Recognition Service

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-3.x%20Microservice-black)](https://flask.palletsprojects.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-High%20Throughput-brightgreen)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5)](https://kubernetes.io/)
[![Terraform](https://img.shields.io/badge/Terraform-IaC%20AWS-7B42BC)](https://www.terraform.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-SQLAlchemy-4169E1)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-Caching%20%26%20RateLimit-DC382D)](https://redis.io/)
[![Tests](https://img.shields.io/badge/Pytest-20%20Passed-success)](https://pytest.org/)

A production-grade, zero-training Facial Expression Recognition (FER) microservice and Affective AI system. Powered by **MediaPipe FaceMesh** for multi-face geometric alignment, an optimized **Vision Transformer (ViT) ONNX Runtime** engine for real-time affective computing, and an autonomous **Affective GenAI Agent** for behavioral insights.

> **Complete Technical Portfolio & Skills Case Study**: See [PORTFOLIO_CASE_STUDY.md](PORTFOLIO_CASE_STUDY.md) for resume impact bullets, STAR interview talking points, and technical mappings.

---

## Key Features

- **Zero-Training Required**: Automatically downloads and caches pre-trained SOTA emotion weights (~85MB) on first boot.
- **Sub-15ms Inference**: Hardware-accelerated with ONNX Runtime (CPU/CUDA auto-detection, batch processing supported).
- **Multi-Face Tracking**: Simultaneous detection, bounding box extraction, and emotion classification for all faces in frame.
- **Affective Dimension & GenAI Agent**: Quantifies discrete emotions as well as continuous **Valence (Pleasantness)** and **Arousal (Activation)** via Russell's Circumplex Model, coupled with an LLM-powered **Affective AI Agent** (`agent/affective_agent.py`) for stress/engagement scoring and empathy coaching.
- **Distributed Caching & Database Persistence**: **Redis** perceptual frame hashing (`cache/redis_client.py`) to bypass redundant inference, paired with **PostgreSQL / SQLite** ORM models (`database/models.py`) for session telemetry logging.
- **Dual Framework Serving**: Production **FastAPI** ASGI microservice (`api/server.py`) and **Flask** WSGI microservice (`api/flask_app.py`).
- **Cloud & DevOps Ready**: Production **Kubernetes** manifests (`k8s/`), **Terraform** AWS IaC (`terraform/`), **GitHub Actions** CI/CD pipeline (`.github/workflows/ci.yml`), and multi-stage `Dockerfile`.

---

## Directory Structure

```text
ADVANCE-FER/
├── api/                     # Production FastAPI Microservice
│   ├── schemas.py           # Pydantic V2 input/output schemas
│   └── server.py            # REST endpoints (/v1/predict, /healthz, /v1/info)
├── deployment/              # Core inference logic
│   └── inference.py         # Multi-face tracking & HUD annotation
├── models/                  # Neural network engines & weights
│   ├── emotion_engine.py    # ONNX Runtime inference & Circumplex mapping
│   ├── weights_manager.py   # Automatic weight download and caching
│   ├── fer_model.py         # Custom PyTorch dual-stream model definition
│   └── temporal.py          # Video sequence models (LSTM/Transformer)
├── preprocessing/           # Computer vision pipelines
│   ├── face_detector.py     # MediaPipe FaceMesh multi-face detector
│   └── alignment.py         # Canonical eye-level face aligner
├── tests/                   # Automated pytest suite
│   ├── test_api.py          # API endpoint integration tests
│   ├── test_detector.py     # Face detector unit tests
│   ├── test_alignment.py    # Canonical alignment unit tests
│   └── test_emotion_engine.py # ONNX model validation tests
├── app.py                   # Streamlit demonstration UI
├── verify_pipeline.py       # Self-test diagnostic utility
├── Dockerfile               # Multi-stage production container
├── docker-compose.yml       # Orchestration for API + UI
├── requirements.txt         # Pinned production dependencies
└── README.md                # System documentation
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Ashutosh-Yadav-256/ADVANCE-FER.git
cd ADVANCE-FER

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify System Health

Run the diagnostic self-test to verify model checkpoints and pipelines:

```bash
python verify_pipeline.py
```

### 3. Run Automated Tests

```bash
pytest tests/ -v
```

---

## Running the Services

### Option A: Production FastAPI Server

Start the REST API microservice:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Probe:** [http://localhost:8000/healthz](http://localhost:8000/healthz)

### Option B: Interactive Streamlit UI

Start the real-time webcam and image analysis dashboard:

```bash
streamlit run app.py
```

- Accessible at [http://localhost:8501](http://localhost:8501)

### Option C: Docker Deployment

Deploy both API and UI in containerized environments:

```bash
docker-compose up --build -d
```

---

## API Usage Examples

### 1. Predict Emotion from Image File (`curl`)

```bash
curl -X POST "http://localhost:8000/v1/predict/image" \
     -H "accept: application/json" \
     -F "file=@sample_face.jpg"
```

**Sample Response:**

```json
{
  "success": true,
  "image_width": 1280,
  "image_height": 720,
  "faces_detected": 1,
  "faces": [
    {
      "face_id": 0,
      "bbox": [450, 180, 220, 220],
      "dominant_emotion": "happy",
      "confidence": 0.9421,
      "probabilities": {
        "happy": 0.9421,
        "neutral": 0.0315,
        "surprise": 0.0152,
        "sad": 0.0041,
        "fear": 0.0032,
        "angry": 0.0021,
        "disgust": 0.0018
      },
      "valence": 0.7712,
      "arousal": 0.4901
    }
  ],
  "latency_ms": 14.8
}
```

### 2. Python Client Example

```python
import requests

url = "http://localhost:8000/v1/predict/image"
with open("face.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

data = response.json()
for face in data["faces"]:
    print(f"Face #{face['face_id']}: {face['dominant_emotion']} ({face['confidence']*100:.1f}%)")
    print(f"Valence: {face['valence']:+.2f}, Arousal: {face['arousal']:+.2f}")
```

### 3. Get Annotated Image Directly

```bash
curl -X POST "http://localhost:8000/v1/predict/annotated" \
     -F "file=@face.jpg" \
     --output annotated_face.jpg
```

---

## Emotion Taxonomy & Affective Space

The engine classifies 7 core discrete emotions and projects them onto **Russell's Circumplex Model of Affect**:

| Emotion | Valence (Pleasantness) | Arousal (Activation) |
| :--- | :---: | :---: |
| **Happy** | `+0.81` | `+0.51` |
| **Surprise** | `+0.40` | `+0.67` |
| **Neutral** | `0.00` | `0.00` |
| **Sad** | `-0.63` | `-0.27` |
| **Fear** | `-0.64` | `+0.60` |
| **Angry** | `-0.43` | `+0.67` |
| **Disgust** | `-0.60` | `+0.35` |

---

## Model Context Protocol (MCP) Server

ADVANCE-FER includes an official **Model Context Protocol (MCP)** server built with `FastMCP`, enabling AI agents (Claude Desktop, Antigravity, Cursor) to directly invoke emotion recognition and affective reasoning as native tools.

### 1. Available MCP Tools

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `detect_emotions_from_file` | `image_path: str` | Detects all faces in a local image file and classifies emotions + valence/arousal. |
| `detect_emotions_from_base64` | `image_base64: str` | Classifies emotions from base64 encoded image strings. |
| `analyze_affective_behavior` | `face_data: dict, context: str` | Uses GenAI Affective Agent to produce psychological stress, engagement, and empathy reasoning. |
| `capture_webcam_and_detect` | `camera_index: int` | Snaps a live frame from the user's webcam and returns real-time emotion telemetry. |
| `get_model_status` | *(none)* | Reports model architecture, active execution provider (CPU/CUDA), and uptime. |

### 2. Available MCP Resources

- `fer://taxonomy`: Returns the 7 emotion classes and Russell Circumplex coordinates.
- `fer://health`: Returns real-time health and memory metrics.

### 3. Running the MCP Server

```bash
# Standard stdio mode (for Claude Desktop, Antigravity, Cursor)
python mcp_server.py

# Or SSE / HTTP transport mode (for web clients)
python mcp_server.py --transport sse --port 8001
```

### 4. Client Configuration (`claude_desktop_config.json` or `mcp_config.json`)

Add the following to your MCP client settings:

```json
{
  "mcpServers": {
    "advance-fer": {
      "command": "python",
      "args": [
        "c:/Desktop/CODING _IS_LIFE/1 ANTI GRAVITY/TASKMANGER/ADVANCE-FER/mcp_server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
