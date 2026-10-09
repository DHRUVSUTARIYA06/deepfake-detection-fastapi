# 🛡️ Deepfake Detection API (FastAPI Backend)

A production-ready, high-performance FastAPI backend for **Image and Video Deepfake Detection** powered by a fine-tuned **EfficientNet-B0** classifier with automated face detection and cropping.

---

## 📋 Table of Contents
1. [Model Specifications](#-model-specifications)
2. [Project Architecture](#-project-architecture)
3. [Quick Start (Local Development)](#-quick-start-local-development)
4. [API Endpoints & Documentation](#-api-endpoints--documentation)
5. [Client Integration Examples](#-client-integration-examples)
    - [cURL](#curl)
    - [Python Requests](#python-requests)
    - [Mobile Applications (Flutter / React Native)](#mobile-applications)
6. [Testing & Verification](#-testing--verification)
7. [Production Deployment](#-production-deployment)
    - [Deploy to Render](#deploy-to-render)
    - [Deploy to Railway](#deploy-to-railway)
    - [Docker Deployment](#docker-deployment)
8. [Configuration & Environment Variables](#-configuration--environment-variables)

---

## 🔍 Model Specifications

The application uses your pre-trained model located in the `models/` directory:

| Property | Value |
|---|---|
| **Model Architecture** | `EfficientNet-B0` (Pre-trained on ImageNet, fine-tuned for Deepfake Detection) |
| **Primary Framework** | **ONNX Runtime** (`CPUExecutionProvider` / `CUDAExecutionProvider`) |
| **Fallback Framework** | **PyTorch** (`torch.nn.Module` using `torchvision.models.efficientnet_b0`) |
| **Weights Files** | `models/deepfake_efficientnet_b0.onnx` (16.0 MB), `models/deepfake_efficientnet_b0.pt` (16.3 MB) |
| **Input Dimensions** | `[1, 3, 224, 224]` (Batch Size, Channels, Height, Width) |
| **Color Space** | `RGB` |
| **Image Preprocessing** | Resize to 224×224 → Scale to $[0.0, 1.0]$ → Standardize with ImageNet stats |
| **Normalization Mean** | `[0.485, 0.456, 0.406]` |
| **Normalization Std** | `[0.229, 0.224, 0.225]` |
| **Class Mapping** | `Index 0` $\rightarrow$ **`FAKE`**, `Index 1` $\rightarrow$ **`REAL`** |
| **Validation Accuracy** | **95.99%** |
| **Face Detection Method** | OpenCV Frontal Face Haar Cascade with 20% margin padding |

> [!NOTE]
> The model is loaded **once during application startup** and kept in memory to serve both image and video inference requests with minimal latency.

---

## 🏗️ Project Architecture

```
Fast_api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan, CORS, and root routes
│   ├── config.py                # Pydantic Settings and environment configuration
│   ├── schemas.py               # Pydantic request and response models
│   ├── core/
│   │   ├── model.py             # Single-instance ONNX/PyTorch model loader
│   │   ├── face_detector.py     # Face detection with 20% margin padding
│   │   └── preprocessor.py     # ImageNet resizing and normalization
│   ├── services/
│   │   ├── detector_service.py  # Image validation, face extraction, inference
│   │   └── video_processor.py   # Video sampling, face extraction, aggregation
│   └── api/
│       └── v1/
│           ├── endpoints.py     # /detect/image, /detect/video, /model-info
│           └── router.py        # API router
├── models/                      # Trained model weights
│   ├── deepfake_efficientnet_b0.onnx
│   └── deepfake_efficientnet_b0.pt
├── samples/                     # Test samples (Real/Fake faces and videos)
├── tests/
│   ├── conftest.py              # Test client fixtures
│   └── test_api.py              # End-to-end integration tests
├── Dockerfile                   # Production CPU Docker container
├── render.yaml                  # Render deployment blueprint
├── railway.json                 # Railway deployment configuration
├── Procfile                     # Process file for cloud platforms
├── requirements.txt             # Locked Python dependencies
└── pytest.ini                   # Pytest configuration
```

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd Fast_api

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Documentation (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 API Endpoints & Documentation

### 1. `GET /` — API Information
Returns API name, version, and route links.

### 2. `GET /health` — Readiness & Health Check
Checks if the application, model weights, and face detector are initialized.

**Response:**
```json
{
  "status": "healthy",
  "ready": true,
  "model_loaded": true,
  "face_detector_loaded": true,
  "device": "CPU",
  "version": "v1"
}
```

### 3. `GET /api/v1/model-info` — Model Metadata
Returns complete architecture, input dimensions, preprocessing parameters, and training accuracy.

---

### 4. `POST /api/v1/detect/image` — Image Detection

Analyzes an uploaded image file for deepfake manipulation.

- **Content-Type**: `multipart/form-data`
- **Field Name**: `file`
- **Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- **Max File Size**: 15 MB (configurable)

#### Processing Pipeline:
1. Validates format and decodes image.
2. Detects all faces using OpenCV Haar Cascade.
3. Crops faces with 20% contextual boundary padding.
4. Normalizes and standardizes to $[1, 3, 224, 224]$.
5. Runs model inference.
6. Returns image-level prediction and confidence.

> [!IMPORTANT]
> If **no faces are detected**, the API returns `422 Unprocessable Entity`. An image with no detectable faces is **never** falsely classified as REAL.

#### Example Response (200 OK):
```json
{
  "success": true,
  "media_type": "image",
  "prediction": "FAKE",
  "fake_probability": 0.91,
  "real_probability": 0.09,
  "faces_detected": 1,
  "faces_analyzed": 1,
  "processing_time_seconds": 0.42,
  "model_version": "v1",
  "details": [
    {
      "face_index": 1,
      "bounding_box": [319, 671, 511, 863],
      "fake_probability": 0.91,
      "real_probability": 0.09,
      "prediction": "FAKE"
    }
  ]
}
```

---

### 5. `POST /api/v1/detect/video` — Video Detection

Analyzes an uploaded video file for deepfake manipulation.

- **Content-Type**: `multipart/form-data`
- **Field Name**: `file`
- **Query Parameter**: `max_frames` (optional, default: 20, max: 60)
- **Supported Formats**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- **Max File Size**: 100 MB (configurable)
- **Max Video Duration**: 120 seconds (configurable)

#### Processing Pipeline:
1. Streams upload safely to a temporary file.
2. Validates duration, frame rate, and frame count.
3. Uniformly samples $N$ frames across the entire video timeline.
4. Detects faces in sampled frames.
5. Runs face crops through the **same** trained model.
6. Computes per-frame fake likelihood and aggregates across the video.
7. Safely unlinks temporary files in a `finally` block even if errors occur.

#### Example Response (200 OK):
```json
{
  "success": true,
  "media_type": "video",
  "prediction": "REAL",
  "fake_probability": 0.12,
  "real_probability": 0.88,
  "frames_analyzed": 20,
  "frames_with_faces": 18,
  "faces_analyzed": 24,
  "processing_time_seconds": 3.5,
  "model_version": "v1",
  "details": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "faces_detected": 1,
      "fake_probability": 0.0003,
      "real_probability": 0.9997,
      "prediction": "REAL"
    }
  ]
}
```

---

## 💻 Client Integration Examples

### cURL

#### Image Detection:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/face_photo.jpg"
```

#### Video Detection:
```bash
curl -X POST "http://localhost:8000/api/v1/detect/video?max_frames=20" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video_sample.mp4"
```

---

### Python Requests

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Health check
health = requests.get(f"{BASE_URL}/health").json()
print("Health:", health)

# 2. Image Detection
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/detect/image",
        files={"file": ("test_image.jpg", f, "image/jpeg")},
    )

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Fake Probability: {result['fake_probability']:.2%}")
    print(f"Real Probability: {result['real_probability']:.2%}")
elif response.status_code == 422:
    print("No faces detected in image.")
else:
    print(f"Error {response.status_code}: {response.text}")

# 3. Video Detection
with open("test_video.mp4", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/detect/video",
        params={"max_frames": 20},
        files={"file": ("test_video.mp4", f, "video/mp4")},
    )

if response.status_code == 200:
    result = response.json()
    print(f"Video Prediction: {result['prediction']}")
    print(f"Fake Probability: {result['fake_probability']:.2%}")
    print(f"Frames Analyzed: {result['frames_analyzed']}")
```

---

### Mobile Applications

#### Flutter (Dart) Example:
```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> detectDeepfakeImage(File imageFile, String baseUrl) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('$baseUrl/api/v1/detect/image'),
  );

  request.files.add(
    await http.MultipartFile.fromPath('file', imageFile.path),
  );

  var streamedResponse = await request.send();
  var response = await http.Response.fromStream(streamedResponse);

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else if (response.statusCode == 422) {
    throw Exception('No detectable face found in the image.');
  } else {
    throw Exception('Detection failed with status: ${response.statusCode}');
  }
}
```

---

## 🧪 Testing & Verification

The test suite runs end-to-end integration tests using the **actual trained model** and sample media files.

Run tests:
```bash
pytest -v
```

All 11 automated test cases verify:
- Root, health, and model-info endpoints
- Image inference on real face samples $\rightarrow$ verified `REAL`
- Image inference on fake face samples $\rightarrow$ verified `FAKE`
- Image with no face $\rightarrow$ correctly returns `422 Unprocessable Entity`
- Corrupt and invalid media formats $\rightarrow$ returns `400 Bad Request`
- Video inference on real video sample $\rightarrow$ verified `REAL`
- Video inference on fake video sample $\rightarrow$ verified `FAKE`
- Temporary file cleanup after processing

---

## 🌐 Production Deployment

### Model Artifact Strategy
The trained model files are:
- `models/deepfake_efficientnet_b0.onnx`: ~16.0 MB
- `models/deepfake_efficientnet_b0.pt`: ~16.3 MB

Because each file is only **16 MB**, it is well within GitHub's standard **100 MB per-file limit**. You can safely commit and push the model directly to your repository:
```bash
git add models/deepfake_efficientnet_b0.onnx
git commit -m "Add pre-trained deepfake detection model"
git push origin main
```

---

### Deploy to Render

1. **Connect GitHub Repository**:
   - Log into [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Select your repository containing this project.

2. **Configure Settings**:
   - **Environment**: `Docker`
   - **Region**: Closest to your users (e.g. Oregon, Frankfurt)
   - **Branch**: `main`
   - **Plan**: `Standard` (or `Starter` / Free)
   - **Health Check Path**: `/health`

3. **Deploy using `render.yaml` Blueprint (Optional)**:
   - If using Render Blueprints, Render automatically reads the included `render.yaml` file.

4. **Verify Deployment**:
   - Visit `https://your-service.onrender.com/health`
   - Expected status: `{"status": "healthy", "ready": true, "model_loaded": true}`

---

### Deploy to Railway

1. **New Project**:
   - Log into [Railway](https://railway.app/).
   - Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
   - Select your repository.

2. **Railway auto-detects `Dockerfile` and `railway.json`**:
   - Builds the Docker image.
   - Automatically injects the `$PORT` environment variable.
   - Sets the startup command:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
     ```

3. **Set Public Domain**:
   - In your Railway service settings $\rightarrow$ **Networking** $\rightarrow$ Click **Generate Domain**.

---

### Docker Deployment (Local or VPS)

Build and run anywhere with Docker:
```bash
# Build Docker image
docker build -t deepfake-detection-api:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name deepfake-api deepfake-detection-api:latest

# Check logs
docker logs -f deepfake-api
```

---

## ⚙️ Configuration & Environment Variables

You can customize runtime behavior by creating a `.env` file or defining environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port to bind the server |
| `HOST` | `0.0.0.0` | Host address |
| `DEBUG` | `false` | Enable FastAPI debug mode |
| `CLASSIFICATION_THRESHOLD` | `0.50` | Fake/Real decision boundary |
| `MAX_IMAGE_SIZE_MB` | `15` | Maximum image upload size in MB |
| `MAX_VIDEO_SIZE_MB` | `100` | Maximum video upload size in MB |
| `MAX_VIDEO_DURATION_SECONDS` | `120` | Maximum video duration in seconds |
| `DEFAULT_SAMPLE_FRAMES` | `20` | Default frames sampled per video |
| `MAX_SAMPLE_FRAMES` | `60` | Upper limit for sampled frames |

---

## 🛡️ Error Handling & HTTP Status Codes

| Status Code | Description | Reason |
|---|---|---|
| `200 OK` | Success | Inference complete; structured prediction returned |
| `400 Bad Request` | Client Error | Unsupported file type, empty upload, or corrupt media |
| `413 Payload Too Large` | Limit Exceeded | Upload exceeds maximum file size limit |
| `422 Unprocessable Entity` | Face Detection Required | No faces detected in media; **never falsely labelled REAL** |
| `503 Service Unavailable` | Model Not Ready | Model failed to load during startup |
