# Algorithim (Cortex Zero / HALT-NN / NS-ARC)

## Overview
This repository functions as a centralized "Monorepo" or "Unified AI Platform" containing several advanced AI technologies:

1.  **Cortex Zero**: An advanced AI orchestration and optimization core.
2.  **HALT-NN (Hallucination Augmented Logical Testing Neural Network)**: A system designed to detect and prevent AI hallucinations using neural NLI, retrieval, and calibration.
3.  **NS-ARC (Neural Semantic Adaptive Robust Compression)**: A next-generation compression algorithm leveraging semantic understanding and neural networks to achieve high compression ratios for various data types (Text, DNA, Time-Series).

## Features

### Unified Interface
The `static/unified.html` dashboard provides a single entry point to experiment with:
- **Universal Compression**: Drag-and-drop compression for files.
- **Verification**: AI-powered verification of compression integrity and claims.
- **Decompression**: Secure, verified decompression and original content visualization.

### Backend API
A unified FastAPI backend (`api.py`) handles:
- **Neural NLI & Verification**: Endpoints for checking entailment and claim verification.
- **Compression**: Integrated GZIP and Semantic compression workflows.
- **System Monitoring**: Real-time status of the neural engines.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Algorithim
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Models**:
    Since large model files (`*.pt`, `*.pth`) are excluded from the repository, please contact the maintainer or run the training scripts (`generate_training_data.py`, etc.) to generate necessary checkpoints.

## Usage

### Running the Server
Start the unified backend server:
```bash
uvicorn api:app --reload
```

### Accessing the Dashboard
Open your browser and navigate to:
`http://localhost:8000/unified`

## Project Structure
- `api.py`: Main backend server entry point.
- `static/`: Frontend assets (HTML, CSS, JS).
- `cortex_zero_core/`: Core optimization logic.
- `ns_arc_core/`: Compression and verification algorithms.
- `demo_*.py`: Demonstration scripts for various components.

## License
MIT License. See `LICENSE` for more details.
