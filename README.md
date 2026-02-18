# Systemd LLM Switch

This project provides a robust **systemd-based** solution for managing local Large Language Models (LLMs) on Linux systems. It enables dynamic switching between models (Coder, Thinking, Embedding) via an intelligent Python proxy, which is crucial for efficient usage of limited VRAM.

## 🚀 Why This Project Exists

Running 30B+ parameter models on consumer hardware requires precise memory management. This project is optimized for systems with **8GB VRAM**, where running multiple large models simultaneously is impossible.

### Key Benefits:

* **VRAM Optimization**: Automatically stops unused models before starting a new one, preventing Out of Memory (OOM) crashes.
* **Roo Code (IDE Integration)**: Perfect for developers. Use the **Thinking** model for architectural planning and logic, then instantly switch to the **Coder** model for the actual implementation.
* **Open WebUI**: Functions as a backend that appears in the UI as a single OpenAI-compatible connection with multiple available models.

---

## 💻 Hardware Requirements

| Component | Minimum Requirement |
| --- | --- |
| **System RAM** | 64GB (to hold the offloaded parts of the model) |
| **VRAM (GPU)** | 8GB (NVIDIA CUDA compatible) |
| **Storage** | 100GB+ free space on SSD (for fast model loading) |

> [!TIP]
> Models run in **split-layer mode**. The configuration uses `--n-gpu-layers 49` and specific `override-tensor` instructions to force Mixture of Experts (MoE) calculations to the CPU, ensuring 30B models fit into 8GB VRAM.

---

## 🏗️ System Architecture

The system consists of three layers:

1. **Systemd Services**: Individual `llama-server` instances for each model.
2. **Python Proxy (Port 3002)**: The "brain" that receives API requests, controls systemd, and forwards queries.
3. **Backend (Port 3004)**: The currently active model instance.

```mermaid
flowchart TD
    subgraph "Client Layer"
        Client[OpenAI-compatible Client]
    end
    
    subgraph "Proxy Layer"
        WebPy[web.py Server<br/>Port 3002]
        ChatProxy[ChatProxy Handler]
        ListModels[ListModels Handler]
    end
    
    subgraph "LLM Management Layer"
        Lock[Threading Lock]
        Switcher[switch_model Logic]
    end
    
    subgraph "Systemd Layer"
        Systemctl[systemctl --user]
        Services[Multiple LLM Services<br/>qwen3-coder.service<br/>qwen3-thinking.service<br/>qwen3-coder-next.service<br/>qwen3-thinking-next.service<br/>bge-embedding.service]
    end
    
    subgraph "Backend Layer"
        Llama[llama.cpp API<br/>Port 3004]
    end
    
    Client --> WebPy
    WebPy --> ChatProxy
    WebPy --> ListModels
    ChatProxy --> Lock
    Lock --> Switcher
    Switcher --> Systemctl
    Systemctl --> Services
    Switcher -->|health check| Llama
    ChatProxy -->|forward requests| Llama
```

---

## 🛠️ Installation & Setup

**llama.cpp**
```
# 1. Clone repository
https://github.com/ggml-org/llama.cpp.git

#2. Code Update (Optional but Recommended)
git pull
git submodule update --init --recursive

# 3. Cleaning up the old build
rm -rf build-cuda
mkdir build-cuda
cd build-cuda

# 4. Configuring CMake with CUDA flag
cmake .. -DGGML_CUDA=ON

# 5. Compilation (uses all available cores for speed)
cmake --build . --config Release -j $(nproc)

# 6. After building, you'll need to:
#    - Download the required models using the llama-server command
#    - Update model paths in systemd service files according to your system setup
#      (e.g., change /home/spravce/llama/model/ to your actual model path)
```

1. **Clone the repository:**
```bash
git clone <repository-url>
cd systemd-llm-switch

```

2. **Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh

```

*The script creates a virtual environment, installs dependencies, and links the systemd services.*
3. **Configure Models:**
Adjust model paths in the files within `deploy/systemd/` and update the mappings in `src/systemd_llm_switch/config.yaml`.

---

## ⚙️ Service Management

The project utilizes `systemd --user`, so it does not require root privileges for daily operation.

| Service | Purpose | Model |
| --- | --- | --- |
| `llm-switch.service` | Main Proxy Server | `main.py` |
| `qwen3-coder.service` | Coding & Syntax | Qwen3-Coder-30B (Q8_K_XL) |
| `qwen3-thinking.service` | Logic & Planning | Qwen3-Thinking-30B (Q8_K_XL) |
| `qwen3-coder-next.service` | Coding & Syntax | Qwen3-Coder-80B (Q4_K_XL) |
| `qwen3-thinking-next.service` | Logic & Planning | Qwen3-Thinking-80B (Q4_K_XL) |
| `bge-m3.service` | Vector Search (RAG) | BGE-M3 (Q8_0) |

**Basic Commands:**

```bash
# Monitor proxy logs (including model switching)
journalctl --user -u llm-switch.service -f

# Manually stop all models (to free VRAM for gaming or other work)
systemctl --user stop qwen3-coder.service qwen3-thinking.service

```

---

## 🧪 Testing

The project includes a test suite to verify correct setup:

* **Unit Tests**: `run_tests.sh` verifies the integrity of the Python code.
* **Smoke Test**: `python3 tests/test_smoke.py` performs a real request to the proxy to verify the model starts and responds correctly.

---

## 🔗 Integrations

### Roo Code (Setup Example)

* **Provider**: OpenAI Compatible
* **Base URL**: `http://localhost:3002/v1`
* **Model ID**: `qwen3-coder-30-a3b-8gb`, `qwen3-thinking-30-a3b-8gb`, `qwen3-coder-80-a3b-8gb`, or `qwen3-thinking-80-a3b-8gb`
* **Context Window**: 20480

### Open WebUI

Add a new OpenAI connection with the URL `http://localhost:3002/v1`.

### Models download llama.cpp command
`llama-server -hf unsloth/Qwen3-Coder-Next-GGUF:Q4_K_XL`

## 🗺️ Repository Structure

```
systemd-llm-switch/
├── AGENTS.md
├── README.md
├── requirements.txt
├── setup.sh
├── run_tests.sh
├── deploy/
│   └── systemd/
│       ├── bge-embedding.service
│       ├── llm-switch.service
│       ├── qwen3-coder.service
│       ├── qwen3-coder-next.service
│       ├── qwen3-thinking.service
│       └── qwen3-thinking-next.service
├── src/
│   └── systemd_llm_switch/
│       ├── __init__.py
│       ├── config.yaml
│       └── main.py
└── tests/
    ├── __init__.py
    ├── test_main.py
    └── test_smoke.py
```

**Descriptions:**
- `deploy/systemd/`: Contains systemd service files for each model instance.
- `src/systemd_llm_switch/`: Main application code (config, proxy logic).
- `tests/`: Test suite for validation.