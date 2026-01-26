import web
import json
import subprocess
import requests
import time
import threading
import yaml
import sys
import logging
from pathlib import Path
from typing import Optional

# --------------------
# - Logging settings -
# -----------------------------------------------------------------------------

# Logging configuration that will direct output to the console
# (and systemd will capture it)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # For systemd, so that logs go to standard output
)
# -----------------------------------------------------------------------------

# -----------------
# - Configuration -
# -----------------------------------------------------------------------------
CONFIG = {}
MODELS = {}
LLAMA_URL = ""


def load_config(path: str = 'config.yaml'):
    """Loads the configuration and explicitly updates global variables."""
    global CONFIG, MODELS, LLAMA_URL
    try:
        config_path = Path(__file__).parent / path
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        if not data or 'server' not in data or 'models' not in data:
            logging.critical(
                "The configuration file is invalid "
                "or the 'server'/'models' section is missing."
            )
            sys.exit(1)

        CONFIG = data
        MODELS = data['models']
        LLAMA_URL = data['server']['llama_url']

        logging.info(f"Configuration loaded. Models: {list(MODELS.keys())}")

    except Exception as e:
        logging.critical(f"Critical error loading config.yaml: {e}")
        sys.exit(1)


load_config()
# -----------------------------------------------------------------------------


# Auxiliary function for calling systemctl --user
def run_systemctl_user(
    action: str,
    service: str
) -> subprocess.CompletedProcess:
    """Runs the systemctl --user command safely."""
    command = ["/usr/bin/systemctl", "--user", action, service]
    return subprocess.run(command, capture_output=True, text=True)


# Routing definitions for web.py
urls = (
    '/v1/chat/completions', 'ChatProxy',
    '/v1/models', 'ListModels'
)


class ChatProxy:
    """Proxy for processing requests with dynamic model switching."""
    _lock = threading.Lock()
    _current_active_model: Optional[str] = None

    def switch_model(self, target_model: str) -> bool:
        target_service = MODELS.get(target_model)
        if not target_service:
            logging.error(
                f"The '{target_model}' model was not found "
                "in the configuration."
            )
            return False

        with ChatProxy._lock:
            # 1. If we already have the model registered as active,
            # we do not take any action.
            if ChatProxy._current_active_model == target_model:
                return True

            # 2. Checking the actual status of the service in the system
            status_result = run_systemctl_user("is-active", target_service)
            if status_result.stdout.strip() == "active":
                ChatProxy._current_active_model = target_model
                return True

            # 3. Switching logic
            logging.info(f"--- Switching to model: {target_model} ---")

            # Stop all models (release VRAM)
            for srv in MODELS.values():
                run_systemctl_user("stop", srv)

            # Start selected model
            logging.info(f"I am starting the service: {target_service}")
            run_systemctl_user("start", target_service)

            # 4. Health check - waiting for API to start (max 60s)
            for _ in range(60):
                try:
                    resp = requests.get(f"{LLAMA_URL}/health", timeout=1)
                    if resp.status_code == 200:
                        logging.info(f"The {target_model} model is ready.")
                        ChatProxy._current_active_model = target_model
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
                if _ % 5 == 0:
                    logging.info(
                        f"Waiting for the model to start {target_model}..."
                    )

            logging.error(f"Model {target_model} did not start on time")
            return False

    def POST(self):
        try:
            # Reading JSON data from the request body
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})

            data = json.loads(raw_body)
            target_model = data.get("model")

            logging.info(f"Model request accepted: {target_model}")

            is_stream = data.get("stream", False)

            # Attempt to switch models
            if not self.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps(
                    {"error": f"Failed to activate model {target_model}"}
                )

            # Forwarding the request to the llama.cpp backend
            resp = requests.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json=data,
                stream=is_stream,
                timeout=(10, 1800)
            )

            if is_stream:
                # Setting headers for SSE
                web.header('Content-Type', 'text/event-stream')
                web.header('Cache-Control', 'no-cache')
                # web.header('Transfer-Encoding', 'chunked')
                web.header('Connection', 'keep-alive')
                # Important for Nginx/Proxy
                web.header('X-Accel-Buffering', 'no')

                def generate():
                    for chunk in resp.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk

                return generate()

            else:
                # Classic JSON response
                web.header('Content-Type', 'application/json')
                return resp.content

        except json.JSONDecodeError:
            web.ctx.status = "400 Bad Request"
            return json.dumps({"error": "Invalid JSON"})
        except Exception as e:
            logging.error(
                f"Unexpected error in POST handler: {e}", exc_info=True
                )
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ListModels:
    """Endpoint for listing available models."""
    def GET(self):
        web.header('Content-Type', 'application/json')
        models_list = [
            {
                "id": m,
                "object": "model",
                "owned_by": "organization"
            } for m in MODELS.keys()
        ]
        return json.dumps({"object": "list", "data": models_list})


if __name__ == "__main__":
    # 1. Setting global variables from the configuration
    server_port = CONFIG['server']['port']
    server_host = CONFIG['server']['host']

    # 2. Adjusting server startup
    # Add the host and port from the configuration to the arguments for web.py
    sys.argv.append(f'{server_host}:{server_port}')

    app = web.application(urls, globals())
    logging.info(
        f"The proxy server runs on http://{server_host}:{server_port}"
    )
    logging.info(f"Available models: {', '.join(MODELS.keys())}")
    app.run()
