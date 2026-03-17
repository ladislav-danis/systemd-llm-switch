import web
import json
import subprocess
import requests
import time
import threading
import yaml
import sys
import logging
from json_repair import repair_json
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
TRACE_LOG_PATH = None


def load_config(path: str = 'config.yaml'):
    """Loads configuration from YAML file and initializes global variables.

    Args:
        path: Path to the configuration file relative to the script location.

    Returns:
        None. Updates global CONFIG, MODELS, and LLAMA_URL variables.

    Raises:
        SystemExit: If configuration file is invalid or missing required
        sections.
    """
    global CONFIG, MODELS, LLAMA_URL, TRACE_LOG_PATH
    try:
        config_path = Path(__file__).parent / path
        if not config_path.exists():
            logging.critical(f"Configuration file not found at: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None or 'server' not in data or 'models' not in data:
            logging.critical(
                "The configuration file is invalid "
                "or the 'server'/'models' section is missing."
            )
            sys.exit(1)

        server_cfg = data['server']
        if 'host' not in server_cfg or 'port' not in server_cfg:
            logging.critical(
                "The 'server' section in config.yaml must contain "
                "'host' and 'port' keys."
            )
            sys.exit(1)

        CONFIG = data
        MODELS = data['models']
        if not MODELS:
            logging.critical("No models defined in configuration.")
            sys.exit(1)

        LLAMA_URL = server_cfg.get('llama_url')
        if not LLAMA_URL:
            logging.critical("'llama_url' is missing in 'server' configuration.")
            sys.exit(1)

        TRACE_LOG_PATH = server_cfg.get('trace_log')
        if TRACE_LOG_PATH:
            TRACE_LOG_PATH = Path(__file__).parent / TRACE_LOG_PATH

        logging.info(f"Configuration loaded. Models: {list(MODELS.keys())}")

    except Exception as e:
        logging.critical(f"Critical error loading config.yaml: {e}")
        sys.exit(1)


load_config()
# -----------------------------------------------------------------------------


def log_trace(
    input_raw: any,
    raw_output: any,
    final_output: any
) -> None:
    """Logs the interaction details to a trace file for debugging.

    Args:
        input_raw: The exact raw bytes/string received from the client.
        raw_output: The exact raw bytes/string received from the backend.
        final_output: The final response after repairs.
    """
    if not TRACE_LOG_PATH:
        return

    def format_data(data):
        if data is None:
            return "<NONE>"
        
        # Maximum size for full logging (1 MB)
        MAX_LOG_SIZE = 1024 * 1024

        if isinstance(data, (dict, list)):
            try:
                text = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                text = str(data)
        elif isinstance(data, bytes):
            if len(data) > MAX_LOG_SIZE:
                return f"<BINARY DATA: {len(data)} bytes, TOO LARGE TO LOG>"
            text = data.decode('utf-8', errors='replace')
        else:
            text = str(data)

        if len(text) > MAX_LOG_SIZE:
            return f"<DATA: {len(text)} characters, TOO LARGE TO LOG (truncated)>"
        return text

    try:
        with open(TRACE_LOG_PATH, 'a', encoding='utf-8') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- TRACE {timestamp} ---\n")
            f.write("=== INPUT ===\n")
            f.write(format_data(input_raw))
            f.write("\n\n=== RAW OUTPUT ===\n")
            f.write(format_data(raw_output))
            f.write("\n\n=== FINAL OUTPUT ===\n")
            f.write(format_data(final_output))
            f.write("\n" + "="*40 + "\n\n")
    except Exception as e:
        logging.error(f"Error writing to trace log: {e}")


# Auxiliary function for calling systemctl --user
def run_systemctl_user(
    action: str,
    service: str,
    now: bool = False,
    timeout: int = 30
) -> subprocess.CompletedProcess:
    """Executes a systemctl --user command for the specified service.

    Args:
        action: The action to perform (e.g., 'start', 'stop', 'is-active').
        service: The service name to operate on.
        now: If True, adds --now flag (for stop/start).
        timeout: Maximum time to wait for the command.

    Returns:
        A CompletedProcess object containing the command execution results.
    """
    command = ["/usr/bin/systemctl", "--user", action]
    if now and action in ("start", "stop", "restart"):
        command.append("--now")
    command.append(service)
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.error(f"Command '{' '.join(command)}' timed out after {timeout}s")
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout="",
            stderr="Command timed out"
        )


# Routing definitions for web.py
urls = (
    '/v1/chat/completions', 'ChatProxy',
    '/v1/embeddings', 'EmbeddingsProxy',
    '/v1/models', 'ListModels'
)


class BaseModelProxy:
    """Base proxy handler with model switching capabilities.

    Attributes:
        _lock: Reentrant threading lock to serialize model switching.
        _current_active_model: Track currently active model name.
    """

    _lock = threading.RLock()
    _current_active_model: Optional[str] = None

    def _get_validated_data(self) -> tuple[Optional[dict], bytes]:
        """Reads and validates the request body.

        Returns:
            A tuple of (parsed JSON data or None, raw bytes body).
        """
        raw_body = b""
        try:
            raw_body = web.data()
            
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return None, raw_body

            # Payload size limit (10 MB)
            if len(raw_body) > 10 * 1024 * 1024:
                web.ctx.status = "413 Payload Too Large"
                return None, raw_body

            data = json.loads(raw_body)
            target_model = data.get("model")

            if not target_model:
                web.ctx.status = "400 Bad Request"
                return None, raw_body

            return data, raw_body

        except json.JSONDecodeError:
            web.ctx.status = "400 Bad Request"
            err_msg = {"error": "Invalid JSON"}
            log_trace(raw_body, None, err_msg)
            return None, raw_body
        except Exception as e:
            logging.error(f"Error in _get_validated_data: {e}", exc_info=True)
            web.ctx.status = "500 Internal Server Error"
            log_trace(raw_body, None, {"error": str(e)})
            return None, raw_body

    def switch_model(self, target_model: str) -> bool:
        """Switches to the specified model by starting its systemd service.

        Stops all other models to free VRAM, then starts the target model
        and verifies it's operational via health check.

        Args:
            target_model: The model identifier from config.yaml.

        Returns:
            True if model was successfully activated, False otherwise.
        """
        if not target_model:
            logging.error("No target model specified.")
            return False

        target_service = MODELS.get(target_model)
        if not target_service:
            logging.error(
                f"The '{target_model}' model was not found "
                "in the configuration."
            )
            return False

        with BaseModelProxy._lock:
            # 1. If we already have the model registered as active,
            # we check if it's REALLY active in systemd.
            if BaseModelProxy._current_active_model == target_model:
                status_result = run_systemctl_user("is-active", target_service)
                if status_result.stdout.strip() == "active":
                    return True

            # 2. Switching logic - always performed under LOCK
            logging.info(f"--- Switching to model: {target_model} ---")
            previous_model = BaseModelProxy._current_active_model

            # Stop all other models (release VRAM)
            for model_id, srv in MODELS.items():
                if model_id == target_model:
                    continue
                
                stop_res = run_systemctl_user("stop", srv, now=True)
                if stop_res.returncode != 0:
                    logging.error(f"Failed to stop service {srv}: {stop_res.stderr}")

            # Even if stop failed, we continue but mark state as unknown for now
            BaseModelProxy._current_active_model = None

            # Reset failed state for target service
            run_systemctl_user("reset-failed", target_service)

            # Start selected model
            logging.info(f"I am starting the service: {target_service}")
            start_result = run_systemctl_user("start", target_service)
            if start_result.returncode != 0:
                logging.error(
                    f"Failed to start service {target_service}: "
                    f"{start_result.stderr}"
                )
                self._rollback(previous_model)
                return False

            # 3. Health check - waiting for API to start (max 120s)
            success = self._wait_for_ready(target_model, timeout=120)

            if not success:
                logging.error(f"Model {target_model} did not start on time")
                self._rollback(previous_model)
                return False

            BaseModelProxy._current_active_model = target_model
            return True

    def _wait_for_ready(self, model_name: str, timeout: int = 120) -> bool:
        """Waits for the model's API to be ready.

        Args:
            model_name: Name of the model to check.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if ready, False otherwise.
        """
        for i in range(timeout):
            try:
                # Check health endpoint
                resp = requests.get(f"{LLAMA_URL}/health", timeout=1)
                if resp.status_code == 200:
                    # Double check with models endpoint to ensure full readiness
                    models_resp = requests.get(
                        f"{LLAMA_URL}/v1/models",
                        timeout=1
                    )
                    if models_resp.status_code == 200:
                        return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            if i % 5 == 0:
                logging.info(f"Waiting for {model_name} to be ready...")
        return False

    def _rollback(self, previous_model: Optional[str]) -> None:
        """Attempts to restore the previously active model if switch fails.

        Args:
            previous_model: The model ID that was active before the failed switch.
        """
        # We assume we are already under lock if called from switch_model,
        # but re-entering RLock is safe.
        with BaseModelProxy._lock:
            if not previous_model or previous_model not in MODELS:
                # Attempt to find if any model is actually active
                self._sync_active_model()
                return

            logging.info(f"Rolling back to previous model: {previous_model}")
            prev_service = MODELS[previous_model]
            run_systemctl_user("reset-failed", prev_service)
            run_systemctl_user("start", prev_service)

            # Verify rollback with a shorter health check (max 30s)
            if self._wait_for_ready(previous_model, timeout=30):
                logging.info(f"Rollback to {previous_model} successful.")
                BaseModelProxy._current_active_model = previous_model
            else:
                logging.error(f"Rollback to {previous_model} failed.")
                # We failed even to rollback, sync with reality
                self._sync_active_model()

    def _sync_active_model(self) -> None:
        """Synchronizes _current_active_model with the actual systemd state."""
        with BaseModelProxy._lock:
            BaseModelProxy._current_active_model = None
            for model_id, srv in MODELS.items():
                res = run_systemctl_user("is-active", srv)
                if res.stdout.strip() == "active":
                    BaseModelProxy._current_active_model = model_id
                    break


class ChatProxy(BaseModelProxy):
    """Proxy handler for processing LLM chat completion requests."""

    def POST(self):
        """Processes incoming chat completions requests.

        Switches to the requested model, forwards the request to the active
        llama.cpp backend, and handles response formatting including JSON
        repair for malformed tool call arguments.

        Args:
            None. Reads request data from web.ctx.

        Returns:
            JSON string or generator for the response body.

        Raises:
           web.ctx.status: Set to "400 Bad Request" for invalid JSON,
                "500 Internal Server Error" for activation failures.
        """
        client_wants_stream = False
        data, raw_body = self._get_validated_data()
        
        if data is None:
            if web.ctx.status.startswith("400"):
                return json.dumps({"error": "Invalid request or missing model"})
            return json.dumps({"error": "Failed to parse request"})

        target_model = data.get("model")
        client_wants_stream = data.get("stream", False)

        # Entire request-response cycle is locked to prevent race conditions
        with BaseModelProxy._lock:
            try:
                # 1. Switch to the requested model
                if not self.switch_model(target_model):
                    web.ctx.status = "500 Internal Server Error"
                    return json.dumps({"error": f"Failed to activate model {target_model}"})

                # 2. Prepare the request (ensure non-streaming backend call)
                backend_data = data.copy()
                backend_data["stream"] = False

                # 3. Forwarding the request to the llama.cpp backend
                resp = requests.post(
                    f"{LLAMA_URL}/v1/chat/completions",
                    json=backend_data,
                    stream=False,
                    timeout=(10, 1800)
                )

                raw_resp_content = resp.content
                try:
                    resp_data = resp.json()
                    # If it's a chat completion, try to repair the content
                    choices = resp_data.get("choices", [])
                    
                    for choice in choices:
                        message = choice.get("message", {})

                        # 1. Ensure content is null if tool_calls present
                        #    (standard OpenAI)
                        tool_calls = message.get("tool_calls", [])
                        if tool_calls and not message.get("content"):
                            message["content"] = None

                        # 2. Repair tool_calls arguments if present
                        for tool in tool_calls:
                            func = tool.get("function", {})
                            args = func.get("arguments")
                            if args:
                                # repair_json handles valid JSON and repairs invalid ones
                                repaired_args = repair_json(args)
                                if repaired_args != args:
                                    logging.info(
                                        "Repaired JSON in tool '%s' args",
                                        func.get("name", "unknown")
                                    )
                                    func["arguments"] = repaired_args

                    log_trace(raw_body, raw_resp_content, resp_data)

                    if client_wants_stream:
                        # Client requested a stream, so we must fake an SSE stream
                        web.header('Content-Type', 'text/event-stream')
                        web.header('Cache-Control', 'no-cache')
                        web.header('Connection', 'keep-alive')
                        web.header('X-Accel-Buffering', 'no')

                        def fake_stream(repaired_data):
                            # Convert chat.completion to chat.completion.chunk
                            chunk_data = repaired_data.copy()
                            chunk_data["object"] = "chat.completion.chunk"

                            choices_list = chunk_data.get("choices", [])
                            if choices_list:
                                choice = choices_list[0]
                                if "message" in choice:
                                    choice["delta"] = choice.pop("message")
                                    # OpenAI stream format for tool_calls
                                    # uses index
                                    if "tool_calls" in choice["delta"]:
                                        for i, tc in enumerate(
                                            choice["delta"]["tool_calls"]
                                        ):
                                            tc["index"] = i

                            yield (
                                f"data: {json.dumps(chunk_data)}\n\n"
                            ).encode('utf-8')
                            yield b"data: [DONE]\n\n"

                        return fake_stream(resp_data)
                    else:
                        # Classic JSON response
                        web.header('Content-Type', 'application/json')
                        return json.dumps(resp_data)

                except Exception as e:
                    logging.warning(
                        "Could not parse or repair backend response: %s", e
                    )
                    error_detail = f"ERROR: {e}"

                    content_text = ""
                    if raw_resp_content is not None:
                        content_text = raw_resp_content.decode('utf-8', errors='replace')
                        error_detail += "\nORIGINAL CONTENT: " + content_text

                    log_trace(raw_body, raw_resp_content, error_detail)

                    if client_wants_stream:
                        web.header('Content-Type', 'text/event-stream')
                        def stream_error(content_str):
                            err_msg = {
                                "error": {
                                    "message": "Failed to parse backend response",
                                    "details": content_str
                                }
                            }
                            yield f"data: {json.dumps(err_msg)}\n\n".encode('utf-8')
                            yield b"data: [DONE]\n\n"
                        return stream_error(content_text)

                    return raw_resp_content or json.dumps({"error": error_detail})
            except Exception as e:
                logging.error(
                    f"Unexpected error in POST handler: {e}", exc_info=True
                    )
                web.ctx.status = "500 Internal Server Error"
                err_msg = {"error": str(e)}
                log_trace(raw_body, None, err_msg)
                
                if client_wants_stream:
                    web.header('Content-Type', 'text/event-stream')
                    def stream_exception(msg):
                        yield f"data: {json.dumps({'error': msg})}\n\n".encode('utf-8')
                        yield b"data: [DONE]\n\n"
                    return stream_exception(str(e))
                
                return json.dumps(err_msg)



class EmbeddingsProxy(BaseModelProxy):
    """Proxy handler for processing LLM embeddings requests."""

    def POST(self):
        """Processes incoming embeddings requests.

        Switches to the requested model and forwards the request to the active
        llama.cpp backend.
        """
        data, raw_body = self._get_validated_data()
        if data is None:
            if web.ctx.status.startswith("400"):
                return json.dumps({"error": "Invalid request or missing model"})
            return json.dumps({"error": "Failed to parse request"})

        target_model = data.get("model")

        # Entire request-response cycle is locked to prevent race conditions
        with BaseModelProxy._lock:
            try:
                # 1. Switch to the requested model
                if not self.switch_model(target_model):
                    web.ctx.status = "500 Internal Server Error"
                    return json.dumps({"error": f"Failed to activate model {target_model}"})

                # 2. Forwarding the request to the llama.cpp backend
                resp = requests.post(
                    f"{LLAMA_URL}/v1/embeddings",
                    json=data,
                    timeout=(10, 600)
                )

                if resp.status_code != 200:
                    logging.error(
                        "Embeddings backend returned error %d: %s",
                        resp.status_code, resp.text
                    )
                    web.ctx.status = f"{resp.status_code} Error"
                    log_trace(raw_body, resp.content, f"BACKEND ERROR: {resp.status_code}")
                    return resp.content

                # Response size limit for embeddings (50 MB)
                if len(resp.content) > 50 * 1024 * 1024:
                    logging.error(
                        "Embeddings response too large: %d bytes",
                        len(resp.content)
                    )
                    web.ctx.status = "502 Bad Gateway"
                    err_msg = {"error": "Response from backend too large"}
                    log_trace(raw_body, resp.content, err_msg)
                    return json.dumps(err_msg)

                resp_data = None
                try:
                    # Basic validation of embeddings response
                    resp_data = resp.json()
                    if "data" not in resp_data:
                        logging.warning(
                            "Embeddings response missing 'data' field: %s",
                            resp_data
                        )
                except Exception as e:
                    logging.warning("Could not parse embeddings JSON: %s", e)

                log_trace(raw_body, resp.content, resp_data or resp.content)
                web.header('Content-Type', 'application/json')
                return resp.content

            except Exception as e:
                logging.error(
                    f"Unexpected error in EmbeddingsProxy POST: {e}", exc_info=True
                )
                web.ctx.status = "500 Internal Server Error"
                err_msg = {"error": str(e)}
                log_trace(raw_body, None, err_msg)
                return json.dumps(err_msg)


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
