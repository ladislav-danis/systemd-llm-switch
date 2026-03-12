import web
import json
import subprocess
import requests
import time
import threading
import yaml
import sys
import logging
import base64
import uuid
import queue
from json_repair import repair_json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Fix imports for both standalone and package execution
try:
    from .db import Database
except (ImportError, ValueError):
    try:
        import db
        Database = db.Database
    except ImportError:
        try:
            from systemd_llm_switch.db import Database
        except ImportError:
            # Last resort: use the sys/Path already imported at the top
            sys.path.append(str(Path(__file__).parent))
            import db
            Database = db.Database

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
db = Database()


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
        TRACE_LOG_PATH = data['server'].get('trace_log')
        if TRACE_LOG_PATH:
            TRACE_LOG_PATH = Path(__file__).parent / TRACE_LOG_PATH

        logging.info(f"Configuration loaded. Models: {list(MODELS.keys())}")

    except Exception as e:
        logging.critical(f"Critical error loading config.yaml: {e}")
        sys.exit(1)


load_config()
# -----------------------------------------------------------------------------


def log_trace(input_raw, raw_output, final_output):
    """Logs the interaction details to a trace file for debugging.

    Args:
        input_raw: The exact raw bytes/string received from the client.
        raw_output: The exact raw bytes/string received from the backend.
        final_output: The final response after repairs.
    """
    if not TRACE_LOG_PATH:
        return

    try:
        with open(TRACE_LOG_PATH, 'a', encoding='utf-8') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- TRACE {timestamp} ---\n")
            f.write("=== INPUT ===\n")
            if isinstance(input_raw, bytes):
                f.write(input_raw.decode('utf-8', errors='replace'))
            else:
                f.write(str(input_raw))
            
            f.write("\n\n=== RAW OUTPUT ===\n")
            if isinstance(raw_output, bytes):
                f.write(raw_output.decode('utf-8', errors='replace'))
            else:
                f.write(str(raw_output))
                
            f.write("\n\n=== FINAL OUTPUT ===\n")
            if isinstance(final_output, (dict, list)):
                f.write(json.dumps(final_output, indent=2, ensure_ascii=False))
            elif isinstance(final_output, bytes):
                f.write(final_output.decode('utf-8', errors='replace'))
            else:
                f.write(str(final_output))
            f.write("\n" + "="*40 + "\n\n")
    except Exception as e:
        logging.error(f"Error writing to trace log: {e}")


# Auxiliary function for calling systemctl --user
def run_systemctl_user(
    action: str,
    service: str
) -> subprocess.CompletedProcess:
    """Executes a systemctl --user command for the specified service.

    Args:
        action: The action to perform (e.g., 'start', 'stop', 'is-active').
        service: The service name to operate on.

    Returns:
        A CompletedProcess object containing the command execution results.
    """
    command = ["/usr/bin/systemctl", "--user", action, service]
    return subprocess.run(command, capture_output=True, text=True)


# Routing definitions for web.py
urls = (
    '/v1/chat/completions', 'ChatProxy',
    '/v1/embeddings', 'EmbeddingsProxy',
    '/v1/models/([^/]+)', 'ModelDetail',
    '/v1/models', 'ListModels',
    '/v1/conversations/([^/]+)/items/([^/]+)', 'ConversationItemDetailHandler',
    '/v1/conversations/([^/]+)/items', 'ConversationItemsHandler',
    '/v1/conversations/([^/]+)', 'ConversationsDetailHandler',
    '/v1/conversations', 'ConversationsHandler',
    '/v1/responses/input_tokens', 'ResponsesInputTokensHandler',
    '/v1/responses/compact', 'ResponsesCompactHandler',
    '/v1/responses/([^/]+)/items', 'ResponsesItemsHandler',
    '/v1/responses/([^/]+)/input_items', 'ResponsesInputItemsHandler',
    '/v1/responses/([^/]+)/cancel', 'ResponsesCancelHandler',
    '/v1/responses/([^/]+)', 'ResponsesDetailHandler',
    '/v1/responses', 'ResponsesHandler',
)


def repair_tool_calls(tool_calls: List[Dict[str, Any]]) -> None:
    """Utility to repair JSON in tool call arguments if they are malformed.

    Args:
        tool_calls: A list of tool call dictionaries to be repaired in-place.
    """
    for tool in tool_calls:
        func = tool.get("function", {})
        args = func.get("arguments", "")
        if args:
            try:
                json.loads(args)
            except json.JSONDecodeError:
                tool_name = func.get("name", "unknown")
                logging.info(
                    "Repairing JSON in tool '%s' args",
                    tool_name
                )
                func["arguments"] = repair_json(args)


class ChatProxy:
    """Proxy handler for processing LLM API requests with dynamic model
    switching.

    Manages model activation through systemd services and forwards API requests
    to the active llama.cpp backend. Uses threading locks to prevent VRAM
    race conditions during model switching.

    Attributes:
        _lock: Threading lock to serialize model switching operations.
        _current_active_model: Track currently active model name.
    """

    _lock = threading.Lock()
    _current_active_model: Optional[str] = None

    @classmethod
    def switch_model(cls, target_model: str) -> bool:
        """Switches to the specified model by starting its systemd service.

        Stops all other models to free VRAM, then starts the target model
        and verifies it's operational via health check.

        Args:
            target_model: The model identifier from config.yaml.

        Returns:
            True if model was successfully activated, False otherwise.

        Raises:
            None. Logs errors but does not propagate exceptions.
        """
        target_service = MODELS.get(target_model)
        if not target_service:
            logging.error(
                f"The '{target_model}' model was not found "
                "in the configuration."
            )
            return False

        with cls._lock:
            # 1. If we already have the model registered as active,
            # we do not take any action.
            if cls._current_active_model == target_model:
                return True

            # 2. Checking the actual status of the service in the system
            status_result = run_systemctl_user("is-active", target_service)
            if status_result.stdout.strip() == "active":
                cls._current_active_model = target_model
                return True

            # 3. Switching logic
            logging.info(f"--- Switching to model: {target_model} ---")

            # Stop all models (release VRAM)
            for srv in MODELS.values():
                run_systemctl_user("stop", srv)

            # Start selected model
            logging.info(f"I am starting the service: {target_service}")
            run_systemctl_user("start", target_service)

            # 4. Health check - waiting for API to start (max 120s)
            for _ in range(120):
                try:
                    resp = requests.get(f"{LLAMA_URL}/health", timeout=1)
                    if resp.status_code == 200:
                        logging.info(f"The {target_model} model is ready.")
                        cls._current_active_model = target_model
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

    def POST(self) -> Any:
        """Processes incoming chat completions requests.

        Returns:
            A JSON response string or a generator for streaming.
        """
        try:
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})

            data = json.loads(raw_body)
            client_wants_stream = data.get("stream", False)
            data["stream"] = False
            target_model = data.get("model")

            logging.info(f"Model request accepted: {target_model}")

            if not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps({"error": f"Failed to activate model {target_model}"})

            if client_wants_stream:
                web.header('Content-Type', 'text/event-stream')
                web.header('Cache-Control', 'no-cache')
                web.header('Connection', 'keep-alive')
                web.header('X-Accel-Buffering', 'no')

                def chat_stream_handler():
                    # Send initial comment to keep connection alive
                    yield b": keep-alive\n\n"
                    
                    q = queue.Queue()
                    def fetch_backend():
                        try:
                            resp = requests.post(f"{LLAMA_URL}/v1/chat/completions", json=data, timeout=(10, 1800))
                            q.put(resp)
                        except Exception as e:
                            q.put(e)

                    t = threading.Thread(target=fetch_backend)
                    t.start()

                    # Wait for thread with heartbeats
                    while t.is_alive():
                        yield b": keep-alive\n\n"
                        # For ChatProxy (OpenAI compatibility), we send a heartbeat as a chunk
                        heartbeat_chunk = {
                            "id": data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": target_model,
                            "choices": [],
                            "heartbeat": True
                        }
                        yield f"data: {json.dumps(heartbeat_chunk)}\n\n".encode('utf-8')
                        try:
                            result = q.get(timeout=2)
                            break
                        except queue.Empty:
                            continue
                    else:
                        result = q.get()

                    if isinstance(result, Exception):
                        yield f"data: {json.dumps({'error': str(result)})}\n\n".encode('utf-8')
                        return

                    if result.status_code != 200:
                        yield f"data: {json.dumps({'error': 'Backend error'})}\n\n".encode('utf-8')
                        return

                    resp_data = result.json()
                    choices = resp_data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        tcs = message.get("tool_calls", [])
                        if tcs:
                            repair_tool_calls(tcs)
                            message["content"] = None
                            # Add index to tool_calls for streaming compatibility
                            for i, tc in enumerate(tcs):
                                if "index" not in tc:
                                    tc["index"] = i
                    
                    log_trace(raw_body, result.content, resp_data)

                    chunk_data = resp_data.copy()
                    chunk_data["object"] = "chat.completion.chunk"
                    if choices:
                        choice = chunk_data["choices"][0]
                        if "message" in choice:
                            choice["delta"] = choice.pop("message")
                    
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"

                return chat_stream_handler()
            else:
                resp = requests.post(f"{LLAMA_URL}/v1/chat/completions", json=data, timeout=(10, 1800))
                resp_data = resp.json()
                if "choices" in resp_data:
                    message = resp_data["choices"][0].get("message", {})
                    tcs = message.get("tool_calls", [])
                    if tcs:
                        repair_tool_calls(tcs)
                        message["content"] = None
                log_trace(raw_body, resp.content, resp_data)
                web.header('Content-Type', 'application/json')
                return json.dumps(resp_data)

        except Exception as e:
            logging.error(f"Error in ChatProxy: {e}", exc_info=True)
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ModelDetail:
    """Endpoint for retrieving details of a specific model."""

    def GET(self, model_id: str) -> str:
        """Retrieves metadata for a specific model.

        Args:
            model_id: The identifier of the model.

        Returns:
            A JSON string with model details.
        """
        if model_id not in MODELS:
            web.ctx.status = "404 Not Found"
            return json.dumps({"error": "Model not found"})
        
        web.header('Content-Type', 'application/json')
        return json.dumps({
            "id": model_id,
            "object": "model",
            "owned_by": "organization",
            "created": int(time.time()),
            "capabilities": {"embeddings": model_id == "bge-m3"}
        })


class ListModels:
    """Endpoint for listing available models."""

    def GET(self) -> str:
        """Lists all configured models.

        Returns:
            A JSON string containing a list of model objects.
        """
        web.header('Content-Type', 'application/json')
        models_list = [
            {
                "id": m,
                "object": "model",
                "owned_by": "organization"
            } for m in MODELS.keys()
        ]
        return json.dumps({"object": "list", "data": models_list})


class EmbeddingsProxy:
    """Proxy handler for processing embedding requests."""

    def POST(self) -> str:
        """Forwards embedding requests to the backend after model switching.

        Returns:
            A JSON string with embeddings or an error message.
        """
        try:
            raw_body = web.data()
            data = json.loads(raw_body)
            target_model = data.get("model")
            if not target_model:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "Model is required"})

            if not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps({"error": f"Failed to activate model {target_model}"})

            resp = requests.post(f"{LLAMA_URL}/v1/embeddings", json=data, timeout=(10, 600))
            if resp.status_code != 200:
                web.ctx.status = f"{resp.status_code} {resp.reason}"
                try:
                    return json.dumps(resp.json())
                except Exception:
                    return json.dumps({"error": f"Backend returned {resp.status_code}"})

            resp_data = resp.json()
            if "data" in resp_data:
                for i, item in enumerate(resp_data["data"]):
                    item.setdefault("object", "embedding")
                    item.setdefault("index", i)
            resp_data.setdefault("object", "list")
            resp_data["model"] = target_model

            web.header('Content-Type', 'application/json')
            return json.dumps(resp_data)
        except Exception as e:
            logging.error(f"Error in EmbeddingsProxy: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ConversationsHandler:
    """Handler for managing conversation creation."""

    def POST(self) -> str:
        """Creates a new conversation, optionally with initial items.

        Returns:
            A JSON string representing the created conversation.
        """
        try:
            raw_body = web.data()
            data = json.loads(raw_body) if raw_body else {}
            conv_id = db.create_conversation(metadata=data.get("metadata", {}))
            for item in data.get("items", []):
                if item.get("type") == "message":
                    db.add_item(conv_id, None, "message", item.get("role", "user"), item.get("content"))
            
            web.header('Content-Type', 'application/json')
            return json.dumps(db.get_conversation(conv_id))
        except Exception as e:
            logging.error(f"Error in ConversationsHandler: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ConversationsDetailHandler:
    """Handler for retrieving, updating or deleting a specific conversation."""
    def GET(self, conversation_id):
        conv = db.get_conversation(conversation_id)
        if not conv:
            web.ctx.status = "404 Not Found"
            return json.dumps({"error": "Conversation not found"})
        web.header('Content-Type', 'application/json')
        return json.dumps(conv)

    def POST(self, conversation_id):
        raw_body = web.data()
        data = json.loads(raw_body)
        if db.update_conversation(conversation_id, data.get("metadata", {})):
            web.header('Content-Type', 'application/json')
            return json.dumps(db.get_conversation(conversation_id))
        web.ctx.status = "404 Not Found"
        return json.dumps({"error": "Conversation not found"})

    def DELETE(self, conversation_id):
        if db.delete_conversation(conversation_id):
            web.header('Content-Type', 'application/json')
            return json.dumps({"id": conversation_id, "object": "conversation.deleted", "deleted": True})
        web.ctx.status = "404 Not Found"
        return json.dumps({"error": "Conversation not found"})


class ConversationItemsHandler:
    """Handler for listing or creating items within a conversation."""
    def GET(self, conversation_id):
        params = web.input(limit=20, after=None, order="desc")
        items = db.get_conversation_items(conversation_id, limit=int(params.limit), after=params.after, order=params.order)
        web.header('Content-Type', 'application/json')
        return json.dumps({"object": "list", "data": items, "has_more": len(items) == int(params.limit)})

    def POST(self, conversation_id):
        data = json.loads(web.data())
        added = []
        for item_data in data.get("items", []):
            if item_data.get("type") == "message":
                item_id = db.add_item(conversation_id, None, "message", item_data.get("role", "user"), item_data.get("content"))
                added.append(db.get_item(item_id))
        web.header('Content-Type', 'application/json')
        return json.dumps({"object": "list", "data": added, "has_more": False})


class ConversationItemDetailHandler:
    def GET(self, conversation_id, item_id):
        item = db.get_item(item_id)
        if not item:
            web.ctx.status = "404 Not Found"
            return json.dumps({"error": "Item not found"})
        web.header('Content-Type', 'application/json')
        return json.dumps(item)

    def DELETE(self, conversation_id, item_id):
        if db.delete_item(item_id):
            web.header('Content-Type', 'application/json')
            return json.dumps(db.get_conversation(conversation_id))
        web.ctx.status = "404 Not Found"
        return json.dumps({"error": "Item not found"})


class ResponsesHandler:
    """Proxy handler for the stateful OpenAI Responses API."""
    def POST(self):
        try:
            raw_body = web.data()
            data = json.loads(raw_body)
            client_wants_stream = data.get("stream", False)
            target_model = data.get("model")
            conv_id = data.get("conversation") or data.get("conversation_id")
            prev_resp_id = data.get("previous_response_id")
            input_items = data.get("input", [])
            instructions = data.get("instructions")
            metadata = data.get("metadata", {})

            if not target_model or not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps({"error": f"Failed to activate model {target_model}"})

            if not conv_id:
                conv_id = db.create_conversation()
            
            current_messages = []
            for item in input_items:
                itype = item.get("type")
                if itype == "input_text":
                    db.add_item(conv_id, None, "message", "user", {"text": item.get("text")})
                    current_messages.append({"role": "user", "content": item.get("text")})
                elif itype == "function_call_output":
                    db.add_item(conv_id, None, "tool", "tool", {"tool_call_id": item.get("call_id"), "content": item.get("output")})
                    current_messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output")})
                elif itype == "compaction":
                    try:
                        summary = base64.b64decode(item.get("encrypted_content", "")).decode('utf-8')
                    except Exception:
                        summary = item.get("encrypted_content", "")
                    current_messages.append({"role": "system", "content": f"Previous conversation summary: {summary}"})

            resp_id = db.create_response(conv_id, target_model, status="in_progress", instructions=instructions, metadata=metadata)

            if client_wants_stream:
                web.header('Content-Type', 'text/event-stream')
                web.header('Cache-Control', 'no-cache')
                web.header('Connection', 'keep-alive')
                web.header('X-Accel-Buffering', 'no')

                def response_stream_handler():
                    seq_counter = [0]
                    def sse(event, data_obj):
                        seq_counter[0] += 1
                        # Create a envelope that matches the Responses API protocol
                        envelope = data_obj.copy()
                        if "type" not in envelope:
                            envelope["type"] = event
                        if "sequence_number" not in envelope:
                            envelope["sequence_number"] = seq_counter[0]
                        return f"event: {event}\ndata: {json.dumps(envelope)}\n\n".encode('utf-8')

                    resp_obj = db.get_response(resp_id)
                    yield sse("response.created", resp_obj)
                    yield sse("response.in_progress", {"response": resp_obj})
                    
                    q = queue.Queue()
                    def fetch_backend():
                        try:
                            hist = db.get_conversation_history(conv_id, up_to_response_id=prev_resp_id)
                            if prev_resp_id: hist.extend(current_messages)
                            if instructions: hist.insert(0, {"role": "system", "content": instructions})
                            
                            tools = []
                            for t in data.get("tools", []):
                                if t.get("type") == "function":
                                    tools.append({"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""), "parameters": t.get("parameters", {})}})
                            
                            payload = {"model": target_model, "messages": hist, "stream": False}
                            if tools: payload["tools"] = tools
                            q.put(requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, timeout=(10, 1800)))
                        except Exception as e:
                            q.put(e)

                    t = threading.Thread(target=fetch_backend)
                    t.start()

                    last_in_progress_time = time.time()
                    while t.is_alive():
                        yield b": keep-alive\n\n"
                        
                        now = time.time()
                        if now - last_in_progress_time >= 2.0:
                            # Send official in_progress event as a keep-alive
                            current_resp = db.get_response(resp_id)
                            yield sse("response.in_progress", {"response": current_resp})
                            last_in_progress_time = now
                        else:
                            # Still send a heartbeat for transport-level keep-alive
                            yield sse("response.heartbeat", {"response_id": resp_id})
                            
                        try:
                            # Check for result every 1s
                            result = q.get(timeout=1)
                            break
                        except queue.Empty:
                            continue
                    else:
                        result = q.get()

                    if isinstance(result, Exception) or result.status_code != 200:
                        yield sse("error", {"message": str(result)})
                        return

                    resp_data = result.json()
                    msg = resp_data["choices"][0].get("message", {})
                    reasoning = msg.get("reasoning_content")
                    
                    if msg.get("content"):
                        db.add_item(conv_id, resp_id, "message", "assistant", {"text": msg.get("content")})
                    
                    t_calls = msg.get("tool_calls", [])
                    repair_tool_calls(t_calls)
                    for tc in t_calls:
                        f = tc.get("function", {})
                        db.add_item(conv_id, resp_id, "function_call", "assistant", {"call_id": tc.get("id"), "name": f.get("name"), "arguments": f.get("arguments")})

                    db.update_response(resp_id, "completed", usage=resp_data.get("usage", {}), metadata=metadata)
                    final_obj = db.get_response(resp_id)

                    for out_idx, item in enumerate(final_obj.get("output", [])):
                        yield sse("response.output_item.added", {"item": item})
                        if item["type"] == "message":
                            for p_idx, part in enumerate(item.get("content", [])):
                                yield sse("response.content_part.added", {"response_id": resp_id, "output_index": out_idx, "content_index": p_idx, "part": part})
                                if part["type"] == "output_text":
                                    if reasoning: yield sse("response.reasoning.delta", {"response_id": resp_id, "output_index": out_idx, "content_index": p_idx, "delta": reasoning})
                                    yield sse("response.text.delta", {"response_id": resp_id, "output_index": out_idx, "content_index": p_idx, "delta": part["text"]})
                                yield sse("response.content_part.done", {"response_id": resp_id, "output_index": out_idx, "content_index": p_idx, "part": part})
                        elif item["type"] == "function_call":
                            yield sse("response.function_call.arguments.delta", {"response_id": resp_id, "output_index": out_idx, "call_id": item.get("call_id"), "delta": item.get("arguments")})
                        yield sse("response.output_item.done", {"item": item})
                    
                    # Send COMPLETED and DONE for multi-client compatibility
                    yield sse("response.completed", final_obj)
                    yield sse("response.done", {"response": final_obj})
                    yield b"data: [DONE]\n\n"

                return response_stream_handler()
            else:
                # Sync mode (simplified)
                hist = db.get_conversation_history(conv_id, up_to_response_id=prev_resp_id)
                if prev_resp_id: hist.extend(current_messages)
                if instructions: hist.insert(0, {"role": "system", "content": instructions})
                
                resp = requests.post(f"{LLAMA_URL}/v1/chat/completions", json={"model": target_model, "messages": hist, "stream": False}, timeout=(10, 1800))
                resp_data = resp.json()
                msg = resp_data["choices"][0].get("message", {})
                if msg.get("content"):
                    db.add_item(conv_id, resp_id, "message", "assistant", {"text": msg.get("content")})
                db.update_response(resp_id, "completed", usage=resp_data.get("usage", {}), metadata=metadata)
                web.header('Content-Type', 'application/json')
                return json.dumps(db.get_response(resp_id))

        except Exception as e:
            logging.error(f"Error in ResponsesHandler: {e}", exc_info=True)
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesDetailHandler:
    def GET(self, response_id):
        res = db.get_response(response_id)
        if not res:
            web.ctx.status = "404 Not Found"
            return json.dumps({"error": "Response not found"})
        web.header('Content-Type', 'application/json')
        return json.dumps(res)

    def DELETE(self, response_id):
        if db.delete_response(response_id):
            web.header('Content-Type', 'application/json')
            return json.dumps({"id": response_id, "object": "response", "deleted": True})
        web.ctx.status = "404 Not Found"
        return json.dumps({"error": "Response not found"})


class ResponsesCompactHandler:
    def POST(self):
        try:
            data = json.loads(web.data())
            target_model = data.get("model")
            if not target_model or not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps({"error": "Model activation failed"})

            full_text = ""
            for item in data.get("input", []):
                if isinstance(item, str): full_text += item + "\n"
                elif isinstance(item, dict):
                    if item.get("type") == "input_text": full_text += f"User: {item.get('text')}\n"
                    elif item.get("role") == "assistant":
                        content = item.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if c.get("type") == "output_text": full_text += f"Assistant: {c.get('text')}\n"
                        elif isinstance(content, str): full_text += f"Assistant: {content}\n"

            resp = requests.post(f"{LLAMA_URL}/v1/chat/completions", json={"model": target_model, "messages": [{"role": "user", "content": f"Summarize:\n{full_text}"}], "stream": False}, timeout=(10, 600))
            summary = resp.json()["choices"][0]["message"]["content"]
            web.header('Content-Type', 'application/json')
            return json.dumps({"id": f"resp_{uuid.uuid4().hex[:12]}", "object": "response", "created_at": int(time.time()), "model": target_model, "status": "completed", "output": [{"id": f"cmp_{uuid.uuid4().hex[:12]}", "type": "compaction", "encrypted_content": base64.b64encode(summary.encode()).decode()}], "usage": resp.json().get("usage", {})})
        except Exception as e:
            logging.error(f"Error in compact: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesItemsHandler:
    def GET(self, response_id):
        items = db.get_response_items(response_id)
        web.header('Content-Type', 'application/json')
        return json.dumps({"object": "list", "data": items, "has_more": False})


class ResponsesInputItemsHandler:
    def GET(self, response_id):
        res = db.get_response(response_id)
        if not res:
            web.ctx.status = "404 Not Found"
            return json.dumps({"error": "Response not found"})
        history = db.get_conversation_history(res["conversation_id"], up_to_response_id=response_id)
        items = [{"id": f"input_{response_id}_{i}", "object": "item", "type": "message", "role": m["role"], "content": [{"type": "output_text", "text": m["content"]}]} for i, m in enumerate(history)]
        web.header('Content-Type', 'application/json')
        return json.dumps({"object": "list", "data": items, "has_more": False})


class ResponsesInputTokensHandler:
    def POST(self):
        data = json.loads(web.data())
        total = sum(len(i.get("text", "")) for i in data.get("input", []) if isinstance(i, dict))
        web.header('Content-Type', 'application/json')
        return json.dumps({"object": "response.input_tokens", "input_tokens": total // 4})


class ResponsesCancelHandler:
    def POST(self, response_id):
        res = db.get_response(response_id)
        if not res:
            web.ctx.status = "404 Not Found"
            return json.dumps({"error": "Response not found"})
        web.header('Content-Type', 'application/json')
        return json.dumps(res)


app = web.application(urls, globals())

if __name__ == "__main__":
    server_port = CONFIG['server']['port']
    server_host = CONFIG['server']['host']
    sys.argv.append(f'{server_host}:{server_port}')
    logging.info(f"The proxy server runs on http://{server_host}:{server_port}")
    app.run()
