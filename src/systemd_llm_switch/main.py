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
    '/v1/models', 'ListModels',
    '/v1/conversations', 'ConversationsHandler',
    '/v1/conversations/([^/]+)', 'ConversationsDetailHandler',
    '/v1/responses', 'ResponsesHandler',
    '/v1/responses/input_tokens', 'ResponsesInputTokensHandler',
    '/v1/responses/compact', 'ResponsesCompactHandler',
    '/v1/responses/([^/]+)', 'ResponsesDetailHandler',
    '/v1/responses/([^/]+)/items', 'ResponsesItemsHandler',
    '/v1/responses/([^/]+)/input_items', 'ResponsesInputItemsHandler',
    '/v1/responses/([^/]+)/cancel', 'ResponsesCancelHandler'
)


def repair_tool_calls(tool_calls: List[Dict[str, Any]]):
    """Utility to repair JSON in tool call arguments if they are malformed."""
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
        try:
            # Reading JSON data from the request body
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})

            data = json.loads(raw_body)
            # Check if the client requested streaming
            client_wants_stream = data.get("stream", False)
            
            # Force stream=False to ensure we can parse and repair the response
            # regardless of client settings.
            data["stream"] = False
            target_model = data.get("model")

            logging.info(f"Model request accepted: {target_model}")

            # Attempt to switch models
            if not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps(
                    {"error": f"Failed to activate model {target_model}"}
                )

            # Forwarding the request to the llama.cpp backend
            # Always use stream=False here because we forced
            # data["stream"] = False
            resp = requests.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json=data,
                stream=False,
                timeout=(10, 1800)
            )

            raw_resp_content = resp.content
            try:
                resp_data = resp.json()
                # If it's a chat completion, try to repair the content
                choices = resp_data.get("choices", [])
                if choices:
                    message = resp_data["choices"][0].get("message", {})

                    # 1. Ensure content is null if tool_calls present
                    #    (standard OpenAI)
                    tool_calls = message.get("tool_calls", [])
                    if tool_calls and not message.get("content"):
                        message["content"] = None

                    # 2. Repair tool_calls arguments if present
                    repair_tool_calls(tool_calls)
                
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
                        
                        if (
                            "choices" in chunk_data
                            and len(chunk_data["choices"]) > 0
                        ):
                            choice = chunk_data["choices"][0]
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
                log_trace(
                    raw_body,
                    raw_resp_content,
                    f"ERROR: {e}\nORIGINAL CONTENT: "
                    f"{raw_resp_content.decode('utf-8', errors='replace')}",
                )
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


class EmbeddingsProxy:
    """Proxy handler for processing embedding requests."""

    def POST(self):
        try:
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})

            data = json.loads(raw_body)
            target_model = data.get("model")

            if not target_model:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "Model is required"})

            # Attempt to switch models
            if not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps(
                    {"error": f"Failed to activate model {target_model}"}
                )

            # Forwarding the request to the llama.cpp backend
            resp = requests.post(
                f"{LLAMA_URL}/v1/embeddings",
                json=data,
                timeout=(10, 600)
            )

            if resp.status_code != 200:
                web.ctx.status = f"{resp.status_code} {resp.reason}"
                return resp.content

            # Parse and ensure 100% OpenAI compliance
            try:
                resp_data = resp.json()
                if "data" in resp_data:
                    for i, item in enumerate(resp_data["data"]):
                        if "object" not in item:
                            item["object"] = "embedding"
                        if "index" not in item:
                            item["index"] = i
                
                # Ensure object is "list"
                if "object" not in resp_data:
                    resp_data["object"] = "list"
                
                # Ensure model name matches requested one
                resp_data["model"] = target_model

                web.header('Content-Type', 'application/json')
                return json.dumps(resp_data)
            except Exception:
                # Fallback to raw content if parsing fails
                web.header('Content-Type', 'application/json')
                return resp.content

        except json.JSONDecodeError:
            web.ctx.status = "400 Bad Request"
            return json.dumps({"error": "Invalid JSON"})
        except Exception as e:
            logging.error(f"Error in EmbeddingsProxy: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ConversationsHandler:
    """Handler for managing conversation creation."""
    
    def POST(self):
        try:
            raw_body = web.data()
            data = json.loads(raw_body) if raw_body else {}
            metadata = data.get("metadata", {})
            items = data.get("items", [])
            
            conv_id = db.create_conversation(metadata=metadata)
            
            # Bootstrap items if provided
            for item in items:
                if item.get("type") == "message":
                    # Extract text content (can be string or list of objects in OpenAI spec)
                    content = item.get("content")
                    db.add_item(
                        conv_id,
                        None,
                        "message",
                        item.get("role", "user"),
                        content
                    )
            
            conv_obj = db.get_conversation(conv_id)
            web.header('Content-Type', 'application/json')
            return json.dumps(conv_obj)
        except Exception as e:
            logging.error(f"Error in ConversationsHandler: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ConversationsDetailHandler:
    """Handler for retrieving, updating or deleting a specific conversation."""
    
    def GET(self, conversation_id):
        try:
            conv_obj = db.get_conversation(conversation_id)
            if not conv_obj:
                web.ctx.status = "404 Not Found"
                return json.dumps({"error": "Conversation not found"})
            
            web.header('Content-Type', 'application/json')
            return json.dumps(conv_obj)
        except Exception as e:
            logging.error(f"Error in ConversationsDetailHandler (GET): {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})

    def POST(self, conversation_id):
        """Update conversation metadata."""
        try:
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})
            
            data = json.loads(raw_body)
            metadata = data.get("metadata")
            
            if metadata is None:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "metadata is required for update"})
            
            if db.update_conversation(conversation_id, metadata):
                conv_obj = db.get_conversation(conversation_id)
                web.header('Content-Type', 'application/json')
                return json.dumps(conv_obj)
            else:
                web.ctx.status = "404 Not Found"
                return json.dumps({"error": "Conversation not found"})
        except Exception as e:
            logging.error(f"Error in ConversationsDetailHandler (POST/Update): {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})

    def DELETE(self, conversation_id):
        try:
            if db.delete_conversation(conversation_id):
                web.header('Content-Type', 'application/json')
                return json.dumps({
                    "id": conversation_id,
                    "object": "conversation.deleted",
                    "deleted": True
                })
            else:
                web.ctx.status = "404 Not Found"
                return json.dumps({"error": "Conversation not found"})
        except Exception as e:
            logging.error(f"Error in ConversationsDetailHandler (DELETE): {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesHandler:
    """Proxy handler for the stateful OpenAI Responses API."""
    
    def POST(self):
        try:
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})

            data = json.loads(raw_body)
            # Check if the client requested streaming
            client_wants_stream = data.get("stream", False)
            
            target_model = data.get("model")
            conversation_id = data.get("conversation") or data.get("conversation_id")
            previous_response_id = data.get("previous_response_id")
            input_items = data.get("input", [])
            instructions = data.get("instructions")
            metadata = data.get("metadata", {})

            if not target_model:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "Model is required"})

            # Ensure model is switched (VRAM management)
            if not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps(
                    {"error": f"Failed to activate model {target_model}"}
                )

            # 1. State Management: Create or retrieve conversation
            if not conversation_id:
                conversation_id = db.create_conversation()
            
            # 2. Add input items to SQLite and prepare for backend
            current_messages = []
            for item in input_items:
                item_type = item.get("type")
                if item_type == "input_text":
                    text_content = item.get("text")
                    db.add_item(
                        conversation_id, 
                        None, 
                        "message", 
                        "user", 
                        {"text": text_content}
                    )
                    current_messages.append({"role": "user", "content": text_content})
                elif item_type == "function_call_output":
                    call_id = item.get("call_id")
                    output = item.get("output")
                    db.add_item(
                        conversation_id,
                        None,
                        "tool",
                        "tool",
                        {
                            "tool_call_id": call_id,
                            "content": output
                        }
                    )
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output
                    })
                elif item_type == "compaction":
                    # Handle compaction items by injecting their content into the history
                    content = item.get("encrypted_content", "")
                    try:
                        summary = base64.b64decode(content).decode('utf-8')
                    except Exception:
                        summary = content
                    
                    current_messages.append({
                        "role": "system", 
                        "content": f"Previous conversation summary: {summary}"
                    })

            # 3. Reconstruct history for the backend
            history = db.get_conversation_history(
                conversation_id, 
                up_to_response_id=previous_response_id
            )
            
            if previous_response_id:
                history.extend(current_messages)
            
            # Add instructions as system message if present
            if instructions:
                history.insert(0, {"role": "system", "content": instructions})

            # 4. Prepare backend request (Chat Completions)
            backend_tools = []
            for tool in data.get("tools", []):
                if tool.get("type") == "function":
                    backend_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {})
                        }
                    })

            backend_payload = {
                "model": target_model,
                "messages": history,
                "stream": False
            }
            if backend_tools:
                backend_payload["tools"] = backend_tools

            # 5. Call backend
            resp = requests.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json=backend_payload,
                timeout=(10, 1800)
            )
            
            if resp.status_code != 200:
                web.ctx.status = f"{resp.status_code} {resp.reason}"
                return resp.content

            resp_data = resp.json()
            
            # 6. Create Response object and store output items
            response_id = db.create_response(
                conversation_id, 
                target_model, 
                instructions=instructions,
                metadata=metadata
            )
            
            if "choices" in resp_data and len(resp_data["choices"]) > 0:
                message = resp_data["choices"][0].get("message", {})
                
                # Handle text content
                assistant_content = message.get("content")
                if assistant_content:
                    db.add_item(
                        conversation_id, 
                        response_id, 
                        "message", 
                        "assistant", 
                        {"text": assistant_content}
                    )
                
                # Handle tool calls
                tool_calls = message.get("tool_calls", [])
                repair_tool_calls(tool_calls)
                for tc in tool_calls:
                    func = tc.get("function", {})
                    db.add_item(
                        conversation_id,
                        response_id,
                        "function_call",
                        "assistant",
                        {
                            "call_id": tc.get("id"),
                            "name": func.get("name"),
                            "arguments": func.get("arguments")
                        }
                    )

            # 7. Finalize response object in DB
            usage = resp_data.get("usage", {})
            db.update_response(
                response_id, 
                "completed", 
                usage=usage,
                metadata=metadata
            )

            # 8. Return Response object
            final_response = db.get_response(response_id)

            if client_wants_stream:
                web.header('Content-Type', 'text/event-stream')
                web.header('Cache-Control', 'no-cache')
                web.header('Connection', 'keep-alive')
                web.header('X-Accel-Buffering', 'no')

                def response_stream(resp_obj):
                    def sse(event, data_obj):
                        return f"event: {event}\ndata: {json.dumps(data_obj)}\n\n".encode('utf-8')

                    yield sse("response.created", resp_obj)
                    for item in resp_obj.get("output", []):
                        yield sse("response.output_item.added", {"item": item})
                        yield sse("response.output_item.done", {"item": item})
                    yield sse("response.completed", resp_obj)
                    yield b"data: [DONE]\n\n"

                return response_stream(final_response)
            else:
                web.header('Content-Type', 'application/json')
                return json.dumps(final_response)

        except Exception as e:
            logging.error(f"Error in ResponsesHandler: {e}", exc_info=True)
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesDetailHandler:
    """Handler for retrieving or deleting a specific response."""
    
    def GET(self, response_id):
        try:
            response_obj = db.get_response(response_id)
            if not response_obj:
                web.ctx.status = "404 Not Found"
                return json.dumps({"error": "Response not found"})
            
            web.header('Content-Type', 'application/json')
            return json.dumps(response_obj)
        except Exception as e:
            logging.error(f"Error in ResponsesDetailHandler (GET): {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})

    def DELETE(self, response_id):
        try:
            success = db.delete_response(response_id)
            if not success:
                web.ctx.status = "404 Not Found"
                return json.dumps({"error": "Response not found"})
            
            web.header('Content-Type', 'application/json')
            return json.dumps({"id": response_id, "object": "response", "deleted": True})
        except Exception as e:
            logging.error(f"Error in ResponsesDetailHandler (DELETE): {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesCompactHandler:
    """Handler for compacting a conversation."""
    
    def POST(self):
        try:
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})

            data = json.loads(raw_body)
            target_model = data.get("model")
            input_items = data.get("input", [])
            
            if not target_model:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "model is required"})

            if not ChatProxy.switch_model(target_model):
                web.ctx.status = "500 Internal Server Error"
                return json.dumps({"error": f"Failed to activate model {target_model}"})

            full_text = ""
            for item in input_items:
                if isinstance(item, str):
                    full_text += item + "\n"
                elif isinstance(item, dict):
                    if item.get("type") == "input_text":
                        full_text += f"User: {item.get('text')}\n"
                    elif item.get("role") == "assistant":
                        content = item.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if c.get("type") == "output_text":
                                    full_text += f"Assistant: {c.get('text')}\n"
                        elif isinstance(content, str):
                            full_text += f"Assistant: {content}\n"

            summary_prompt = f"Summarize the following conversation concisely for future context:\n\n{full_text}"
            backend_payload = {
                "model": target_model,
                "messages": [{"role": "user", "content": summary_prompt}],
                "stream": False
            }

            resp = requests.post(f"{LLAMA_URL}/v1/chat/completions", json=backend_payload, timeout=(10, 600))
            if resp.status_code != 200:
                web.ctx.status = f"{resp.status_code} {resp.reason}"
                return resp.content

            summary = resp.json()["choices"][0]["message"]["content"]
            encoded_summary = base64.b64encode(summary.encode('utf-8')).decode('utf-8')

            result = {
                "id": f"resp_{uuid.uuid4().hex[:12]}",
                "object": "response",
                "created_at": int(time.time()),
                "model": target_model,
                "status": "completed",
                "output": [
                    {
                        "id": f"cmp_{uuid.uuid4().hex[:12]}",
                        "type": "compaction",
                        "encrypted_content": encoded_summary
                    }
                ],
                "usage": resp.json().get("usage", {})
            }
            
            web.header('Content-Type', 'application/json')
            return json.dumps(result)
        except Exception as e:
            logging.error(f"Error in ResponsesCompactHandler: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesItemsHandler:
    """Handler for listing output items of a specific response."""
    
    def GET(self, response_id):
        try:
            items = db.get_response_items(response_id)
            web.header('Content-Type', 'application/json')
            return json.dumps({"object": "list", "data": items, "has_more": False})
        except Exception as e:
            logging.error(f"Error in ResponsesItemsHandler: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesInputItemsHandler:
    """Handler for listing input items of a specific response."""
    
    def GET(self, response_id):
        try:
            response_obj = db.get_response(response_id)
            if not response_obj:
                web.ctx.status = "404 Not Found"
                return json.dumps({"error": "Response not found"})
            
            conv_id = response_obj["conversation_id"]
            history = db.get_conversation_history(conv_id, up_to_response_id=response_id)
            
            items = []
            for i, msg in enumerate(history):
                items.append({
                    "id": f"input_item_{response_id}_{i}",
                    "object": "item",
                    "type": "message",
                    "role": msg["role"],
                    "content": [{"type": "output_text", "text": msg["content"]}]
                })

            web.header('Content-Type', 'application/json')
            return json.dumps({"object": "list", "data": items, "has_more": False})
        except Exception as e:
            logging.error(f"Error in ResponsesInputItemsHandler: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesInputTokensHandler:
    """Handler for counting input tokens."""
    
    def POST(self):
        try:
            raw_body = web.data()
            if not raw_body:
                web.ctx.status = "400 Bad Request"
                return json.dumps({"error": "No data provided"})

            data = json.loads(raw_body)
            input_items = data.get("input", [])
            total_chars = 0
            for item in input_items:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    total_chars += len(item.get("text", ""))
            
            result = {
                "object": "response.input_tokens",
                "input_tokens": total_chars // 4
            }
            
            web.header('Content-Type', 'application/json')
            return json.dumps(result)
        except Exception as e:
            logging.error(f"Error in ResponsesInputTokensHandler: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


class ResponsesCancelHandler:
    """Handler for canceling a specific response."""
    
    def POST(self, response_id):
        try:
            response_obj = db.get_response(response_id)
            if not response_obj:
                web.ctx.status = "404 Not Found"
                return json.dumps({"error": "Response not found"})
            
            web.header('Content-Type', 'application/json')
            return json.dumps(response_obj)
        except Exception as e:
            logging.error(f"Error in ResponsesCancelHandler: {e}")
            web.ctx.status = "500 Internal Server Error"
            return json.dumps({"error": str(e)})


app = web.application(urls, globals())

if __name__ == "__main__":
    server_port = CONFIG['server']['port']
    server_host = CONFIG['server']['host']
    sys.argv.append(f'{server_host}:{server_port}')

    logging.info(f"The proxy server runs on http://{server_host}:{server_port}")
    logging.info(f"Available models: {', '.join(MODELS.keys())}")
    app.run()
