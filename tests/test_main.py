import unittest
from unittest.mock import patch, MagicMock
import json
import sys
from types import ModuleType

# --- MOCKING web.py BEFORE IMPORTING MAIN ---
# We need to create a fake web module because
# it may not be installed in the test environment.
mock_web = ModuleType('web')
mock_web.ctx = MagicMock()
mock_web.header = MagicMock()
# Global storage for web.data() simulation
mock_web._test_data = ""
mock_web.data = lambda: mock_web._test_data
sys.modules['web'] = mock_web
sys.path.append('src/systemd_llm_switch')
import main # noqa


class TestModelProxy(unittest.TestCase):
    """Comprehensive test suite for Llama Service Proxy."""

    def setUp(self):
        """Reset status before each test and inject configuration."""
        main.CONFIG = {
            'server': {
                'host': '0.0.0.0',
                'port': 3002,
                'llama_url': 'http://localhost:3004'
            },
            'models': {
                'qwen3-coder-flash': 'qwen3-coder-flash.service',
                'qwen3-coder-next': 'qwen3-coder-next.service',
                'qwen3-thinking': 'qwen3-thinking.service',
                'bge-m3': 'bge-m3.service'
            }
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']
        """Reset status before each test."""
        main.ChatProxy._current_active_model = None
        main.web.ctx.status = "200 OK"
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False
        })

    def _get_result_content(self, result):
        """Auxiliary function for generator exhaustion or text return."""
        if hasattr(result, '__iter__') and not isinstance(
            result,
            (bytes, str)
        ):
            return b"".join(list(result))
        return result

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    @patch('main.requests.get')
    def test_switch_to_coder_and_chat(self, mock_get, mock_post, mock_run):
        """Test switching to the Coder model and obtaining a JSON response."""
        mock_run.return_value = MagicMock(stdout="inactive")
        mock_get.return_value = MagicMock(status_code=200)

        mock_response = MagicMock()
        mock_response.status_code = 200  # noqa: E501
        mock_response_data = {
            "choices": [{"message": {"content": "Python code"}}]
        }
        mock_response.json.return_value = mock_response_data
        mock_response.content = json.dumps(mock_response_data).encode('utf-8')
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()
        # content is a string if it's JSON-encoded result from json.dumps
        # or bytes if it's resp.content
        if isinstance(result, bytes):
            self.assertIn(b"Python code", result)  # noqa: E501
        else:
            self.assertIn("Python code", result)

        # Checking whether it stopped and started
        calls = [str(c) for c in mock_run.call_args_list]  # noqa: E501
        self.assertTrue(any("stop" in c for c in calls))
        self.assertTrue(
            any(
                "start" in c and "qwen3-coder-flash.service" in c
                for c in calls
            )
        )

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    @patch('main.requests.get')
    def test_switch_to_thinking_model(self, mock_get, mock_post, mock_run):
        """Test switching to the Thinking model and verification
        of the correct service."""
        main.web._test_data = json.dumps({
            "model": "qwen3-thinking",
            "stream": False,
            "messages": []
        })
        mock_run.return_value = MagicMock(stdout="inactive")
        mock_get.return_value = MagicMock(status_code=200)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response_data = {"thinking": "..."}
        mock_response.json.return_value = mock_response_data
        mock_response.content = json.dumps(mock_response_data).encode('utf-8')
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        proxy.POST()  # noqa: E501

        # Verification that the correct service
        # for the thinking model has been activated
        calls = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(
            any(
                "start" in c and "qwen3-thinking.service" in c
                for c in calls
            )
        )  # noqa: E501

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    def test_streaming_disabled_always(self, mock_post, mock_run):
        """Test that even when stream: True is requested,
        it is forced to False and returns a standard JSON response."""
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-flash",  # noqa: E501
            "stream": True,
            "messages": []
        })
        # We simulate that the model is already running
        mock_run.return_value = MagicMock(stdout="active")

        mock_response = MagicMock()  # noqa: E501
        mock_response.status_code = 200
        mock_response_data = {
            "choices": [{"message": {"content": "Normal response"}}]  # noqa: E501
        }
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()

        # It should NOT be a generator, but a JSON string
        # (since mock_web mocks it)
        self.assertIsInstance(result, str)
        self.assertIn("Normal response", result)

        # Verify that requests.post was called with stream=False
        args, kwargs = mock_post.call_args
        self.assertFalse(kwargs.get('stream'))

    @patch('main.subprocess.run')
    def test_invalid_model_error(self, mock_run):
        """Testing the handling of requests for non-existent models."""
        main.web._test_data = json.dumps({"model": "unknown-model"})

        proxy = main.ChatProxy()
        result = proxy.POST()

        self.assertIn("Failed to activate model", result)
        self.assertEqual(main.web.ctx.status, "500 Internal Server Error")

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    @patch('main.requests.get')
    def test_tool_call_content_null_and_repair(self, mock_get, mock_post, mock_run):
        """Test that content is set to null when tool_calls are present
        and that tool_calls arguments are repaired."""
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-flash",
            "stream": False,
            "messages": [{"role": "user", "content": "Get weather"}]
        })
        mock_run.return_value = MagicMock(stdout="active")
        mock_get.return_value = MagicMock(status_code=200)

        # Simulating response with tool_calls and malformed arguments
        mock_response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Prague", }' # Trailing comma
                        }
                    }]
                }
            }]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()

        parsed_result = json.loads(result)
        message = parsed_result["choices"][0]["message"]

        # 1. Check if content is null
        self.assertIsNone(message["content"])

        # 2. Check if arguments are repaired
        repaired_args = message["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(repaired_args, '{"location": "Prague"}')

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    @patch('main.requests.get')
    def test_no_json_repair_in_content(self, mock_get, mock_post, mock_run):
        """Test that main content is NOT repaired even if it looks like malformed JSON."""
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-flash",
            "messages": [{"role": "user", "content": "Give me example"}]
        })
        mock_run.return_value = MagicMock(stdout="active")
        mock_get.return_value = MagicMock(status_code=200)

        malformed_json_text = 'Here is bad json: {"key": "value", }'
        mock_response_data = {
            "choices": [{"message": {"content": malformed_json_text}}]
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()

        parsed_result = json.loads(result)
        # Content should remain exactly as it was
        self.assertEqual(parsed_result["choices"][0]["message"]["content"], malformed_json_text)

    def test_list_models_endpoint(self):
        """Test that the endpoint /v1/models returns all defined models."""
        list_models = main.ListModels()
        result = list_models.GET()
        data = json.loads(result)

        model_ids = [m["id"] for m in data["data"]]
        self.assertIn("qwen3-coder-flash", model_ids)
        self.assertIn("qwen3-coder-next", model_ids)
        self.assertIn("qwen3-thinking", model_ids)
        self.assertIn("bge-m3", model_ids)
        self.assertEqual(data["object"], "list")


if __name__ == '__main__':
    unittest.main()
