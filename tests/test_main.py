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
                'qwen3-coder-30-a3b-8gb': 'qwen3-coder.service',
                'qwen3-thinking-30-a3b-8gb': 'qwen3-thinking.service',
                'bge-m3': 'bge-embedding.service'
            }
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']
        """Reset status before each test."""
        main.ChatProxy._current_active_model = None
        main.web.ctx.status = "200 OK"
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-30-a3b-8gb",
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
        mock_response.status_code = 200
        mock_response.content = (
            b'{"choices": [{"message": {"content": "Python code"}}]}'
        )
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()
        content = self._get_result_content(result)

        self.assertIn(b"Python code", content)
        # Checking whether it stopped and started
        calls = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(any("stop" in c for c in calls))
        self.assertTrue(
            any("start" in c and "qwen3-coder.service" in c for c in calls)
        )

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    @patch('main.requests.get')
    def test_switch_to_thinking_model(self, mock_get, mock_post, mock_run):
        """Test switching to the Thinking model and verification
        of the correct service."""
        main.web._test_data = json.dumps({
            "model": "qwen3-thinking-30-a3b-8gb",
            "stream": False,
            "messages": []
        })
        mock_run.return_value = MagicMock(stdout="inactive")
        mock_get.return_value = MagicMock(status_code=200)

        mock_post.return_value = MagicMock(
            status_code=200, content=b'{"thinking": "..."}'
        )

        proxy = main.ChatProxy()
        proxy.POST()

        # Verification that the correct service
        # for the thinking model has been activated
        calls = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(
            any("start" in c and "qwen3-thinking.service" in c for c in calls)
        )

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    def test_streaming_enabled(self, mock_post, mock_run):
        """Test that when stream: True,
        a generator with the correct content is returned."""
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-30-a3b-8gb",
            "stream": True,
            "messages": []
        })
        # We simulate that the model is already running
        mock_run.return_value = MagicMock(stdout="active")

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [
            b'data: {"text": "A"}',
            b'data: {"text": "B"}'
        ]
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result_gen = proxy.POST()

        # We will exhaust the generator
        chunks = list(result_gen)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], b'data: {"text": "A"}')
        self.assertEqual(chunks[1], b'data: {"text": "B"}')

    @patch('main.subprocess.run')
    def test_invalid_model_error(self, mock_run):
        """Testing the handling of requests for non-existent models."""
        main.web._test_data = json.dumps({"model": "unknown-model"})

        proxy = main.ChatProxy()
        result = proxy.POST()

        self.assertIn("Failed to activate model", result)
        self.assertEqual(main.web.ctx.status, "500 Internal Server Error")

    def test_list_models_endpoint(self):
        """Test that the endpoint /v1/models returns all defined models."""
        list_models = main.ListModels()
        result = list_models.GET()
        data = json.loads(result)

        model_ids = [m["id"] for m in data["data"]]
        self.assertIn("qwen3-coder-30-a3b-8gb", model_ids)
        self.assertIn("qwen3-thinking-30-a3b-8gb", model_ids)
        self.assertEqual(data["object"], "list")


if __name__ == '__main__':
    unittest.main()
