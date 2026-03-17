import unittest
from unittest.mock import patch, MagicMock
import json
import sys
from types import ModuleType

# --- MOCKING web.py ---
mock_web = ModuleType('web')
mock_web.ctx = MagicMock()
mock_web.header = MagicMock()
mock_web._test_data = ""
mock_web.data = lambda: mock_web._test_data
sys.modules['web'] = mock_web
sys.path.append('src/systemd_llm_switch')
import main  # noqa


class TestEmbeddingsProxy(unittest.TestCase):
    """Test suite for the Embeddings endpoint."""

    def setUp(self):
        """Reset status before each test."""
        main.CONFIG = {
            'server': {
                'host': '0.0.0.0',
                'port': 3002,
                'llama_url': 'http://localhost:3004'
            },
            'models': {
                'bge-m3': 'bge-m3.service',
                'qwen3-coder-flash': 'qwen3-coder-flash.service'
            }
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']
        main.BaseModelProxy._current_active_model = None
        main.web.ctx.status = "200 OK"

    @patch('main.subprocess.run')
    @patch('main.requests.post')
    @patch('main.requests.get')
    def test_embeddings_switch_and_post(self, mock_get, mock_post, mock_run):
        """Test switching to an embeddings model and getting a response."""
        mock_run.return_value = MagicMock(stdout="inactive", returncode=0)
        mock_get.return_value = MagicMock(status_code=200)

        main.web._test_data = json.dumps({
            "model": "bge-m3",
            "input": "The food was delicious and the service was excellent."
        })

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response_data = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [
                    0.1, 0.2, 0.3
                ]}
            ],
            "model": "bge-m3"
        }
        mock_response.content = json.dumps(mock_response_data).encode('utf-8')
        mock_post.return_value = mock_response

        proxy = main.EmbeddingsProxy()
        result = proxy.POST()

        if isinstance(result, bytes):
            self.assertIn(b"embedding", result)
            self.assertIn(b"0.1, 0.2, 0.3", result)
        else:
            self.assertIn("embedding", result)
            self.assertIn("0.1, 0.2, 0.3", result)

        # Verify model switching
        calls = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(
            any("start" in c and "bge-m3.service" in c for c in calls)
        )
        self.assertEqual(main.BaseModelProxy._current_active_model, "bge-m3")

    @patch('main.subprocess.run')
    def test_embeddings_invalid_model(self, mock_run):
        """Test requesting a model that doesn't exist."""
        main.web._test_data = json.dumps({"model": "non-existent"})

        proxy = main.EmbeddingsProxy()
        result = proxy.POST()

        self.assertIn("Failed to activate model", result)
        self.assertEqual(main.web.ctx.status, "500 Internal Server Error")


if __name__ == '__main__':
    unittest.main()
