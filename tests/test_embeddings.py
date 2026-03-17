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
                'qwen3-coder-next': 'qwen3-coder-next.service'
            }
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']
        main.BaseModelProxy._current_active_model = None
        main.web.ctx.status = "200 OK"

    @patch('main.requests.get')
    @patch('main.requests.post')
    @patch('main.subprocess.run')
    def test_embeddings_switch_and_post(self, mock_run, mock_post, mock_get):
        """Test switching to an embeddings model and getting a response."""
        def run_side_effect(command, **kwargs):
            cmd_str = " ".join(command)
            res = MagicMock(returncode=0, stdout="inactive", stderr="")
            if "is-failed" in cmd_str:
                res.returncode = 1
            return res
            
        mock_run.side_effect = run_side_effect
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
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
            ],
            "model": "bge-m3"
        }
        mock_response.json.return_value = mock_response_data
        mock_response.content = json.dumps(mock_response_data).encode('utf-8')
        mock_post.return_value = mock_response

        proxy = main.EmbeddingsProxy()
        result = proxy.POST()

        if isinstance(result, bytes):
            self.assertIn(b"embedding", result)
        else:
            self.assertIn("embedding", result)

        self.assertEqual(main.BaseModelProxy._current_active_model, "bge-m3")

    @patch('main.subprocess.run')
    def test_embeddings_invalid_model(self, mock_run):
        main.web._test_data = json.dumps({"model": "non-existent"})
        proxy = main.EmbeddingsProxy()
        result = proxy.POST()
        self.assertIn("Failed to activate model", result)
        self.assertEqual(main.web.ctx.status, "500 Internal Server Error")


if __name__ == '__main__':
    unittest.main()
