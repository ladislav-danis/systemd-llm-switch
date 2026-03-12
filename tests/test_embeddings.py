import unittest
from unittest.mock import patch, MagicMock
import json
import sys
from types import ModuleType
from pathlib import Path

# --- MOCKING web.py BEFORE IMPORTING MAIN ---
mock_web = ModuleType('web')
mock_web.ctx = MagicMock()
mock_web.header = MagicMock()
mock_web.application = MagicMock()
mock_web._test_data = ""
mock_web.data = lambda: mock_web._test_data
sys.modules['web'] = mock_web

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from systemd_llm_switch import main

class TestEmbeddingsAPI(unittest.TestCase):
    def setUp(self):
        main.CONFIG = {
            'server': {'llama_url': 'http://localhost:3004'},
            'models': {'bge-m3': 'bge-m3.service'}
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']

    @patch('requests.post')
    @patch('systemd_llm_switch.main.ChatProxy.switch_model')
    def test_embeddings_proxy(self, mock_switch, mock_post):
        """Test the /v1/embeddings proxy, ensuring it switches to the correct model and forwards the response."""
        mock_switch.return_value = True
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        resp_data = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "bge-m3",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        mock_resp.json.return_value = resp_data
        mock_resp.content = json.dumps(resp_data).encode('utf-8')
        mock_post.return_value = mock_resp

        # Mock web.data()
        main.web._test_data = json.dumps({
            "model": "bge-m3",
            "input": "The food was delicious and the service was excellent."
        })

        handler = main.EmbeddingsProxy()
        result = handler.POST()
        
        # Verify result
        data = json.loads(result)
        self.assertEqual(data["model"], "bge-m3")
        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["embedding"], [0.1, 0.2])
        self.assertEqual(data["data"][0]["object"], "embedding")
        self.assertEqual(data["data"][0]["index"], 0)
        
        # Verify switch_model was called
        mock_switch.assert_called_with("bge-m3")

if __name__ == '__main__':
    unittest.main()
