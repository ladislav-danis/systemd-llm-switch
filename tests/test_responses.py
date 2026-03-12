import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os
from types import ModuleType
from pathlib import Path

# --- MOCKING web.py BEFORE IMPORTING MAIN ---
mock_web = ModuleType('web')
mock_web.ctx = MagicMock()
mock_web.header = MagicMock()
mock_web.application = MagicMock()
# Global storage for web.data() simulation
mock_web._test_data = ""
mock_web.data = lambda: mock_web._test_data
sys.modules['web'] = mock_web

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from systemd_llm_switch import main
from systemd_llm_switch.db import Database

class TestResponsesAPI(unittest.TestCase):
    def setUp(self):
        # Use a temporary database for testing
        self.db_path = "test_llm_switch.db"
        self.db = Database(self.db_path)
        main.db = self.db
        main.CONFIG = {
            'server': {'llama_url': 'http://localhost:3004'},
            'models': {'qwen3-coder-next': 'qwen3-coder-next.service'}
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']

    def tearDown(self):
        # Clean up temporary database
        db_file = Path(main.__file__).parent / self.db_path
        if db_file.exists():
            db_file.unlink()

    def test_database_persistence(self):
        """Test that conversation items can be correctly stored and retrieved from the database."""
        conv_id = self.db.create_conversation()
        self.assertTrue(conv_id.startswith("conv_"))
        
        self.db.add_item(conv_id, None, "message", "user", {"text": "Hello"})
        history = self.db.get_conversation_history(conv_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[0]['content'], 'Hello')

    def test_responses_endpoint_routing(self):
        """Test that the /v1/responses endpoint is properly routed."""
        # Verify our handler is in the urls list
        self.assertIn('/v1/responses', main.urls)
        self.assertIn('ResponsesHandler', main.urls)

    @patch('requests.post')
    @patch('systemd_llm_switch.main.ChatProxy.switch_model')
    def test_streaming_logic_responses(self, mock_switch, mock_post):
        """Test that the streaming mode correctly produces OpenAI Responses API compatible SSE events."""
        mock_switch.return_value = True
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10}
        }
        mock_post.return_value = mock_resp

        # Mock web.data() for the handler
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-next",
            "input": [{"type": "input_text", "text": "Hi"}],
            "stream": True
        })

        # Instantiate handler and call POST
        handler = main.ResponsesHandler()
        result_gen = handler.POST()

        # The result should be a generator
        self.assertTrue(hasattr(result_gen, '__iter__'))
        
        # Collect SSE events
        events = [item.decode('utf-8') if isinstance(item, bytes) else item for item in result_gen]
        content = "".join(events)
        
        self.assertIn('event: response.created', content)
        self.assertIn('event: response.output_item.added', content)
        self.assertIn('event: response.content_part.added', content)
        self.assertIn('event: response.content_part.delta', content)
        self.assertIn('event: response.content_part.done', content)
        self.assertIn('event: response.output_item.done', content)
        self.assertIn('event: response.completed', content)
        self.assertIn('event: response.done', content)
        self.assertIn('data: [DONE]', content)

if __name__ == '__main__':
    unittest.main()
