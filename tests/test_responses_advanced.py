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
from systemd_llm_switch.db import Database

class TestResponsesAdvanced(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_llm_switch_adv.db"
        self.db = Database(self.db_path)
        main.db = self.db
        main.CONFIG = {
            'server': {'llama_url': 'http://localhost:3004'},
            'models': {'qwen3.5-thinking': 'qwen3.5-thinking.service'}
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']

    def tearDown(self):
        db_file = Path(main.__file__).parent / self.db_path
        if db_file.exists():
            db_file.unlink()

    @patch('requests.post')
    @patch('systemd_llm_switch.main.ChatProxy.switch_model')
    def test_multimodal_and_metadata(self, mock_switch, mock_post):
        mock_switch.return_value = True
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "I see a cat."}}],
            "usage": {"total_tokens": 15}
        }
        mock_post.return_value = mock_resp

        # 1. Test input_image and instructions
        main.web._test_data = json.dumps({
            "model": "qwen3.5-thinking",
            "instructions": "Be very concise.",
            "input": [
                {"type": "input_text", "text": "What is in this image?"},
                {"type": "input_image", "image": "base64_data_here"}
            ],
            "metadata": {"user_id": "123"}
        })

        handler = main.ResponsesHandler()
        result_json = handler.POST()
        response = json.loads(result_json)

        self.assertEqual(response["metadata"]["user_id"], "123")
        self.assertEqual(response["instructions"], "Be very concise.")
        
        # Verify instructions were added to history
        args, kwargs = mock_post.call_args
        history = kwargs["json"]["messages"]
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[0]["content"], "Be very concise.")

    @patch('requests.post')
    @patch('systemd_llm_switch.main.ChatProxy.switch_model')
    def test_tool_output_and_branching(self, mock_switch, mock_post):
        mock_switch.return_value = True
        
        # First turn: model calls a tool
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]
                }
            }],
            "usage": {}
        }
        mock_post.return_value = mock_resp1

        main.web._test_data = json.dumps({
            "model": "qwen3.5-thinking",
            "input": [{"type": "input_text", "text": "What time is it?"}]
        })
        handler = main.ResponsesHandler()
        resp1 = json.loads(handler.POST())
        conv_id = resp1["conversation_id"]
        resp1_id = resp1["id"]

        # Second turn: provide tool output
        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "It is 12:00."}}],
            "usage": {}
        }
        mock_post.return_value = mock_resp2

        main.web._test_data = json.dumps({
            "model": "qwen3.5-thinking",
            "conversation_id": conv_id,
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "12:00"}]
        })
        resp2 = json.loads(handler.POST())
        self.assertIn("12:00", resp2["output"][0]["content"][0]["text"])

        # Verify history reconstruction for tool output
        args, kwargs = mock_post.call_args
        history = kwargs["json"]["messages"]
        self.assertEqual(history[-1]["role"], "tool")
        self.assertEqual(history[-1]["tool_call_id"], "call_1")

        # Third turn: Branching with previous_response_id
        # We branch from resp1_id (the tool call) and ask something else
        main.web._test_data = json.dumps({
            "model": "qwen3.5-thinking",
            "conversation_id": conv_id,
            "previous_response_id": resp1_id,
            "input": [{"type": "input_text", "text": "Actually, ignore the time. What is 2+2?"}]
        })
        handler.POST()
        
        args, kwargs = mock_post.call_args
        history = kwargs["json"]["messages"]
        # History should NOT contain the tool output from resp2
        roles = [m["role"] for m in history]
        self.assertEqual(roles.count("tool"), 0)
        self.assertEqual(history[-1]["content"], "Actually, ignore the time. What is 2+2?")

if __name__ == '__main__':
    unittest.main()
