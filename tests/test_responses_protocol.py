import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import uuid
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

class TestResponsesProtocol(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_llm_switch_protocol.db"
        self.db = Database(self.db_path)
        main.db = self.db
        main.CONFIG = {
            'server': {'llama_url': 'http://localhost:3004'},
            'models': {'qwen3-coder-next': 'qwen3-coder-next.service'}
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']

    def tearDown(self):
        db_file = Path(main.__file__).parent / self.db_path
        if db_file.exists():
            db_file.unlink()

    def parse_sse_events(self, result_gen):
        events = []
        current_event = None
        for chunk in result_gen:
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')
            
            lines = chunk.strip().split('\n')
            for line in lines:
                if line.startswith('event: '):
                    current_event = line[7:]
                elif line.startswith('data: '):
                    data_str = line[6:]
                    if data_str != '[DONE]':
                        try:
                            data_obj = json.loads(data_str)
                            events.append((current_event, data_obj))
                        except json.JSONDecodeError:
                            pass
        return events

    @patch('requests.post')
    @patch('systemd_llm_switch.main.ChatProxy.switch_model')
    def test_full_protocol_sequence_message(self, mock_switch, mock_post):
        mock_switch.return_value = True
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"total_tokens": 10, "input_tokens": 5, "output_tokens": 5}
        }
        mock_post.return_value = mock_resp

        main.web._test_data = json.dumps({
            "model": "qwen3-coder-next",
            "input": [{"type": "input_text", "text": "Hi"}],
            "stream": True
        })

        handler = main.ResponsesHandler()
        result_gen = handler.POST()
        events = self.parse_sse_events(result_gen)

        # Expected Sequence:
        # 1. response.created
        # 2. response.heartbeat (optional, depends on timing)
        # 3. response.output_item.added
        # 4. response.content_part.added
        # 5. response.content_part.delta
        # 6. response.content_part.done
        # 7. response.output_item.done
        # 8. response.completed
        # 9. response.done

        event_types = [e[0] for e in events]
        
        self.assertEqual(event_types[0], 'response.created')
        self.assertIn('response.output_item.added', event_types)
        self.assertIn('response.content_part.added', event_types)
        self.assertIn('response.content_part.delta', event_types)
        self.assertIn('response.content_part.done', event_types)
        self.assertIn('response.output_item.done', event_types)
        
        # Terminal events
        self.assertEqual(events[-2][0], 'response.completed')
        self.assertEqual(events[-1][0], 'response.done')

        # Check sequence numbers
        for i, (etype, data) in enumerate(events):
            self.assertEqual(data['sequence_number'], i + 1)
            self.assertEqual(data['type'], etype)

        # Check response.completed structure
        completed_data = events[-2][1]
        self.assertEqual(completed_data['response']['status'], 'completed')
        self.assertIsNone(completed_data['response']['error'])
        self.assertIsNotNone(completed_data['response']['usage'])

    @patch('requests.post')
    @patch('systemd_llm_switch.main.ChatProxy.switch_model')
    def test_protocol_sequence_tool_call(self, mock_switch, mock_post):
        mock_switch.return_value = True
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Prague\"}"}}]
                }
            }],
            "usage": {"total_tokens": 20}
        }
        mock_post.return_value = mock_resp

        main.web._test_data = json.dumps({
            "model": "qwen3-coder-next",
            "input": [{"type": "input_text", "text": "Weather in Prague?"}],
            "stream": True
        })

        handler = main.ResponsesHandler()
        result_gen = handler.POST()
        events = self.parse_sse_events(result_gen)

        event_types = [e[0] for e in events]
        
        # Verify function_call item sequence
        self.assertIn('response.output_item.added', event_types)
        
        # Find the tool call item added event
        tool_item_added = next(e[1] for e in events if e[0] == 'response.output_item.added')
        self.assertEqual(tool_item_added['item']['type'], 'function_call')
        
        # Find the content part delta for arguments
        tool_delta = next(e[1] for e in events if e[0] == 'response.content_part.delta')
        self.assertIn('arguments', tool_delta['delta'])
        self.assertIn('Prague', tool_delta['delta']['arguments'])

    @patch('requests.post')
    @patch('systemd_llm_switch.main.ChatProxy.switch_model')
    def test_error_protocol(self, mock_switch, mock_post):
        mock_switch.return_value = True
        
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        main.web._test_data = json.dumps({
            "model": "qwen3-coder-next",
            "input": [{"type": "input_text", "text": "Fail me"}],
            "stream": True
        })

        handler = main.ResponsesHandler()
        result_gen = handler.POST()
        events = self.parse_sse_events(result_gen)

        # Should still have terminal events
        self.assertEqual(events[-2][0], 'response.completed')
        self.assertEqual(events[-1][0], 'response.done')
        
        completed_data = events[-2][1]
        self.assertEqual(completed_data['response']['status'], 'failed')
        self.assertIsNotNone(completed_data['response']['error'])
        self.assertIn('Backend error', completed_data['response']['error']['message'])

if __name__ == '__main__':
    unittest.main()
