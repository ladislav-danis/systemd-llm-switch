import unittest
from unittest.mock import patch, MagicMock
import json
import sys
from types import ModuleType

# --- MOCKING web.py BEFORE IMPORTING MAIN ---
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
                'qwen3-coder-next': 'qwen3-coder-next.service',
                'qwen3.5-thinking': 'qwen3.5-thinking.service',
                'bge-m3': 'bge-m3.service'
            }
        }
        main.MODELS = main.CONFIG['models']
        main.LLAMA_URL = main.CONFIG['server']['llama_url']
        """Reset status before each test."""
        main.BaseModelProxy._current_active_model = None
        main.web.ctx.status = "200 OK"
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-next",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False
        })

    def _get_result_content(self, result):
        if hasattr(result, '__iter__') and not isinstance(result, (bytes, str)):
            return b"".join(list(result))
        return result

    @patch('main.requests.get')
    @patch('main.requests.post')
    @patch('main.subprocess.run')
    def test_switch_to_coder_and_chat(self, mock_run, mock_post, mock_get):
        """Test switching to the Coder model and obtaining a JSON response."""
        def run_side_effect(command, **kwargs):
            cmd_str = " ".join(command)
            res = MagicMock(returncode=0, stdout="inactive", stderr="")
            if "is-active" in cmd_str and "qwen3-coder-next.service" not in cmd_str:
                res.stdout = "active"
            return res
        
        mock_run.side_effect = run_side_effect
        mock_get.return_value = MagicMock(status_code=200)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response_data = {"choices": [{"message": {"content": "Python code"}}]}
        mock_response.json.return_value = mock_response_data
        mock_response.content = json.dumps(mock_response_data).encode('utf-8')
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()
        
        if isinstance(result, bytes):
            self.assertIn(b"Python code", result)
        else:
            self.assertIn("Python code", result)

        calls = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(any("stop" in c for c in calls))
        self.assertTrue(any("start" in c and "qwen3-coder-next.service" in c for c in calls))

    @patch('main.requests.get')
    @patch('main.requests.post')
    @patch('main.subprocess.run')
    def test_switch_to_thinking_model(self, mock_run, mock_post, mock_get):
        main.web._test_data = json.dumps({
            "model": "qwen3.5-thinking",
            "stream": False,
            "messages": []
        })
        mock_run.return_value = MagicMock(stdout="inactive", returncode=0)
        mock_get.return_value = MagicMock(status_code=200)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response_data = {"choices": [{"message": {"content": "Thinking..."}}]}
        mock_response.json.return_value = mock_response_data
        mock_response.content = json.dumps(mock_response_data).encode('utf-8')
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        proxy.POST()

        calls = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(any("start" in c and "qwen3.5-thinking.service" in c for c in calls))

    @patch('main.requests.get')
    @patch('main.requests.post')
    @patch('main.subprocess.run')
    def test_streaming_disabled_always(self, mock_run, mock_post, mock_get):
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-next",
            "stream": True,
            "messages": []
        })
        def run_side_effect(command, **kwargs):
            cmd_str = " ".join(command)
            if "is-active" in cmd_str and "qwen3-coder-next.service" in cmd_str:
                return MagicMock(stdout="active", returncode=0)
            return MagicMock(stdout="inactive", returncode=0)
        mock_run.side_effect = run_side_effect
        mock_get.return_value = MagicMock(status_code=200)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response_data = {"choices": [{"message": {"content": "Normal", "role": "assistant"}}]}
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()

        import types
        self.assertIsInstance(result, types.GeneratorType)
        chunks = list(result)
        self.assertTrue(chunks[0].startswith(b'data: {'))
        self.assertEqual(chunks[1], b'data: [DONE]\n\n')

    @patch('main.subprocess.run')
    def test_invalid_model_error(self, mock_run):
        main.web._test_data = json.dumps({"model": "unknown-model"})
        proxy = main.ChatProxy()
        result = proxy.POST()
        self.assertIn("Failed to activate model", result)
        self.assertEqual(main.web.ctx.status, "500 Internal Server Error")

    def test_missing_model_error(self):
        main.web._test_data = json.dumps({"messages": [{"role": "user", "content": "Hi"}]})
        proxy = main.ChatProxy()
        result = proxy.POST()
        self.assertIn("No model specified in the request", result)
        self.assertEqual(main.web.ctx.status, "400 Bad Request")

    def test_payload_too_large(self):
        large_data = "a" * (11 * 1024 * 1024)
        main.web._test_data = large_data
        proxy = main.ChatProxy()
        result = proxy.POST()
        self.assertIn("Payload exceeds 10MB limit", result)
        self.assertEqual(main.web.ctx.status, "413 Payload Too Large")

    @patch('main.requests.get')
    @patch('main.requests.post')
    @patch('main.subprocess.run')
    def test_rollback_on_failure(self, mock_run, mock_post, mock_get):
        main.BaseModelProxy._current_active_model = "qwen3-coder-next"
        main.web._test_data = json.dumps({"model": "qwen3.5-thinking", "messages": []})

        def side_effect(command, **kwargs):
            cmd_str = " ".join(command)
            if "is-active" in cmd_str:
                # Coder is active for rollback success, thinking is inactive for start
                if "qwen3-coder-next.service" in cmd_str:
                    return MagicMock(stdout="active", returncode=0)
                return MagicMock(stdout="inactive", returncode=0)
            if "start" in cmd_str and "qwen3.5-thinking.service" in cmd_str:
                return MagicMock(returncode=1, stderr="Failed to start")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect
        mock_get.return_value = MagicMock(status_code=200)

        proxy = main.ChatProxy()
        result = proxy.POST()
        self.assertIn("Failed to activate model", result)
        self.assertEqual(main.BaseModelProxy._current_active_model, "qwen3-coder-next")

    @patch('main.requests.get')
    @patch('main.requests.post')
    @patch('main.subprocess.run')
    def test_tool_call_content_null_and_repair(self, mock_run, mock_post, mock_get):
        main.web._test_data = json.dumps({
            "model": "qwen3-coder-next",
            "messages": [{"role": "user", "content": "Get weather"}]
        })
        def run_side_effect(command, **kwargs):
            cmd_str = " ".join(command)
            if "is-active" in cmd_str and "qwen3-coder-next.service" in cmd_str:
                return MagicMock(stdout="inactive", returncode=0)
            return MagicMock(stdout="inactive", returncode=0)
            
        mock_run.side_effect = run_side_effect
        mock_get.return_value = MagicMock(status_code=200)
        mock_response_data = {
            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Prague", }'}}]}}]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.content = json.dumps(mock_response_data).encode()
        mock_post.return_value = mock_response

        proxy = main.ChatProxy()
        result = proxy.POST()
        parsed_result = json.loads(result)
        message = parsed_result["choices"][0]["message"]
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"location": "Prague"}')

    @patch('main.requests.get')
    @patch('main.requests.post')
    @patch('main.subprocess.run')
    def test_trace_logging(self, mock_run, mock_post, mock_get):
        from pathlib import Path
        test_log = Path("test_trace_main.log")
        if test_log.exists(): test_log.unlink()
        main.TRACE_LOG_PATH = test_log
        try:
            main.web._test_data = b'{"model": "qwen3-coder-next"}'
            def run_side_effect(command, **kwargs):
                cmd_str = " ".join(command)
                if "is-active" in cmd_str and "qwen3-coder-next.service" in cmd_str:
                    return MagicMock(stdout="active", returncode=0)
                return MagicMock(stdout="inactive", returncode=0)
            mock_run.side_effect = run_side_effect
            mock_get.return_value = MagicMock(status_code=200)
            mock_response_data = {"choices": [{"message": {"content": "ok"}}]}
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.content = json.dumps(mock_response_data).encode()
            mock_post.return_value = mock_response

            proxy = main.ChatProxy()
            proxy.POST()
            self.assertTrue(test_log.exists())
        finally:
            if test_log.exists(): test_log.unlink()
            main.TRACE_LOG_PATH = None

if __name__ == '__main__':
    unittest.main()
