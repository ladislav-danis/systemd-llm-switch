import unittest
import requests
import json
import os
import time

# Configuration for integration tests
PROXY_URL = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:3002")
TEST_MODEL = os.environ.get("LLM_TEST_MODEL", "qwen3-coder-next")
TEST_EMBEDDING_MODEL = os.environ.get("LLM_TEST_EMBEDDING_MODEL", "bge-m3")

def is_proxy_available():
    """Check if the proxy server is running."""
    try:
        response = requests.get(f"{PROXY_URL}/v1/models", timeout=2)
        if response.status_code == 200:
            # Check if our test models are in the list
            models = [m["id"] for m in response.json().get("data", [])]
            return TEST_MODEL in models or TEST_EMBEDDING_MODEL in models
    except:
        pass
    return False

@unittest.skipUnless(is_proxy_available(), f"Proxy not available at {PROXY_URL}")
class TestRealProxyIntegration(unittest.TestCase):
    """Real integration tests against a running proxy instance."""

    def setUp(self):
        self.url = f"{PROXY_URL}/v1/chat/completions"
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

    def test_single_tool_call(self):
        """Test a simple single tool call."""
        payload = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "What is the weather in Prague?"}],
            "tools": self.tools,
            "tool_choice": "auto",
            "stream": False
        }
        
        response = requests.post(self.url, json=payload, timeout=120)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        message = data["choices"][0]["message"]
        
        self.assertIn("tool_calls", message)
        self.assertTrue(len(message["tool_calls"]) >= 1)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertIsNone(message.get("content"), "Content should be null when tool_calls are present")

    def test_parallel_tool_calls(self):
        """Test multiple parallel tool calls with different functions."""
        payload = {
            "model": TEST_MODEL,
            "messages": [{
                "role": "user", 
                "content": "Give me the weather and current time in Prague. Use parallel calls for both functions."
            }],
            "tools": self.tools,
            "tool_choice": "auto",
            "stream": False,
            "parallel_tool_calls": True
        }
        
        response = requests.post(self.url, json=payload, timeout=120)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        message = data["choices"][0]["message"]
        
        self.assertIn("tool_calls", message)
        tool_names = [tc["function"]["name"] for tc in message["tool_calls"]]
        self.assertIn("get_weather", tool_names)
        self.assertIn("get_time", tool_names)
        self.assertIsNone(message.get("content"))

    def test_streaming_simulated_tool_call(self):
        """Test that streaming request also returns valid tool calls (simulated by proxy)."""
        payload = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": self.tools,
            "stream": True
        }
        
        response = requests.post(self.url, json=payload, timeout=120, stream=True)
        self.assertEqual(response.status_code, 200)
        
        tool_calls_detected = False
        for line in response.iter_lines():
            if not line: continue
            line_str = line.decode('utf-8')
            if line_str.startswith("data: ") and line_str != "data: [DONE]":
                data = json.loads(line_str[6:])
                delta = data["choices"][0].get("delta", {})
                if "tool_calls" in delta:
                    tool_calls_detected = True
                    self.assertEqual(delta["tool_calls"][0]["function"]["name"], "get_weather")
        
        self.assertTrue(tool_calls_detected, "No tool calls detected in stream")

    def test_embeddings(self):
        """Test the embeddings endpoint."""
        payload = {
            "model": TEST_EMBEDDING_MODEL,
            "input": "This is a test for embeddings."
        }
        
        response = requests.post(f"{PROXY_URL}/v1/embeddings", json=payload, timeout=120)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("embedding", data["data"][0])
        self.assertEqual(data["model"], TEST_EMBEDDING_MODEL)

if __name__ == "__main__":
    unittest.main()
