import unittest
import requests
import time
import socket

PROXY_URL = "http://localhost:3002/v1/chat/completions"
TEST_MODEL = "qwen3-coder-next"

def is_proxy_running() -> bool:
    """Checks if the proxy server is listening on port 3002.
    
    Returns:
        True if the port is open, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', 3002)) == 0

class TestSmoke(unittest.TestCase):
    """Smoke tests for the running proxy server."""

    @unittest.skipUnless(is_proxy_running(), "Proxy server is not running on localhost:3002")
    def test_chat_completion_smoke(self) -> None:
        """Performs a real-world integration test against the running proxy.
        
        This test expects the proxy server to be already active. It verifies
        that the model can be loaded and returns a valid response.
        """
        print(f"\n🚀 Starting smoke test against {PROXY_URL}...")
        print(f"📦 Target model: {TEST_MODEL} (may take time if loading)")

        payload = {
            "model": TEST_MODEL,
            "messages": [
                {"role": "user", "content": "Say 'OK' if you are working."}
            ],
            "stream": False
        }

        start_time = time.time()
        # Longer time limit because the model
        # may be loaded into VRAM for the first time
        response = requests.post(PROXY_URL, json=payload, timeout=120)
        duration = time.time() - start_time

        self.assertEqual(response.status_code, 200, f"Failed! Status code: {response.status_code}, Details: {response.text}")
        
        data = response.json()
        self.assertIn('choices', data)
        self.assertTrue(len(data['choices']) > 0)
        
        answer = data['choices'][0]['message']['content']
        print(f"✅ Success! (Time: {duration:.2f}s)")
        print(f"🤖 Model response: {answer}")
        
        self.assertTrue(len(answer) > 0)

if __name__ == "__main__":
    unittest.main()
