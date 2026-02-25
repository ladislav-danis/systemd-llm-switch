import requests
import time
import sys

PROXY_URL = "http://localhost:3002/v1/chat/completions"
TEST_MODEL = "qwen3-coder-flash"


def run_smoke_test():
    """
    Performs a real-world integration test against the running proxy.
    This test expects the proxy server to be already active.
    """
    print(f"🚀 Starting smoke test against {PROXY_URL}...")
    print(
        f"📦 Target model: {TEST_MODEL} "
        "(this may take a minute if model is loading)"
    )

    payload = {
        "model": TEST_MODEL,
        "messages": [
            {"role": "user", "content": "Say 'OK' if you are working."}
        ],
        "stream": False
    }

    start_time = time.time()
    try:
        # Longer time limit because the model
        # may be loaded into VRAM for the first time
        response = requests.post(PROXY_URL, json=payload, timeout=120)
        duration = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content']
            print(f"✅ Success! (Time: {duration:.2f}s)")
            print(f"🤖 Model response: {answer}")
        else:
            print(f"❌ Failed! Status code: {response.status_code}")
            print(f"📝 Error details: {response.text}")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Proxy server is not running on localhost:3002")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
