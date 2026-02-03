## 1. Python Style & Typing
* **Style Guide:** Strictly adhere to **PEP 8**.
* **Typing:** Use explicit **type hints** in all function and method signatures (e.g., `def scale(scalar: float, vector: list[float]) -> list[float]:`).
* **Naming:** Use `snake_case` for functions/variables and `PascalCase` for classes.

## 2. DRY Principle (Don't Repeat Yourself)
* **Abstraction:** Abstract repetitive logic into reusable functions, methods, or decorators.
* **Single Source of Truth:** Constants, configuration strings, and validation rules reside in config.yaml.
* **Rule of Three:** If a logic pattern repeats more than twice, it **must** be refactored into a shared component or utility.
* **Centralization:** Constants, configuration strings, and validation rules should reside in a single location.

## 3. Documentation
* **Docstrings:** Use **Google Style Docstrings**.
* **Mandatory Fields:** Every non-trivial function must include `Args:` and `Returns:` sections.
* **Verification:** Use the `Examples:` section with **doctests** for core business logic to ensure documentation stays functional.

## 4. Error Handling & Logging
* **Library:** Use the standard Python `logging` library for all diagnostic output.
* **Tracebacks:** Always include `exc_info=True` when logging exceptions within `except` blocks to ensure full debuggability.
* **Clean Exits:** Use try-except blocks strategically to catch expected failures and log them appropriately.

## 5. Frameworks & Tools
* **Primary Framework:** Use **web.py**. 
* **Concurrency:** Sequential model switching. Use threading.Lock to ensure only one model is handled at a time, preventing VRAM race conditions.
* **Dependencies:** Use `requests` with `stream=True` for forwarding. Do not use `httpx` or other async libraries unless explicitly requested.

## 6. Deployment & Structure
* **Native Execution:** The project runs natively on the host OS to allow direct interaction with `systemctl --user`. Do NOT use Docker/Podman for deployment.
* **Environment:** Use a local Python Virtual Environment (`.venv`) located in the root directory.
* **Isolation:** All source code and user configuration (config.yaml) must reside in the src/systemd_llm_switch/ directory.
* **Installation:** Provide a `setup.sh` or clear instructions in `README.md` for symlinking systemd unit files from `deploy/systemd/` to `~/.config/systemd/user/`.

## 7. Systemd (User Mode)
* **Execution:** Always use `systemctl --user` for service management.
* **Privileges:** No `sudo` is required. The script has native permission to manage its own user services.
* **Linger:** Ensure `loginctl enable-linger $USER` is mentioned in docs to keep services running after logout.

## 8. Streaming Implementation
* Web.py does not use async/await. Streaming must be implemented using **Python generators** (`yield`) within the handler methods.
