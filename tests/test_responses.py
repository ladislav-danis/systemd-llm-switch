import unittest
import json
import sqlite3
from pathlib import Path
from src.systemd_llm_switch.db import Database
from src.systemd_llm_switch.main import app

class TestResponsesAPI(unittest.TestCase):
    def setUp(self):
        # Use a temporary database for testing
        self.db_path = "test_llm_switch.db"
        self.db = Database(self.db_path)
        # Mock the global db in main.py
        import src.systemd_llm_switch.main as main
        main.db = self.db

    def tearDown(self):
        # Clean up temporary database
        db_file = Path(__file__).parent.parent / "src" / "systemd_llm_switch" / self.db_path
        if db_file.exists():
            db_file.unlink()

    def test_database_persistence(self):
        conv_id = self.db.create_conversation()
        self.assertTrue(conv_id.startswith("conv_"))
        
        self.db.add_item(conv_id, None, "message", "user", "Hello")
        history = self.db.get_conversation_history(conv_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[0]['content'], 'Hello')

    def test_responses_endpoint_routing(self):
        # The mapping contains tuples of (pattern, handler)
        patterns = [r[0] for r in app.mapping]
        self.assertIn('/v1/responses', patterns)
        self.assertIn('/v1/responses/([^/]+)', patterns)

if __name__ == '__main__':
    unittest.main()
