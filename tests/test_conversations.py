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

class TestConversationsAPI(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_llm_switch_conv.db"
        self.db = Database(self.db_path)
        main.db = self.db

    def tearDown(self):
        db_file = Path(main.__file__).parent / self.db_path
        if db_file.exists():
            db_file.unlink()

    def test_create_conversation(self):
        main.web._test_data = json.dumps({
            "metadata": {"test": "val"},
            "items": [{"type": "message", "role": "user", "content": "Hello"}]
        })
        handler = main.ConversationsHandler()
        resp = json.loads(handler.POST())
        
        self.assertEqual(resp["object"], "conversation")
        self.assertEqual(resp["metadata"]["test"], "val")
        conv_id = resp["id"]
        
        # Verify item was added
        history = self.db.get_conversation_history(conv_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "Hello")

    def test_retrieve_update_delete_conversation(self):
        conv_id = self.db.create_conversation(metadata={"old": "meta"})
        
        handler = main.ConversationsDetailHandler()
        
        # 1. Retrieve
        resp = json.loads(handler.GET(conv_id))
        self.assertEqual(resp["id"], conv_id)
        self.assertEqual(resp["metadata"]["old"], "meta")
        
        # 2. Update
        main.web._test_data = json.dumps({"metadata": {"new": "meta"}})
        resp = json.loads(handler.POST(conv_id))
        self.assertEqual(resp["metadata"]["new"], "meta")
        self.assertNotIn("old", resp["metadata"])
        
        # 3. Delete
        resp = json.loads(handler.DELETE(conv_id))
        self.assertTrue(resp["deleted"])
        self.assertEqual(resp["object"], "conversation.deleted")
        
        # 4. Verify 404
        handler.GET(conv_id)
        self.assertEqual(main.web.ctx.status, "404 Not Found")

if __name__ == '__main__':
    unittest.main()
