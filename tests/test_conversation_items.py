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
mock_web.input = lambda **kwargs: MagicMock(**kwargs)
sys.modules['web'] = mock_web

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from systemd_llm_switch import main
from systemd_llm_switch.db import Database

class TestConversationItemsAPI(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_llm_switch_items.db"
        self.db = Database(self.db_path)
        main.db = self.db

    def tearDown(self):
        db_file = Path(main.__file__).parent / self.db_path
        if db_file.exists():
            db_file.unlink()

    def test_item_management(self):
        """Test creating, listing, retrieving, and deleting conversation items via the API handlers."""
        conv_id = self.db.create_conversation()
        
        # 1. Create Items
        main.web._test_data = json.dumps({
            "items": [{"type": "message", "role": "user", "content": "Hello"}]
        })
        handler = main.ConversationItemsHandler()
        resp = json.loads(handler.POST(conv_id))
        self.assertEqual(resp["object"], "list")
        self.assertEqual(len(resp["data"]), 1)
        item_id = resp["data"][0]["id"]
        
        # 2. List Items
        with patch('web.input', return_value=MagicMock(limit=20, after=None, order="desc")):
            resp = json.loads(handler.GET(conv_id))
            self.assertEqual(len(resp["data"]), 1)
            self.assertEqual(resp["data"][0]["id"], item_id)
        
        # 3. Retrieve Detail
        detail_handler = main.ConversationItemDetailHandler()
        resp = json.loads(detail_handler.GET(conv_id, item_id))
        self.assertEqual(resp["id"], item_id)
        self.assertEqual(resp["role"], "user")
        
        # 4. Delete Item
        resp = json.loads(detail_handler.DELETE(conv_id, item_id))
        self.assertEqual(resp["object"], "conversation")
        self.assertEqual(resp["id"], conv_id)
        
        # Verify gone
        detail_handler.GET(conv_id, item_id)
        self.assertEqual(main.web.ctx.status, "404 Not Found")

if __name__ == '__main__':
    unittest.main()
