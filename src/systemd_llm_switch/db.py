import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any

class Database:
    def __init__(self, db_path: str = "llm_switch.db"):
        self.db_path = Path(__file__).parent / db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at INTEGER,
                    metadata TEXT
                )
            """)
            # Responses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    model TEXT,
                    status TEXT,
                    usage TEXT,
                    created_at INTEGER,
                    metadata TEXT,
                    instructions TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            # Items table (input/output)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    response_id TEXT,
                    conversation_id TEXT,
                    type TEXT,
                    role TEXT,
                    content TEXT,
                    created_at INTEGER,
                    FOREIGN KEY (response_id) REFERENCES responses(id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            conn.commit()

    def create_conversation(self, metadata: Dict = None) -> str:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        created_at = int(time.time())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, metadata) VALUES (?, ?, ?)",
                (conv_id, created_at, json.dumps(metadata or {}))
            )
            conn.commit()
        return conv_id

    def create_response(
        self, 
        conversation_id: str, 
        model: str, 
        status: str = "in_progress",
        metadata: Dict = None,
        instructions: str = None
    ) -> str:
        resp_id = f"resp_{uuid.uuid4().hex[:12]}"
        created_at = int(time.time())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO responses (id, conversation_id, model, status, created_at, metadata, instructions) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (resp_id, conversation_id, model, status, created_at, json.dumps(metadata or {}), instructions)
            )
            conn.commit()
        return resp_id

    def update_response(self, resp_id: str, status: str, usage: Dict = None, metadata: Dict = None):
        with self._get_connection() as conn:
            if metadata is not None:
                conn.execute(
                    "UPDATE responses SET status = ?, usage = ?, metadata = ? WHERE id = ?",
                    (status, json.dumps(usage or {}), json.dumps(metadata), resp_id)
                )
            else:
                conn.execute(
                    "UPDATE responses SET status = ?, usage = ? WHERE id = ?",
                    (status, json.dumps(usage or {}), resp_id)
                )
            conn.commit()

    def add_item(
        self, 
        conversation_id: str, 
        response_id: Optional[str], 
        item_type: str, 
        role: str, 
        content: Any
    ) -> str:
        item_id = f"item_{uuid.uuid4().hex[:12]}"
        created_at = int(time.time())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO items (id, response_id, conversation_id, type, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item_id, response_id, conversation_id, item_type, role, json.dumps(content), created_at)
            )
            conn.commit()
        return item_id

    def get_conversation_history(self, conversation_id: str, up_to_response_id: Optional[str] = None) -> List[Dict]:
        if up_to_response_id:
            # More precise filtering: get items that either:
            # 1. Belong to a response that was created at or before the target response
            # 2. Are input items (response_id IS NULL) created before the target response
            query = """
                SELECT type, role, content FROM items 
                WHERE conversation_id = ? 
                AND (
                    (response_id IS NOT NULL AND response_id IN (
                        SELECT id FROM responses WHERE conversation_id = ? 
                        AND created_at <= (SELECT created_at FROM responses WHERE id = ?)
                    ))
                    OR 
                    (response_id IS NULL AND created_at < (
                        SELECT created_at FROM responses WHERE id = ?
                    ))
                )
                ORDER BY created_at ASC
            """
            params = [conversation_id, conversation_id, up_to_response_id, up_to_response_id]
        else:
            query = "SELECT type, role, content FROM items WHERE conversation_id = ? ORDER BY created_at ASC"
            params = [conversation_id]
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            history = []
            
            for row in rows:
                item_type = row['type']
                content = json.loads(row['content'])
                
                if item_type == "message":
                    history.append({
                        "role": row['role'],
                        "content": content.get("text") if isinstance(content, dict) else content
                    })
                elif item_type == "function_call":
                    # Tool call from assistant
                    # Check if last message was also an assistant message with tool_calls
                    if history and history[-1]["role"] == "assistant" and "tool_calls" in history[-1]:
                        history[-1]["tool_calls"].append({
                            "id": content.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": content.get("name"),
                                "arguments": content.get("arguments")
                            }
                        })
                    else:
                        history.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": content.get("call_id"),
                                "type": "function",
                                "function": {
                                    "name": content.get("name"),
                                    "arguments": content.get("arguments")
                                }
                            }]
                        })
                elif item_type == "tool":
                    # Output from a tool
                    history.append({
                        "role": "tool",
                        "tool_call_id": content.get("tool_call_id"),
                        "content": content.get("content")
                    })
            
            return history

    def get_response(self, resp_id: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM responses WHERE id = ?", (resp_id,)).fetchone()
            if not row:
                return None
            
            output_items = self.get_response_items(resp_id)

            return {
                "id": row['id'],
                "object": "response",
                "created_at": row['created_at'],
                "model": row['model'],
                "status": row['status'],
                "conversation_id": row['conversation_id'],
                "usage": json.loads(row['usage']) if row['usage'] else None,
                "output": output_items,
                "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                "instructions": row['instructions']
            }

    def get_response_items(self, resp_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            items_rows = conn.execute(
                "SELECT * FROM items WHERE response_id = ?", 
                (resp_id,)
            ).fetchall()
            
            output_items = []
            for ir in items_rows:
                item_type = ir['type']
                content_raw = json.loads(ir['content'])
                
                item = {
                    "id": ir['id'],
                    "object": "item",
                    "type": item_type,
                    "status": "completed",
                    "created_at": ir['created_at']
                }

                if item_type == "message":
                    item["role"] = ir['role']
                    item["content"] = [
                        {
                            "type": "output_text",
                            "text": content_raw.get("text") if isinstance(content_raw, dict) else content_raw,
                            "annotations": []
                        }
                    ]
                elif item_type == "function_call":
                    item["call_id"] = content_raw.get("call_id")
                    item["name"] = content_raw.get("name")
                    item["arguments"] = content_raw.get("arguments")
                
                output_items.append(item)
            return output_items

    def delete_response(self, resp_id: str) -> bool:
        with self._get_connection() as conn:
            # Delete associated items first
            conn.execute("DELETE FROM items WHERE response_id = ?", (resp_id,))
            # Delete the response
            cursor = conn.execute("DELETE FROM responses WHERE id = ?", (resp_id,))
            conn.commit()
            return cursor.rowcount > 0
