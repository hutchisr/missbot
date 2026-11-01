"""Small durable inbox, outbox, and private conversation history for hosted Buzz."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..core import HistoryTurn


class BuzzStore:
    def __init__(self, path: str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(target)
        target.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=1000;
            CREATE TABLE IF NOT EXISTS communities (
                community TEXT PRIMARY KEY, name TEXT NOT NULL, picture TEXT,
                owner TEXT, auth TEXT, active INTEGER NOT NULL DEFAULT 0,
                since INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inbox (
                community TEXT NOT NULL, id TEXT NOT NULL, event TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending', received INTEGER NOT NULL,
                conversation TEXT, author TEXT, reply TEXT, outgoing TEXT,
                PRIMARY KEY (community,id)
            );
            CREATE TABLE IF NOT EXISTS history (
                community TEXT NOT NULL, conversation TEXT NOT NULL, turns TEXT NOT NULL,
                touched INTEGER NOT NULL, PRIMARY KEY (community,conversation)
            );
        """)

    def communities(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.execute("SELECT * FROM communities ORDER BY community")]

    def community(self, community: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM communities WHERE community=?", (community,)).fetchone()
        return dict(row) if row is not None else None

    def provision(self, community: str, name: str, picture: str | None) -> dict[str, Any]:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO communities(community,name,picture,since) VALUES(?,?,?,?)",
                (community, name, picture, int(time.time())),
            )
        result = self.community(community)
        assert result is not None
        return result

    def activate(self, community: str, owner: str, auth: list[str], *, active: bool = True) -> None:
        with self.db:
            self.db.execute(
                "UPDATE communities SET owner=?,auth=?,active=? WHERE community=?",
                (owner, json.dumps(auth), int(active), community),
            )

    def profile(self, community: str, name: str, picture: str | None) -> None:
        with self.db:
            self.db.execute("UPDATE communities SET name=?,picture=? WHERE community=?", (name, picture, community))

    def deactivate(self, community: str) -> None:
        # The shared Nostr key and ownership record survive Buzz removal.
        with self.db:
            self.db.execute("UPDATE communities SET active=0 WHERE community=?", (community,))
            self.db.execute("DELETE FROM inbox WHERE community=?", (community,))
            self.db.execute("DELETE FROM history WHERE community=?", (community,))

    def enqueue(self, community: str, event: dict[str, Any]) -> bool:
        if (
            self.db.execute("SELECT count(*) FROM inbox WHERE community=? AND state!='done'", (community,)).fetchone()[
                0
            ]
            >= 128
        ):
            return False
        with self.db:
            inserted = self.db.execute(
                "INSERT OR IGNORE INTO inbox(community,id,event,received) VALUES(?,?,?,?)",
                (community, event["id"], json.dumps(event), int(time.time())),
            ).rowcount
            self.db.execute(
                "UPDATE communities SET since=max(since,?) WHERE community=?", (event["created_at"], community)
            )
        return bool(inserted)

    def next_event(self, community: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM inbox WHERE community=? AND state!='done' ORDER BY received,rowid LIMIT 1", (community,)
        ).fetchone()
        return dict(row) if row is not None else None

    def outbox(self, community: str, event_id: str, conversation: str, author: str, reply: str, outgoing: dict) -> None:
        with self.db:
            self.db.execute(
                "UPDATE inbox SET state='outbox',conversation=?,author=?,reply=?,outgoing=? WHERE community=? AND id=?",
                (conversation, author, reply, json.dumps(outgoing), community, event_id),
            )

    def history(self, community: str, conversation: str) -> list[HistoryTurn]:
        row = self.db.execute(
            "SELECT turns FROM history WHERE community=? AND conversation=?", (community, conversation)
        ).fetchone()
        return [HistoryTurn(**turn) for turn in json.loads(row[0])] if row is not None else []

    def abandon(self, community: str, event_id: str) -> None:
        """Keep deduplication, but never record an unconfirmed reply as delivered."""
        with self.db:
            self.db.execute(
                "UPDATE inbox SET conversation=NULL,reply=NULL,outgoing=NULL WHERE community=? AND id=?",
                (community, event_id),
            )
        self.finish(community, event_id)

    def finish(self, community: str, event_id: str, *, history_turns: int = 20) -> None:
        row = self.db.execute("SELECT * FROM inbox WHERE community=? AND id=?", (community, event_id)).fetchone()
        if row is None or row["state"] == "done":
            return
        with self.db:
            if row["conversation"]:
                history = self.history(community, row["conversation"])
                event = json.loads(row["event"])
                turns = [{"role": turn.role, "text": turn.text, "author": turn.author} for turn in history]
                turns.append({"role": "user", "text": event["content"], "author": row["author"]})
                if row["reply"]:
                    turns.append({"role": "assistant", "text": row["reply"]})
                self.db.execute(
                    "INSERT OR REPLACE INTO history VALUES(?,?,?,?)",
                    (community, row["conversation"], json.dumps(turns[-2 * history_turns :]), int(time.time())),
                )
            self.db.execute(
                "UPDATE inbox SET state='done',outgoing=NULL WHERE community=? AND id=?", (community, event_id)
            )
            self.db.execute(
                "DELETE FROM inbox WHERE community=? AND state='done' AND rowid NOT IN "
                "(SELECT rowid FROM inbox WHERE community=? AND state='done' ORDER BY received DESC,rowid DESC LIMIT 2048)",
                (community, community),
            )
            self.db.execute(
                "DELETE FROM history WHERE community=? AND rowid NOT IN "
                "(SELECT rowid FROM history WHERE community=? ORDER BY touched DESC,rowid DESC LIMIT 128)",
                (community, community),
            )

    def close(self) -> None:
        self.db.close()
