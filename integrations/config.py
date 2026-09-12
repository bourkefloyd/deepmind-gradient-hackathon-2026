"""Environment-driven configuration. Nothing here reads the network."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    nango_secret_key: str
    nango_base_url: str
    integration_id: str          # Nango "Provider-Config-Key" (integration unique key)
    connection_id: str           # Nango "Connection-Id"
    recap_action: str            # Nango action name that posts text to Discord
    public_url: str              # where humans join, e.g. https://wordhunt.example.run.app
    timeout_s: float             # per-request wall clock; the game never waits longer than this
    discord_channel_id: str      # v2 only (thread flow via proxy / second action)

    @property
    def dry_run(self) -> bool:
        return not self.nango_secret_key

    def room_url(self, code: str) -> str:
        return f"{self.public_url.rstrip('/')}/r/{code}"

    def missing(self) -> list[str]:
        """Env vars still needed before live sends work."""
        out = []
        if not self.nango_secret_key:
            out.append("NANGO_SECRET_KEY")
        if not self.connection_id:
            out.append("NANGO_CONNECTION_ID")
        return out


def load() -> Settings:
    return Settings(
        nango_secret_key=os.environ.get("NANGO_SECRET_KEY", "").strip(),
        nango_base_url=os.environ.get("NANGO_BASE_URL", "https://api.nango.dev").rstrip("/"),
        integration_id=os.environ.get("NANGO_DISCORD_INTEGRATION_ID", "discord-wordhunt").strip(),
        connection_id=os.environ.get("NANGO_CONNECTION_ID", os.environ.get("NANGO_DISCORD_CONNECTION_ID", "")).strip(),
        recap_action=os.environ.get("NANGO_DISCORD_RECAP_ACTION", "send-discord-recap").strip(),
        public_url=os.environ.get("WH_PUBLIC_URL", "http://localhost:8000").strip(),
        timeout_s=float(os.environ.get("WH_INTEGRATIONS_TIMEOUT_S", "8")),
        discord_channel_id=os.environ.get("DISCORD_CHANNEL_ID", "").strip(),
    )
