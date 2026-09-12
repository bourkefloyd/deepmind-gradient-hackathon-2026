"""Nango integrations for the Word Hunt arena (Discord first).

Layout:
    config.py      env vars, dry-run detection
    nango.py       thin HTTP client: action trigger, connection list, proxy (stdlib only)
    discord.py     Discord operations on top of nango.py (send_recap; v2 thread flow)
    formatters.py  message templates for the game events
    events.py      async fire-and-forget dispatcher: emit(event_type, payload)
    handlers.py    event -> formatter -> discord wiring; register() installs the v1 handlers
    tools.py       OpenAI tool schema so a Gemma commentator can call send-discord-recap
    demo.py        python -m integrations.demo [--live] [--connections]

The game only touches `integrations.events.emit`; everything else is replaceable.
"""
