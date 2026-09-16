"""Live Jump Guy constants, extracted from https://game.jumpguy.net (2026-09-16).

The Viva+ page embeds this origin in an iframe (`url="https://game.jumpguy.net"`).
Play the game origin directly: the embed parent is a Fourthwall storefront with
reCAPTCHA, and the game CSP is `frame-ancestors` viva+/fourthwall only.

Physics / input (Phaser 3 Arcade, from the shipped `main-*.js` bundle):

- canvas 960x540, gravity y=1800, jump velocity=-700
- jump buffer 130 ms, coyote time 100 ms
- Space / tap / click to jump; R or Space to restart after game over
- obstacles spawn to the right, drift left at 260..520 px/s (+4.5 px/s^2)
- +1 score when an obstacle's right edge passes the player's left edge
- server-authoritative scoring via POST /api/runs, /pass, /end, /submit
"""

from __future__ import annotations

# Official playable origin (not the Viva+ marketing wrapper).
GAME_URL = "https://game.jumpguy.net"
VIVA_PAGE_URL = "https://vivaplus.tv/pages/jumpguy"

# If a score/achievement UI asks for a player name, use this exactly.
PLAYER_NAME = "Grok Bot Son"

CANVAS_W = 960
CANVAS_H = 540
GRAVITY_Y = 1800.0
JUMP_VELOCITY = -700.0
JUMP_BUFFER_MS = 130.0
COYOTE_MS = 100.0

PLAYER_X_FRAC = 0.22
GROUND_Y_FRAC = 0.72
PLAYER_DISPLAY_W = 72.0
PLAYER_DISPLAY_H = 84.0

SPEED_START = 260.0
SPEED_MAX = 520.0
SPEED_ACCEL = 4.5  # px/s per second of running

SPAWN_COOLDOWN_RESET_MS = 800.0
SPAWN_COOLDOWN_BASE_MS = 1200.0
SPAWN_COOLDOWN_JITTER = (-120, 240)
SPAWN_COOLDOWN_CLAMP_MS = (700.0, 1500.0)
SPAWN_X_JITTER = (40, 120)
OBSTACLE_SIZE_RANGE = (28, 34)

# Arcade world / ground body (visual collision uses display bounds, not the body).
GROUND_BODY_Y_OFFSET = 30.0
GROUND_BODY_H = 60.0

TICK_HZ = 60.0
DT = 1.0 / TICK_HZ

# Observation for the learned policy.
FRAME_SIZE = 84
FRAME_STACK = 4

# Obstacle tints from the live spawn list (0xRRGGBB).
OBSTACLE_COLORS = (
    (255, 107, 107),
    (255, 179, 71),
    (255, 217, 61),
    (107, 203, 119),
    (77, 150, 255),
    (179, 157, 255),
)

# Approximate player palette (generated at runtime in BootScene).
PLAYER_HAT = (88, 106, 136)
PLAYER_SKIN = (240, 196, 160)
PLAYER_SHIRT = (111, 124, 124)
PLAYER_PANTS = (47, 93, 61)
PLAYER_SHOES = (111, 76, 40)

AIR_TIME_S = 2.0 * abs(JUMP_VELOCITY) / GRAVITY_Y  # ~0.778 s
TIME_TO_PEAK_S = abs(JUMP_VELOCITY) / GRAVITY_Y  # ~0.389 s
MAX_JUMP_HEIGHT = (JUMP_VELOCITY**2) / (2.0 * GRAVITY_Y)  # ~136 px
