"""On-device server for the Guandan family edition.

The Android host starts this module inside Chaquopy.  It intentionally keeps
the same Socket.IO event contract as the desktop family edition, so the React
UI and its AI/debug controls remain the source of truth instead of falling
back to the reference game's JavaScript engine.
"""

from __future__ import annotations

import os
import socket
import threading
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room

from .game_manager import Game


_lock = threading.RLock()
_rooms: dict[str, dict] = {}
_started = False


def _addresses():
    result = []
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = item[4][0]
            if not ip.startswith("127.") and ip not in result:
                result.append(ip)
    except OSError:
        pass
    return result


def start(data_dir: str, port: int = 5000):
    """Start the family-edition game server once for this Android process."""
    global _started
    with _lock:
        if _started:
            return "already-started"
        _started = True

    static_dir = os.path.join(data_dir, "family-web")
    app = Flask(__name__, static_folder=static_dir, static_url_path="")
    app.config["SECRET_KEY"] = "guandan-family-mobile"
    CORS(app)
    io = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/server-info")
    def server_info():
        return jsonify({"addresses": _addresses(), "port": port,
                        "edition": "family"})

    @app.get("/api/agents/status")
    def agents_status():
        # The mobile runtime deliberately exposes only dependency-free AI.
        return jsonify({name: {"available": True, "requires": []}
                        for name in ("random", "base1", "base2", "base3", "base4",
                                     "base5", "base6", "base7", "base8")})

    def room_payload(room, viewer):
        data = room["game"].frontend_payload(
            viewer_player_id=viewer,
            include_debug=bool(room.get("debug_enabled")),
            viewer_can_debug=bool(room.get("debug_enabled")),
            viewer_is_host=viewer == room.get("host_seat"),
        )
        return {"state": data["play_state"], "debug_state": data["debug_state"],
                "current_player": room["game"].current_player()}

    def broadcast(room_id, event="update_state"):
        room = _rooms.get(room_id)
        if not room or not room.get("started"):
            return
        for sid, seat in list(room["clients"].items()):
            io.emit(event, room_payload(room, seat), to=sid)

    def drive_ai(room_id):
        room = _rooms.get(room_id)
        if not room or not room.get("started"):
            return
        while room["game"].step_one_ai():
            broadcast(room_id)
            io.sleep(0.45 if room["game"].ai_speed == "slow" else 0.18)
        broadcast(room_id)

    def start_if_ready(room_id):
        room = _rooms.get(room_id)
        if not room or room.get("started"):
            return
        if len(room["clients"]) != len(room["human_seats"]):
            return
        room["started"] = True
        broadcast(room_id, event="game_started")
        io.start_background_task(drive_ai, room_id)

    @io.on("create_room")
    def create_room(data):
        config = dict((data or {}).get("player_config") or {})
        humans = list(config.get("human_player_ids") or [0])
        if not humans:
            emit("error", {"message": "至少需要一名真人玩家。"})
            return
        room_id = uuid.uuid4().hex[:6].upper()
        game = Game(config, seed=config.get("seed"))
        game.init_game()
        room = {"game": game, "host_seat": humans[0], "host_sid": request.sid,
                "clients": {}, "human_seats": humans,
                "debug_enabled": bool(config.get("debug_enabled")), "started": False}
        _rooms[room_id] = room
        seat = humans[0]
        room["clients"][request.sid] = seat
        join_room(room_id)
        emit("room_created", {"roomId": room_id, "playerId": seat,
                              "participantId": f"mobile-{seat}",
                              "sessionId": (data or {}).get("sessionId"),
                              "participants": [{"player_id": x} for x in humans]})
        start_if_ready(room_id)

    @io.on("join_room")
    def join_existing(data):
        data = dict(data or {})
        room_id = str(data.get("roomId") or "").upper()
        room = _rooms.get(room_id)
        if not room:
            emit("error", {"message": "房间不存在。"})
            return
        available = [x for x in room["human_seats"] if x not in room["clients"].values()]
        if not available:
            emit("error", {"message": "没有可加入的真人座位。"})
            return
        seat = available[0]
        room["clients"][request.sid] = seat
        join_room(room_id)
        emit("joined_room", {"roomId": room_id, "playerId": seat,
                              "participantId": f"mobile-{seat}",
                              "sessionId": data.get("sessionId")})
        if room.get("started"):
            emit("game_started", room_payload(room, seat))
        else:
            start_if_ready(room_id)

    @io.on("player_action")
    def player_action(data):
        data = dict(data or {})
        room = _rooms.get(str(data.get("roomId") or "").upper())
        if not room:
            emit("error", {"message": "房间不存在。"})
            return
        sid = request.sid
        seat = room["clients"].get(sid)
        if seat != data.get("playerId"):
            emit("error", {"message": "无权代替其他玩家出牌。"})
            return
        try:
            room["game"].perform_action(seat, data.get("action"))
        except ValueError as exc:
            emit("error", {"message": str(exc)})
            return
        io.start_background_task(drive_ai, str(data.get("roomId")).upper())

    @io.on("set_ai_speed")
    def set_ai_speed(data):
        room = _rooms.get(str((data or {}).get("roomId") or "").upper())
        if room:
            room["game"].set_ai_speed((data or {}).get("speed", "normal"))
            broadcast(str((data or {}).get("roomId")).upper())

    @io.on("set_debug_mode")
    def set_debug_mode(data):
        room = _rooms.get(str((data or {}).get("roomId") or "").upper())
        if room:
            room["debug_enabled"] = bool((data or {}).get("enabled"))
            room["game"].set_debug_enabled(room["debug_enabled"])
            broadcast(str((data or {}).get("roomId")).upper())

    @io.on("disconnect")
    def disconnected():
        for room in _rooms.values():
            room.get("clients", {}).pop(request.sid, None)

    thread = threading.Thread(
        target=lambda: io.run(app, host="0.0.0.0", port=int(port),
                              allow_unsafe_werkzeug=True), daemon=True)
    thread.start()
    return "started"
