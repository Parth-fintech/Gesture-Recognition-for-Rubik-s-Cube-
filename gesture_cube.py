"""gesture_cube.py - a hand-tracked, gesture-controlled virtual Rubik's cube.

Point your right index finger into the on-screen touchpad, pinch to lock a
tile, and drag to turn the layer it belongs to - the sticker follows your
finger around its true curved path in real time. Your left hand orbits the
whole puzzle with an open palm and freezes it with a fist.

Run:
    python gesture_cube.py

Requires a webcam and a MediaPipe hand-landmarker model file
(hand_landmarker.task) - see README.md for setup.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time, sys, os, random
from collections import deque
from dotenv import load_dotenv

# Load variables from a .env file sitting next to this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# MediaPipe
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]
LANDMARK_NAMES = [
    "WRIST","TH_CMC","TH_MCP","TH_IP","TH_TIP",
    "IX_MCP","IX_PIP","IX_DIP","IX_TIP",
    "MD_MCP","MD_PIP","MD_DIP","MD_TIP",
    "RG_MCP","RG_PIP","RG_DIP","RG_TIP",
    "PK_MCP","PK_PIP","PK_DIP","PK_TIP",
]

W_FRAME, H_FRAME = 640, 480
CX, CY = 460, 190
FOCAL  = 20

def project(pts3d):
    return [(int(CX + x * FOCAL), int(CY - y * FOCAL)) for x, y, z in pts3d]

# Rotation
def Rx(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def R_mat(ax, ay, az):
    return Rz(az) @ Ry(ay) @ Rx(ax)

ROT = (Rx, Ry, Rz)   # index by axis: 0=x, 1=y, 2=z

def rot90_int(axis, sign):
    # Exact integer 90 degree rotation about an axis - cube state never drifts.
    return np.rint(ROT[axis](90 * sign)).astype(int)

#  Puzzle cube model - 26 cubies, each tracked by position + orientation

CORNERS = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
], dtype=float)

FACES_TOPO = [
    ([4,5,6,7], ( 0, 0, 1)),
    ([0,3,2,1], ( 0, 0,-1)),
    ([1,2,6,5], ( 1, 0, 0)),
    ([0,4,7,3], (-1, 0, 0)),
    ([3,7,6,2], ( 0, 1, 0)),
    ([0,1,5,4], ( 0,-1, 0)),
]

FACE_COLORS = {
    ( 1, 0, 0): ( 44,  48, 235),   # Right  - red     (vivid, molded plastic)
    (-1, 0, 0): (  0, 126, 255),   # Left   - orange
    ( 0, 1, 0): (250, 250, 250),   # Up     - white
    ( 0,-1, 0): (  0, 208, 255),   # Down   - yellow
    ( 0, 0, 1): ( 64, 205,  72),   # Front  - green
    ( 0, 0,-1): (255, 144,  30),   # Back   - blue    (bright azure)
}
GAP           = 0.13
CUBIE_H       = 0.5 * (1.0 - GAP)

# --- Stickerless-speedcube look: no black frames - each cubie face is a
# --- near-full-bleed rounded tile of colored plastic; the dark base quad
# --- peeking through at corners and edges reads as the seams between pieces.
SEAM          = ( 30,  28,  28)   # plastic shadow between tiles
TILE_BLEED    = 0.965             # tile size vs cubie face (thin seams)
TILE_ROUND    = 0.38              # corner radius ratio (photo-matched)
PILLOW_SCALE  = 0.60              # inner highlight size -> domed tile look
PILLOW_LIFT   = 0.22              # how much the dome center lightens
SHADE_FLOOR   = 0.60              # ambient term keeps colors candy-bright

def _rounded_square2d(half, ratio, seg=4):
    r, c = half * ratio, half * (1.0 - ratio)
    out = []
    for cx, cy, a0 in ((c, c, 0.0), (-c, c, 90.0), (-c, -c, 180.0), (c, -c, 270.0)):
        for k in range(seg + 1):
            a = np.radians(a0 + 90.0 * k / seg)
            out.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(out)

_T2D    = _rounded_square2d(CUBIE_H * TILE_BLEED, TILE_ROUND)
_T2D_HL = _rounded_square2d(CUBIE_H * 1.06, TILE_ROUND)   # hover ring shape

TILE3 = {}   # per outward normal: rounded tile points in cubie-local coords
for _n in FACE_COLORS:
    _nv = np.array(_n, dtype=float)
    _a  = int(np.argmax(np.abs(_nv)))
    _t1 = np.zeros(3); _t1[(_a + 1) % 3] = 1.0
    _t2 = np.cross(_nv, _t1)
    TILE3[_n] = np.array([_nv * CUBIE_H + u * _t1 + v * _t2 for u, v in _T2D])

class Cubie:
    def __init__(self, grid):
        self.pos = np.array(grid, dtype=int)
        self.ori = np.eye(3, dtype=int)
        self.stickers = {n: col for n, col in FACE_COLORS.items()
                         if int(np.dot(self.pos, n)) == 1}

def solved_cubies():
    return [Cubie((x, y, z))
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)]

cubies = solved_cubies()

#  Moves

MOVES = {
    'r': (0,  1, -1), 'l': (0, -1,  1),
    'u': (1,  1, -1), 'd': (1, -1,  1),
    'f': (2,  1, -1), 'b': (2, -1,  1),
}
MID_NAME = {0: "M", 1: "E", 2: "S"}          # middle slices, gesture-only
MID_CANON = {0: 1, 1: 1, 2: -1}              # M follows L, E follows D, S follows F

TURN_TIME  = 0.26
move_queue = []
anim       = None
last_move  = "--"
move_count = 0

def move_label(axis, layer, sign):
    if layer == 0:
        return MID_NAME[axis] if sign == MID_CANON[axis] else MID_NAME[axis] + "'"
    for k, (a, l, s) in MOVES.items():
        if a == axis and l == layer:
            return k.upper() if s == sign else k.upper() + "'"
    return "?"

def start_move(axis, layer, sign):
    global anim, last_move
    anim = {"axis": axis, "layer": layer, "sign": sign, "t0": time.time(),
            "ids": [i for i, c in enumerate(cubies) if c.pos[axis] == layer]}
    last_move = move_label(axis, layer, sign)

def finish_move():
    global anim, move_count
    M = rot90_int(anim["axis"], anim["sign"])
    for i in anim["ids"]:
        c = cubies[i]
        c.pos = M @ c.pos
        c.ori = M @ c.ori
    move_count += 1
    anim = None

def scramble(n=14):
    prev = None
    added = 0
    while added < n:
        axis  = random.randrange(3)
        layer = random.choice((-1, 1))
        sign  = random.choice((-1, 1))
        if prev == (axis, layer):
            continue
        move_queue.append((axis, layer, sign))
        prev = (axis, layer)
        added += 1

def reset_cube():
    global cubies, anim, move_count, last_move
    cubies = solved_cubies()
    move_queue.clear()
    anim = None
    move_count = 0
    last_move = "--"

def is_solved():
    seen = {}
    for c in cubies:
        for n_local, col in c.stickers.items():
            g = tuple((c.ori @ np.array(n_local)).tolist())
            if g in seen and seen[g] != col:
                return False
            seen[g] = col
    return True

# --- An anim can be "auto" (keyboard/queued move), "drag" (finger-driven,
# --- live), or "settle" (springing home to 0 / +90 / -90 on release or
# --- auto-commit). draw_puzzle's "angle" branch renders the latter two.

def begin_drag(axis, layer):
    global anim
    anim = {"mode": "drag", "axis": axis, "layer": layer, "sign": 0,
            "t0": time.time(), "angle": 0.0,
            "ids": [i for i, c in enumerate(cubies) if c.pos[axis] == layer]}

def settle_from_drag(target):
    # Re-target the current drag anim to glide to 0 / +90 / -90.
    global anim
    if anim is None:
        return
    anim["mode"]   = "settle"
    anim["from"]   = float(anim.get("angle", 0.0))
    anim["target"] = float(target)
    anim["t0"]     = time.time()

def update_settle():
    global anim, last_move
    if anim is None or anim.get("mode") != "settle":
        return
    t = (time.time() - anim["t0"]) / SETTLE_TIME
    if t >= 1.0:
        if anim["target"] != 0.0:
            anim["sign"] = int(anim["target"] / 90.0)
            last_move = move_label(anim["axis"], anim["layer"], anim["sign"])
            finish_move()                      # commits via the existing path
        else:
            anim = None                        # sprang back, no state change
    else:
        e = t * t * (3.0 - 2.0 * t)
        anim["angle"] = anim["from"] + (anim["target"] - anim["from"]) * e

#  Renderer

def draw_puzzle(frame, Rg, half_edge, anim_state):
    S = half_edge / 1.5

    A = None
    moving = ()
    if anim_state is not None:
        if "angle" in anim_state:                                  # unused
            A = ROT[anim_state["axis"]](float(anim_state["angle"]))
        else:
            t = min((time.time() - anim_state["t0"]) / TURN_TIME, 1.0)
            e = t * t * (3.0 - 2.0 * t)
            A = ROT[anim_state["axis"]](90.0 * anim_state["sign"] * e)
        moving = set(anim_state["ids"])

    render = []
    for i, c in enumerate(cubies):
        Rl  = c.ori.astype(float)
        pts = (CORNERS * CUBIE_H) @ Rl.T + c.pos
        Ai  = A if (A is not None and i in moving) else None
        if Ai is not None:
            pts = pts @ Ai.T
        wpts = (pts * S) @ Rg.T
        render.append((wpts[:, 2].mean(), i, wpts, Ai, Rl))

    render.sort(key=lambda r: r[0])

    hull_pts = np.array([pt for r in render for pt in project(r[2])], np.int32)
    cv2.fillPoly(frame, [cv2.convexHull(hull_pts)], (16, 14, 14))

    for _, i, wpts, Ai, Rl in render:
        c  = cubies[i]
        p2 = project(wpts)
        for verts, n_local in FACES_TOPO:
            n = Rl @ np.array(n_local, dtype=float)
            if Ai is not None:
                n = Ai @ n
            n_w = Rg @ n
            if n_w[2] < 0.03:
                continue
            quad = np.array([p2[v] for v in verts], np.int32)
            cv2.fillPoly(frame, [quad], SEAM)          # seams + bare internals

            col = c.stickers.get(n_local)
            if col is None:
                continue
            # Rounded near-full-bleed tile, transformed through the exact same
            # chain as the cubie corners (orientation -> layer anim -> world).
            tp = TILE3[n_local] @ Rl.T + c.pos
            if Ai is not None:
                tp = tp @ Ai.T
            poly = np.array(project((tp * S) @ Rg.T), np.int32)
            sh   = SHADE_FLOOR + (1.0 - SHADE_FLOOR) * float(n_w[2])
            base = tuple(int(v * sh) for v in col)
            cv2.fillPoly(frame, [poly], base)
            # Pillow: a lighter inner tile scaled about the centroid - under
            # orthographic projection 2D scaling equals scaling in 3D, so the
            # dome highlight stays glued to the tile through every animation.
            ctr = poly.mean(axis=0)
            pil = (ctr + (poly - ctr) * PILLOW_SCALE).astype(np.int32)
            pcol = tuple(int(min(255, v * sh + (255 - v * sh) * PILLOW_LIFT)) for v in col)
            cv2.fillPoly(frame, [pil], pcol)

# ============================================================================
#  GESTURE INTERACTION LAYER
#
#  The cube never has to be touched on screen: a virtual touchpad in the
#  bottom-left corner is the only thing
#  the right hand interacts with. Everything below translates
#  index-fingertip motion inside that box -> a cursor on whichever cube face
#  is currently most camera-facing -> a circular "confirm" gesture that
#  locks a tile -> a directional flick that becomes exactly one call into
#  the SAME move_queue / start_move / finish_move pipeline above.
# ============================================================================

# --- Tunables ----------------------------------------------------- [TUNE] --
SURF_HALF        = 1.0 + CUBIE_H     # true sticker surface plane (cube units)
MIN_TANGENT      = 25.0    # px/rad: below this a layer is too edge-on to turn
TANGENT_QUALITY  = 0.30    # fraction of the tangent that must survive screen
                           # projection; gates out hair-trigger oblique locks


# Virtual control box (bottom-left corner)
BOX_MARGIN       = 60
BOX_W, BOX_H     = 250, 250
BOX_X0, BOX_Y1   = BOX_MARGIN, H_FRAME - BOX_MARGIN -60
BOX_X1, BOX_Y0   = BOX_X0 + BOX_W, BOX_Y1 - BOX_H
BOX_EXIT_MARGIN  = 16

IDX_EMA          = 0.40    # index-fingertip smoothing (higher = snappier)
WRIST_EMA        = 0.50    # wrist smoothing, used only for frame-to-frame ID
TRAIL_LEN        = 24      # points kept for the on-screen finger trail

# Pinch is a held CLUTCH: pinch down selects and
# locks the hovered tile; pinch up deselects - mid-drag, that release also
# springs the layer to the nearest quarter turn. Same thumb/index-distance
# classifier as always, with hysteresis so it can't chatter at the threshold.
PINCH_ON         = 0.32    # pinch registers CLOSED below this (of palm scale)
PINCH_OFF        = 0.48    # must open back past this before it can re-fire

AXIS_PICK_PX     = 14.0    # dead zone (raw px) before a direction commits
                           # to a specific axis - first meaningful move wins
DRAG_GAIN        = 1.8     # >1 = less finger travel per degree of turn
ANGLE_CLAMP      = 120.0   # max live-drag angle before it's clamped
AUTO_COMMIT_DEG  = 92.0    # reaching this mid-drag commits the quarter turn
COMMIT_DEG       = 45.0    # release beyond this snaps forward, else springs back
SETTLE_TIME      = 0.15    # seconds for the release/auto-commit snap glide
COOLDOWN_T       = 0.5   # strict post-move lockout (spec-mandated)

ORBIT_OPEN_RATIO = 0.28    # avg fingertip/scale ratio: open hand vs fist
ROTATION_SLOWDOWN = 0.83   # whole-cube rotation at 83% speed (17% slower),
                           # applied uniformly across the whole speed range
                           # so it's 17% slower whether you nudge or swipe
HAND_DROP_FRAMES = 4       # tracking-dropout grace before a hand is "lost"
HAND_MATCH_PX    = 160     # frame-to-frame hand identity matching radius
HAND_JUMP_PX     = 120     # orbit centroid teleport guard (hand-swap safety)
HAND_SWAP        = False   # press H at runtime if your camera's Left/Right
                           # labels arrive inverted (MediaPipe convention
                           # varies with mirroring) - roles flip instantly
# ----------------------------------------------------------------------------

def tangent_screen(rot_axis, p, Rg, S):
    # Screen direction the point p moves for +rotation about rot_axis, plus
    # its magnitude in px/rad. Returns None when the tangent is degenerate:
    # too small in pixels, or pointing mostly at the camera (quality gate) -
    # in that zone tiny finger motion would map to a huge, unstable rotation.
    ev = np.zeros(3); ev[rot_axis] = 1.0
    tau = np.cross(ev, p)                            # true 3D tangent
    tw  = Rg @ (tau * S)
    ts  = np.array([tw[0] * FOCAL, -tw[1] * FOCAL])  # note screen-y flip
    m    = float(np.linalg.norm(ts))
    full = float(np.linalg.norm(tau)) * S * FOCAL    # unforeshortened size
    if m < MIN_TANGENT or full < 1e-9 or (m / full) < TANGENT_QUALITY:
        return None
    return ts / m, m

def layer_candidates(p, face_axis, cell, Rg, S):
    # For each twistable layer through this point, the screen tangent
    # tau = e_axis x p tells us where that layer visually moves for
    # +rotation - this is what turns a screen-space finger direction into
    # a specific, orientation-correct axis + sign, with zero hard-coding.
    out = []
    for v in range(3):
        if v == face_axis:
            continue
        t = tangent_screen(v, p, Rg, S)
        if t is not None:
            out.append({"axis": v, "layer": int(cell[v]), "that": t[0], "tmag": t[1]})
    return out

def facelet_quad_screen(face_axis, sgn, cell, Rg, S, lift=0.05):
    # Rounded tile-shaped ring for a specific cell, lifted off the surface.
    n = np.zeros(3); n[face_axis] = float(sgn)
    t1 = np.zeros(3); t1[(face_axis + 1) % 3] = 1.0
    t2 = np.cross(n, t1)
    ctr = np.array(cell, dtype=float)
    ctr[face_axis] = sgn * (1.0 + CUBIE_H + lift)
    pts = np.array([ctr + u * t1 + v * t2 for u, v in _T2D_HL])
    w = (pts * S) @ Rg.T
    return np.array(project(w), np.int32)

def cube_point_screen(p, Rg, S):
    return project([(p * S) @ Rg.T])[0]

#  Active face + the analytic box -> face mapping (NO ray-casting: the whole
#  point of this design is that the fingertip never has to overlap the
#  rendered cube, so we never intersect a screen ray with anything).

def active_face(Rg):
    # "The currently visible/active face" = whichever face is most square-on
    # to the camera right now. Re-evaluated every frame while just aiming,
    # so rotating the cube with the left hand changes what the box targets.
    best, best_z = (2, 1), -1e9
    for axis in range(3):
        for sgn in (-1, 1):
            z = (Rg @ (sgn * np.eye(3)[axis]))[2]
            if z > best_z:
                best_z, best = z, (axis, sgn)
    return best

def face_screen_basis(face_axis, Rg, S):
    # Which of the face's two in-plane world axes reads as "box-horizontal"
    # vs "box-vertical" on THIS frame's screen, and which sign of each reads
    # as "box-right"/"box-down" - this is the piece that keeps finger-up
    # meaning the same thing regardless of how the cube is currently turned.
    nonface = [a for a in range(3) if a != face_axis]
    delta = {}
    for a in nonface:
        e = np.zeros(3); e[a] = 1.0
        w = Rg @ (e * S)
        delta[a] = (w[0] * FOCAL, -w[1] * FOCAL)      # pixel-space (x, y-down)
    a0, a1 = nonface
    if abs(delta[a0][0]) >= abs(delta[a1][0]):
        u_axis, v_axis = a0, a1
    else:
        u_axis, v_axis = a1, a0
    u_sign = 1.0 if delta[u_axis][0] >= 0 else -1.0
    v_sign = 1.0 if delta[v_axis][1] >= 0 else -1.0
    return u_axis, u_sign, v_axis, v_sign

def box_axis_coord(u):
    # Normalized box position [0,1] -> face-local coordinate [-1,1] in tile-
    # center units, split into even thirds so the three tiles get equal
    # "reach" inside the box (u=1/3 and 2/3 land exactly on tile edges).
    return float(np.clip(3.0 * (u - 0.5), -1.0, 1.0))

def box_to_face(u, v, face_axis, face_sign, Rg, S):
    # The whole "virtual fingertip mapped to the cube" step, in one function:
    # normalized box coords -> a 3D point on the active face -> its cell.
    u_axis, u_sign, v_axis, v_sign = face_screen_basis(face_axis, Rg, S)
    au, av = box_axis_coord(u), box_axis_coord(v)
    p = np.zeros(3)
    p[face_axis] = face_sign * SURF_HALF
    p[u_axis] = au * u_sign
    p[v_axis] = av * v_sign
    cell = np.zeros(3, dtype=int)
    cell[face_axis] = face_sign
    cell[u_axis] = int(np.clip(round(au * u_sign), -1, 1))
    cell[v_axis] = int(np.clip(round(av * v_sign), -1, 1))
    return p, cell

#  Hand tracking: identity by wrist proximity, EMA smoothing, open/fist and
#  pinch classification

def lm_vec(lm, i):
    return np.array([lm[i].x, lm[i].y, lm[i].z])

def is_hand_open(lm):
    wrist = lm_vec(lm, 0)
    scale = np.linalg.norm(lm_vec(lm, 17) - wrist) + 1e-6
    avg_d = np.mean([np.linalg.norm(lm_vec(lm, t) - wrist) for t in (4, 8, 12, 16, 20)])
    return (avg_d / scale) > ORBIT_OPEN_RATIO

def fingertip_centroid(lm):
    xs = [lm[i].x * W_FRAME for i in (8, 12, 16, 20)]
    ys = [lm[i].y * H_FRAME for i in (8, 12, 16, 20)]
    return np.array([np.mean(xs), np.mean(ys)])

class TrackedHand:
    def __init__(self, hid, wrist_px, lm):
        self.id = hid
        self.wrist_ema = np.array(wrist_px, dtype=float)  # ID + smoothing only
        self.lm = lm
        self.idx_raw = None
        self.idx_ema = None      # smoothed index-fingertip pixel pos - the
                                 # single control point the whole touchpad
                                 # pipeline is driven from
        self.centroid = None     # 4-fingertip centroid - orbit driver
        self.open = False        # is_hand_open() classification
        self.pinch_norm = 1.0    # thumb-index distance / palm scale
        self.pinch_down = False  # hysteresis-smoothed pinch classification
        self.raw_side = None     # "L"/"R" from MediaPipe (sticky, pre-swap)
        self.missing = 0

def eff_side(h):
    # User-facing side after the runtime H-key swap.
    if h.raw_side is None:
        return None
    if not HAND_SWAP:
        return h.raw_side
    return "L" if h.raw_side == "R" else "R"

class HandTracker:
    # MediaPipe gives anonymous hands each frame; matching by WRIST proximity
    # (not any specific gesture) keeps identity valid no matter what the
    # fingers are doing - this version never needs a pinch at all.
    def __init__(self):
        self.hands = []
        self._next = 1

    def update(self, lms, labels=None):
        labels = labels or []
        obs = []
        for i, lm in enumerate(lms):
            wp = np.array([lm[0].x * W_FRAME, lm[0].y * H_FRAME])
            obs.append((wp, lm, labels[i] if i < len(labels) else None))
        used = set()
        for h in self.hands:
            best, bd = None, 1e9
            for j, (wp, lm, lab) in enumerate(obs):
                if j in used:
                    continue
                d = float(np.linalg.norm(wp - h.wrist_ema))
                if d < bd:
                    best, bd = j, d
            if best is not None and bd < HAND_MATCH_PX:
                used.add(best)
                wp, lm, lab = obs[best]
                h.wrist_ema += WRIST_EMA * (wp - h.wrist_ema)
                h.lm = lm
                h.missing = 0
                self._label(h, lab)
                self._features(h)
            else:
                h.missing += 1
        self.hands = [h for h in self.hands if h.missing <= HAND_DROP_FRAMES]
        for j, (wp, lm, lab) in enumerate(obs):
            if j in used:
                continue
            h = TrackedHand(self._next, wp, lm)
            self._next += 1
            self._label(h, lab)
            self._features(h)
            self.hands.append(h)

    def _label(self, h, lab):
        # Handedness labels flicker; only let a CONFIDENT label flip the side.
        if lab is None:
            return
        name, score = lab
        s = "R" if name == "Right" else "L"
        if h.raw_side is None or s == h.raw_side or score >= 0.75:
            h.raw_side = s

    def _features(self, h):
        lm = h.lm
        h.open = is_hand_open(lm)
        h.centroid = fingertip_centroid(lm)
        ip = np.array([lm[8].x * W_FRAME, lm[8].y * H_FRAME])
        h.idx_raw = ip
        if h.idx_ema is None:
            h.idx_ema = ip.copy()
        else:
            h.idx_ema = h.idx_ema + IDX_EMA * (ip - h.idx_ema)
        wrist = lm_vec(lm, 0)
        scale = np.linalg.norm(lm_vec(lm, 17) - wrist) + 1e-6
        d = np.linalg.norm(np.array([lm[4].x - lm[8].x, lm[4].y - lm[8].y]))
        h.pinch_norm = float(d / scale)
        if h.pinch_down:
            h.pinch_down = h.pinch_norm < PINCH_OFF     # hysteresis: sticky
        else:
            h.pinch_down = h.pinch_norm < PINCH_ON

#  Touchpad controller: the whole interaction is this state machine.
#  Selection is a held CLUTCH: pinch DOWN selects+locks whatever's under
#  the cursor; the finger's live position from that instant directly
#  steers the layer (continuous arc-following, re-linearized every frame -
#  stall-zone-safe); pinch UP deselects - and if a turn was mid-flight,
#  releases it into a spring to the nearest quarter turn via the settle anim.

class TouchpadController:
    def __init__(self):
        self.state = "AIMING"       # AIMING | TILE_SELECTED | MOVING_LAYER | COOLDOWN
        self._inside = False
        self._pinch_prev = False    # edge detector for pinch down/up
        self._settling = False      # a release/auto-commit glide is in flight
        self._pending_commit = False
        self.trail = deque(maxlen=TRAIL_LEN)
        self.hover = None           # (face_axis, sign, cell) while aiming
        self.cursor_pt = None       # 3D point under the cursor while aiming
        self.face_axis = self.face_sign = None
        self.cell = None
        self.horiz_cand = self.vert_cand = None
        self.chosen = None          # candidate actually being dragged
        self.anchor = None          # idx_ema at lock time (axis dead zone)
        self.grab_point = None      # 3D point at lock time (arc re-linearize)
        self.last_ptr = None        # idx_ema last frame (incremental drag)
        self.theta = 0.0            # live drag angle (deg)
        self.cool_until = 0.0

    def display_state(self, hand_present):
        if self.state == "AIMING":
            return "AIMING" if (hand_present and self._inside) else "IDLE"
        return self.state

    def freeze_orbit(self):
        # Real-cube behavior: the whole cube holds still from the instant a
        # tile locks through the end of its cooldown (settling included) -
        # you don't spin the puzzle while your other hand is mid-turn.
        return self.state in ("TILE_SELECTED", "MOVING_LAYER", "COOLDOWN")

    def force_idle(self):
        # Safety net for external state changes (scramble/reset) that would
        # otherwise leave a stale lock or live drag pointing at cubies that
        # just moved out from under it.
        self.state = "AIMING"
        self._inside = False
        self._settling = False
        self._pending_commit = False
        self.trail.clear()
        self.hover = self.cursor_pt = None
        self.face_axis = self.face_sign = self.cell = None
        self.horiz_cand = self.vert_cand = self.chosen = None
        self.anchor = self.grab_point = self.last_ptr = None
        self.theta = 0.0

    def _box_hysteresis(self, px, py):
        in_core = BOX_X0 <= px <= BOX_X1 and BOX_Y0 <= py <= BOX_Y1
        if self._inside:
            out = (px < BOX_X0 - BOX_EXIT_MARGIN or px > BOX_X1 + BOX_EXIT_MARGIN or
                   py < BOX_Y0 - BOX_EXIT_MARGIN or py > BOX_Y1 + BOX_EXIT_MARGIN)
            self._inside = not out
        else:
            self._inside = in_core
        return self._inside

    def _clear_selection(self):
        self.face_axis = self.face_sign = self.cell = None
        self.horiz_cand = self.vert_cand = self.chosen = None
        self.anchor = self.grab_point = self.last_ptr = None
        self.theta = 0.0

    def update(self, pad_hand, Rg, S, t_now, can_start):
        self.hover = None
        self.cursor_pt = None

        # Advance the pinch edge-detector unconditionally, every frame, in
        # every state, so a pinch held across a state change can't
        # masquerade as a fresh edge later.
        pad_pinch = bool(pad_hand.pinch_down) if pad_hand is not None else False
        pinch_falling = (not pad_pinch) and self._pinch_prev
        pinch_rising  = pad_pinch and not self._pinch_prev
        self._pinch_prev = pad_pinch

        if self._settling:
            # A release or auto-commit glide is running - hands-off until it
            # finishes; update_settle() (main loop) owns the animation itself.
            if anim is None:
                self.state = "COOLDOWN" if self._pending_commit else "AIMING"
                if self.state == "COOLDOWN":
                    self.cool_until = t_now + COOLDOWN_T
                self._settling = False
                self._clear_selection()
            return

        if self.state == "COOLDOWN":
            if t_now >= self.cool_until:
                self.state = "AIMING"
            return

        if self.state == "TILE_SELECTED":
            if pad_hand is None or pinch_falling:
                # Hand vanished or pinch released before any direction was
                # given - nothing ever animated, so just let go, cleanly.
                self.state = "AIMING"
                self._clear_selection()
                return
            disp = pad_hand.idx_ema - self.anchor
            if float(np.linalg.norm(disp)) >= AXIS_PICK_PX and can_start:
                cand = self.horiz_cand if abs(disp[0]) >= abs(disp[1]) else self.vert_cand
                begin_drag(cand["axis"], cand["layer"])
                self.chosen = cand
                self.theta = 0.0
                self.last_ptr = pad_hand.idx_ema.copy()
                self.state = "MOVING_LAYER"
            return

        if self.state == "MOVING_LAYER":
            if pad_hand is None or pinch_falling:
                # Let go mid-turn: spring to whichever quarter turn is closer.
                target = 90.0 * (1 if self.theta > 0 else -1) if abs(self.theta) >= COMMIT_DEG else 0.0
                self._pending_commit = (target != 0.0)
                settle_from_drag(target)
                self._settling = True
                return
            c = self.chosen
            dstep = pad_hand.idx_ema - self.last_ptr
            self.last_ptr = pad_hand.idx_ema.copy()
            # Arc following: re-linearize every frame at the CURRENT layer
            # angle, so the sticker chases the finger around its true curved
            # path - a tangent frozen at grab time inverts once the arc bends
            # away from it at foreshortened grabs.
            p_now = ROT[c["axis"]](self.theta) @ self.grab_point
            t = tangent_screen(c["axis"], p_now, Rg, S)
            if t is not None:
                that, tmag = t
            else:
                # Stall zone: the grabbed point's own screen motion has
                # foreshortened to ~nothing - fall back to the lock-time
                # tangent so the drag keeps its established direction.
                that, tmag = c["that"], c["tmag"]
            self.theta += float(np.degrees(DRAG_GAIN * float(dstep @ that) / tmag))
            self.theta = float(np.clip(self.theta, -ANGLE_CLAMP, ANGLE_CLAMP))
            if anim is not None and anim.get("mode") == "drag":
                anim["angle"] = self.theta
                if abs(self.theta) >= AUTO_COMMIT_DEG:
                    self._pending_commit = True
                    settle_from_drag(90.0 * (1 if self.theta > 0 else -1))
                    self._settling = True
            return

        # ---- AIMING ----
        if pad_hand is None:
            self.trail.clear()
            return
        px, py = pad_hand.idx_ema
        if not self._box_hysteresis(px, py):
            self.trail.clear()
            return

        self.trail.append(pad_hand.idx_ema.copy())
        face_axis, face_sign = active_face(Rg)
        u = float(np.clip((px - BOX_X0) / BOX_W, 0.0, 1.0))
        v = float(np.clip((py - BOX_Y0) / BOX_H, 0.0, 1.0))
        p, cell = box_to_face(u, v, face_axis, face_sign, Rg, S)
        self.hover = (face_axis, face_sign, cell)
        self.cursor_pt = p

        if pinch_rising and can_start:
            cands = layer_candidates(p, face_axis, cell, Rg, S)
            if not cands:
                # Too edge-on to turn anything from here - don't lock;
                # let the user re-orient (left hand) and try again.
                return
            if len(cands) == 2:
                c0, c1 = cands
                if abs(c0["that"][0]) >= abs(c1["that"][0]):
                    self.horiz_cand, self.vert_cand = c0, c1
                else:
                    self.horiz_cand, self.vert_cand = c1, c0
            else:
                self.horiz_cand = self.vert_cand = cands[0]
            self.face_axis, self.face_sign, self.cell = face_axis, face_sign, cell
            self.anchor = pad_hand.idx_ema.copy()
            self.grab_point = p.copy()
            self.state = "TILE_SELECTED"
            self.trail.clear()

#  Single-hand orbit: LEFT hand only, open-palm/fist gate. The RIGHT hand
#  (the touchpad hand) never contributes to whole-cube rotation.

DEG_PER_PX_BASE = 0.55
DEG_PER_PX_MAX  = 2.2
SPEED_SCALE     = 0.08
DEADZONE_PX     = 3.0
INERTIA         = 0.88

spin_x = spin_y = 0.0
_prev_orbit_pt  = None
_orbit_id       = None

def orbit_update(hands):
    global spin_x, spin_y, _prev_orbit_pt, _orbit_id
    driver = next((h for h in hands if eff_side(h) == "L"), None)
    if driver is None:
        spin_x *= 0.94; spin_y *= 0.94
        _prev_orbit_pt = None; _orbit_id = None
        return
    if not driver.open:
        # Fist: immediately stop responding and hold this exact orientation.
        spin_x = spin_y = 0.0
        _prev_orbit_pt = None
        return
    if _orbit_id != driver.id:
        _prev_orbit_pt = None
        _orbit_id = driver.id
    pt = driver.centroid
    if _prev_orbit_pt is not None and \
       float(np.linalg.norm(pt - _prev_orbit_pt)) > HAND_JUMP_PX:
        _prev_orbit_pt = None
    if _prev_orbit_pt is not None:
        dx, dy = pt[0] - _prev_orbit_pt[0], pt[1] - _prev_orbit_pt[1]
        if abs(dx) < DEADZONE_PX: dx = 0.0
        if abs(dy) < DEADZONE_PX: dy = 0.0
        raw_speed = (dx**2 + dy**2) ** 0.5
        dpp = np.clip(DEG_PER_PX_BASE + raw_speed * SPEED_SCALE,
                      DEG_PER_PX_BASE, DEG_PER_PX_MAX) * ROTATION_SLOWDOWN
        spin_y = spin_y * INERTIA + (dx * dpp) * (1.0 - INERTIA)
        spin_x = spin_x * INERTIA + (dy * dpp) * (1.0 - INERTIA)
    _prev_orbit_pt = pt.copy()

def orbit_freeze():
    global spin_x, spin_y, _prev_orbit_pt
    spin_x = spin_y = 0.0
    _prev_orbit_pt = None

#  Drawing
def draw_hand(frame, lm, label, col, centroid):
    pts = [(int(l.x * W_FRAME), int(l.y * H_FRAME)) for l in lm]
    dot_col = tuple(min(255, int(v * 1.35)) for v in col)

    # Hand skeleton
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], col, 2, cv2.LINE_AA)

    # Hand landmarks - no numbers
    for px, py in pts:
        cv2.circle(frame, (px, py), 6, col, 1)
        cv2.circle(frame, (px, py), 4, dot_col, -1)

    # Fingertip highlights
    for ti in (8, 12, 16, 20):
        px, py = pts[ti]
        cv2.circle(frame, (px, py), 10, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3, (0, 255, 200), -1, cv2.LINE_AA)

    # Centroid marker
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.drawMarker(
        frame,
        (cx, cy),
        (0, 255, 180),
        cv2.MARKER_CROSS,
        20,
        2,
        cv2.LINE_AA
    )

def draw_hand_lite(frame, lm, col):
    pts = [(int(l.x*W_FRAME), int(l.y*H_FRAME)) for l in lm]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], col, 1, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 2, col, -1)

_STATE_COLORS = {
    "IDLE":           ( 90,  90,  90),
    "AIMING":         ( 60, 190, 255),
    "TILE_SELECTED":  (120, 255, 120),
    "MOVING_LAYER":   (255, 190,  60),
    "COOLDOWN":       (120, 120, 255),
}

def draw_gesture_box(frame, ctrl, pad_hand, label):
    col = _STATE_COLORS.get(label, (150, 150, 150))
    cv2.rectangle(frame, (BOX_X0, BOX_Y0), (BOX_X1, BOX_Y1), col, 2, cv2.LINE_AA)
    cv2.putText(frame, "GESTURE CONTROL", (BOX_X0, BOX_Y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    pts = list(ctrl.trail)
    for i in range(1, len(pts)):
        a = i / max(len(pts) - 1, 1)
        fade = tuple(int(v * (0.25 + 0.75 * a)) for v in col)
        p0 = tuple(pts[i-1].astype(int)); p1 = tuple(pts[i].astype(int))
        cv2.line(frame, p0, p1, fade, 1, cv2.LINE_AA)

    if pad_hand is not None and pad_hand.idx_ema is not None:
        ix, iy = pad_hand.idx_ema.astype(int)
        cv2.circle(frame, (ix, iy), 6, col, -1, cv2.LINE_AA)
        # Pinch meter: shrinks as thumb and index close - the select/
        # unselect trigger, so it's worth showing how close you are.
        pr = int(9 + 16 * min(pad_hand.pinch_norm, 1.0))
        pcol = (120, 255, 120) if pad_hand.pinch_down else col
        cv2.circle(frame, (ix, iy), pr, pcol, 1, cv2.LINE_AA)

        if ctrl.state == "TILE_SELECTED" and ctrl.anchor is not None:
            axp, ayp = ctrl.anchor.astype(int)
            cv2.arrowedLine(frame, (axp, ayp), (ix, iy), (120, 255, 120),
                            2, cv2.LINE_AA, tipLength=0.35)
            disp = pad_hand.idx_ema - ctrl.anchor
            prog = min(float(np.linalg.norm(disp)) / AXIS_PICK_PX, 1.0)
            bx0, by0 = BOX_X0, BOX_Y1 + 8
            cv2.rectangle(frame, (bx0, by0), (bx0 + BOX_W, by0 + 6), (70, 70, 70), 1)
            cv2.rectangle(frame, (bx0, by0), (bx0 + int(BOX_W * prog), by0 + 6),
                         (120, 255, 120), -1)
        elif ctrl.state == "MOVING_LAYER" and not ctrl._settling:
            cv2.putText(frame, f"{ctrl.theta:+.0f} deg", (ix + 12, iy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 120), 1, cv2.LINE_AA)

    status = label
    if label == "COOLDOWN":
        status = f"COOLDOWN {max(ctrl.cool_until - time.time(), 0.0):.1f}s"
    elif label == "TILE_SELECTED":
        status = "LOCKED - move to turn, release to deselect"
    elif label == "MOVING_LAYER":
        status = "releasing..." if ctrl._settling else "TURNING - release to let go"
    cv2.putText(frame, status, (BOX_X0, BOX_Y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

# ============================================================================
#  Runtime  (guarded so the math above is importable for headless tests)
# ============================================================================
if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.getenv(
        "HAND_LANDMARKER_MODEL",
        os.path.join(SCRIPT_DIR, "hand_landmarker.task"),
    )
    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}"); sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6))

    def open_camera():
        for idx in range(5):
            for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened(): cap.release(); continue
                time.sleep(1.5)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(20): cap.read()
                ok, frm = cap.read()
                if ok and frm is not None and frm.size > 0:
                    print(f"OK: Camera index={idx}")
                    return cap
                cap.release()
        return None

    print("Finding camera...")
    cap = open_camera()
    if cap is None:
        print("No camera found"); sys.exit(1)

    tracker = HandTracker()
    ctrl    = TouchpadController()

    ax, ay, az    = 15.0, 25.0, 0.0
    half_edge     = 5.0
    fail_cnt      = 0
    prev_solved     = True
    celebrate_until = 0.0
    fps, t_prev   = 30.0, time.time()

    print("LEFT hand : open palm = rotate & revolve the cube | fist = freeze")
    print("            (17% slower than before, same inertial feel)")
    print("RIGHT hand: point your index finger into the GESTURE CONTROL box")
    print("            (bottom-right). PINCH to select and lock the targeted")
    print("            tile. While pinched, moving your fingers directly")
    print("            steers that layer - left/right/up/down turns it the")
    print("            same way it looks like it should. Release to let go:")
    print("            past the halfway point it springs to the nearest")
    print("            quarter turn, short of it it springs back to zero.")
    print(f"1.5s cooldown after every completed turn. H swaps hands if")
    print("labels are backwards | -/= resize | RLUDFB keys still work")
    print("S scramble | Z reset | Q quit\n")

    # cv2.getWindowImageRect proved unreliable for this - on macOS/Qt it can
    # report the window's size before the WM has actually finished resizing
    # it to fullscreen, so any canvas built from it ends up smaller than the
    # real window and leaves a stray gap of raw window background around the
    # video (that's the white band). Ask the OS for the real screen
    # resolution directly instead - it doesn't depend on OpenCV's window
    # state or its timing at all.
    def get_screen_size():
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        return 1920, 1080  # fallback if tkinter isn't available

    SCREEN_W, SCREEN_H = get_screen_size()
    # tkinter's reported screen size and the cv2 window's actual pixel size
    # don't always agree exactly (Retina/display-scaling rounding) - render
    # a few px larger than we think the screen is so the frame always
    # slightly overflows the real window instead of matching it exactly.
    # OpenCV clips whatever doesn't fit; there's no longer any way for a
    # measurement mismatch to leave a gap on any edge.
    OVERSCAN_PX = 8
    RENDER_W, RENDER_H = SCREEN_W + OVERSCAN_PX, SCREEN_H + OVERSCAN_PX

    WINDOW_NAME = "Gesture Cube"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def fit_to_screen(img, sw, sh):
        # Scale to COVER the full screen (no black bars) with NO stretching/
        # distortion - aspect ratio is preserved, we just scale up until the
        # shorter screen dimension is matched, then center-crop whatever
        # overhangs on the other axis. Trade-off vs. letterboxing: nothing is
        # ever squashed, but the very left/right (or top/bottom) edges of the
        # camera frame get cropped off-screen.
        #
        # nw/nh are pushed up with ceil (never floor/round) and then clamped
        # to be at least sw/sh - float rounding in the scale math can
        # otherwise land the resized image a pixel short of the screen, and
        # the crop below would then come up short too, leaving a hairline of
        # raw window background (white) down the edge instead of video.
        fh, fw = img.shape[:2]
        scale = max(sw / fw, sh / fh)
        nw = max(int(np.ceil(fw * scale)), sw)
        nh = max(int(np.ceil(fh * scale)), sh)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = max((nw - sw) // 2, 0)
        y0 = max((nh - sh) // 2, 0)
        return resized[y0:y0 + sh, x0:x0 + sw]

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            fail_cnt += 1
            time.sleep(0.03)
            if fail_cnt > 80:
                cap.release(); cap = open_camera()
                if cap is None: break
                fail_cnt = 0
            continue
        fail_cnt = 0
        frame = cv2.flip(frame, 1)

        try:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        except Exception:
            result = None

        lms, labs = [], []
        if result and result.hand_landmarks:
            lms = result.hand_landmarks
            for cats in (result.handedness or []):
                labs.append((cats[0].category_name, cats[0].score) if cats else None)
        tracker.update(lms, labs)
        hand_detected = len(tracker.hands) > 0

        pad_hand = next((h for h in tracker.hands if eff_side(h) == "R"), None)

        # --- Advance any in-flight settle glide FIRST, so the controller
        # --- below sees this frame's FINAL anim state (not last frame's) -
        # --- otherwise resolving a just-finished settle lags by one frame.
        update_settle()

        # --- Gesture layer: aim / pinch-lock / continuous direct-turn on the
        # --- box (RIGHT hand runs the touchpad)
        pick_R = R_mat(ax, ay, az)
        S_now  = half_edge / 1.5
        ctrl.update(pad_hand, pick_R, S_now, time.time(), can_start=(anim is None))
        freeze = ctrl.freeze_orbit()

        # --- Navigation: LEFT hand only, open/fist gated (right hand never
        # --- rotates the whole cube)
        if freeze:
            orbit_freeze()
        else:
            orbit_update(tracker.hands)

        ay = (ay + spin_y) % 360
        ax = (ax + spin_x) % 360
        az = ay * 0.08        # cinematic roll - set factor to 0.0 for a steadier view

        # --- Cube state machine: auto anim (keyboard/queue moves) is handled
        # --- below; drag/settle anims are driven by ctrl.update() above and
        # --- the update_settle() call earlier this frame.
        if anim is not None and anim.get("mode", "auto") == "auto" \
           and time.time() - anim["t0"] >= TURN_TIME:
            finish_move()
        if anim is None and move_queue:
            start_move(*move_queue.pop(0))

        R = R_mat(ax, ay, az)
        draw_puzzle(frame, R, half_edge, anim)

        # --- Cube-side overlays: hover ring while aiming, locked ring while
        # --- selected/moving/cooling down, plus the analytic cursor dot.
        if ctrl.hover is not None:
            q = facelet_quad_screen(*ctrl.hover, R, S_now)
            cv2.polylines(frame, [q], True, (110, 60, 0), 5, cv2.LINE_AA)
            cv2.polylines(frame, [q], True, (60, 190, 255), 2, cv2.LINE_AA)
        if ctrl.cursor_pt is not None:
            cx_, cy_ = cube_point_screen(ctrl.cursor_pt, R, S_now)
            cv2.circle(frame, (cx_, cy_), 5, (60, 190, 255), -1, cv2.LINE_AA)
        if ctrl.state in ("TILE_SELECTED", "MOVING_LAYER", "COOLDOWN") and ctrl.cell is not None:
            q = facelet_quad_screen(ctrl.face_axis, ctrl.face_sign, ctrl.cell, R, S_now)
            cv2.polylines(frame, [q], True, (0, 90, 0), 5, cv2.LINE_AA)
            cv2.polylines(frame, [q], True, (120, 255, 120), 2, cv2.LINE_AA)

        # --- Gesture box widget (always drawn - decoupled from the cube)
        disp_label = ctrl.display_state(pad_hand is not None)
        draw_gesture_box(frame, ctrl, pad_hand, disp_label)

        # --- Hand overlays: full panel for LEFT (rotates), lite for RIGHT
        # --- (touchpad)
        orbit_hand = next((h for h in tracker.hands if eff_side(h) == "L"), None)
        if orbit_hand is not None:
            ltxt = "LEFT Â· open = rotate" if orbit_hand.open else "LEFT Â· FIST = frozen"
            lcol = (0, 220, 90) if orbit_hand.open else (60, 60, 200)
            draw_hand(frame, orbit_hand.lm, ltxt, lcol, orbit_hand.centroid)
        if pad_hand is not None:
            draw_hand_lite(frame, pad_hand.lm, (60, 190, 255))
        for h in tracker.hands:
            s = eff_side(h)
            wx, wy = int(h.lm[0].x * W_FRAME), int(h.lm[0].y * H_FRAME)
            tag = {"L": "L-rotate", "R": "R-touchpad", None: "?"}[s]
            tcol = (0, 220, 90) if s == "L" else (60, 190, 255)
            cv2.putText(frame, tag, (wx - 26, wy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, tcol, 1, cv2.LINE_AA)

        # --- Solve detection + celebration
        solved_now = (anim is None) and (not move_queue) and is_solved()
        if solved_now and not prev_solved and move_count > 0:
            celebrate_until = time.time() + 2.6
        prev_solved = solved_now
        if time.time() < celebrate_until:
            pulse = int(180 + 75 * np.sin(time.time() * 9))
            cv2.putText(frame, "SOLVED!", (235, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (60, pulse, 60), 3, cv2.LINE_AA)

        # --- HUD
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-3))
        t_prev = t_now

        hx = 200
        status = "HAND" if hand_detected else "no hand"
        cv2.putText(frame, f"[{status}]  Y:{spin_y:+.1f}  X:{spin_x:+.1f}  "
                           f"size:{half_edge:.1f}  fps:{fps:.0f}  "
                           f"hands:{'SWAPPED' if HAND_SWAP else 'std'}",
                    (hx, H_FRAME-38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"last:{last_move}  queue:{len(move_queue)}  moves:{move_count}  "
                           f"state:{'SOLVED' if solved_now else 'mixed'}  "
                           f"gesture:{disp_label}",
                    (hx, H_FRAME-22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1, cv2.LINE_AA)
        cv2.putText(frame, "pinch+hold = lock+turn | release = let go | H swap | RLUDFB | -/= | S | Z | Q",
                    (hx, H_FRAME-6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110,110,110), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, fit_to_screen(frame, RENDER_W, RENDER_H))

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')):
            break
        elif k in (ord('h'), ord('H')):
            HAND_SWAP = not HAND_SWAP
        elif k in (ord('s'), ord('S')):
            scramble()
            ctrl.force_idle()
        elif k in (ord('z'), ord('Z')):
            reset_cube()
            ctrl.force_idle()
        elif k in (ord('-'), ord('_')):
            half_edge = max(3.5, half_edge - 0.5)
        elif k in (ord('='), ord('+')):
            half_edge = min(8.5, half_edge + 0.5)
        elif 0 < k < 128:
            ch  = chr(k)
            low = ch.lower()
            if low in MOVES and len(move_queue) < 20:
                axis, layer, sign = MOVES[low]
                if ch.isupper():
                    sign = -sign
                move_queue.append((axis, layer, sign))

    cap.release()
    cv2.destroyAllWindows()