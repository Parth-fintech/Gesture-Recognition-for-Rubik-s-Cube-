"""Headless smoke tests for gesture_cube.py - no camera or MediaPipe model
required. Everything the main loop touches (cv2 window, hand landmarker) is
guarded behind ``if __name__ == "__main__"``, so importing the script is
safe and exercises the puzzle/gesture math on its own.
"""

import random

import numpy as np
import pytest

import gesture_cube as gc


@pytest.fixture(autouse=True)
def fresh_cube():
    gc.reset_cube()
    yield
    gc.reset_cube()


def test_starts_solved():
    assert gc.is_solved()
    assert len(gc.cubies) == 26


def test_move_and_inverse_returns_to_solved():
    axis, layer, sign = gc.MOVES['u']
    gc.start_move(axis, layer, sign); gc.finish_move()
    gc.start_move(axis, layer, -sign); gc.finish_move()
    assert gc.is_solved()
    assert gc.move_count == 2


def test_four_quarter_turns_is_identity():
    axis, layer, sign = gc.MOVES['f']
    for _ in range(4):
        gc.start_move(axis, layer, sign); gc.finish_move()
    assert gc.is_solved()


def test_scramble_then_solve_via_recorded_inverse():
    random.seed(7)
    gc.scramble(12)
    applied = []
    while gc.move_queue:
        axis, layer, sign = gc.move_queue.pop(0)
        gc.start_move(axis, layer, sign); gc.finish_move()
        applied.append((axis, layer, sign))
    assert not gc.is_solved()
    for axis, layer, sign in reversed(applied):
        gc.start_move(axis, layer, -sign); gc.finish_move()
    assert gc.is_solved()


def test_drag_settle_forward_commits_move():
    gc.begin_drag(0, 1)
    gc.settle_from_drag(90.0)
    gc.anim["t0"] -= 10.0  # fast-forward the glide
    gc.update_settle()
    assert gc.anim is None
    assert gc.move_count == 1


def test_rot90_is_exact_and_order_four():
    for axis in range(3):
        M = gc.rot90_int(axis, 1)
        assert np.array_equal(np.linalg.matrix_power(M, 4), np.eye(3, dtype=int))


def test_draw_puzzle_renders_something():
    import cv2  # noqa: F401  (import here so the rest of the suite works without it)
    frame = np.zeros((gc.H_FRAME, gc.W_FRAME, 3), dtype=np.uint8)
    gc.draw_puzzle(frame, gc.R_mat(15, 25, 0), 5.0, None)
    assert (frame.sum(axis=2) > 0).sum() > 0


def test_touchpad_controller_starts_idle():
    ctrl = gc.TouchpadController()
    assert ctrl.state == "AIMING"
    assert ctrl.display_state(hand_present=False) == "IDLE"
    assert ctrl.freeze_orbit() is False


def test_active_face_picks_camera_facing_face():
    axis, sign = gc.active_face(gc.R_mat(0, 0, 0))
    assert (axis, sign) == (2, 1)
