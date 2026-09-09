# gesture-cube

A virtual Rubik's cube you never touch. Webcam + [MediaPipe](https://developers.google.com/mediapipe)
hand tracking drives it entirely by gesture: your left hand orbits the
whole puzzle, your right hand aims at a virtual touchpad, pinches to lock
a tile, and drags to turn the layer it belongs to- with the sticker
following your finger around its real curved path in real time.

![status](https://github.com/<Parth-fintech>/gesture-cube/actions/workflows/ci.yml/badge.svg)

## Controls

| Hand | Gesture | Action |
|---|---|---|
| Left | open palm, move | orbit the whole cube |
| Left | fist | freeze orbit |
| Right | point index finger into the **GESTURE CONTROL** box | aim / hover a tile |
| Right | pinch (thumb + index) | lock the hovered tile |
| Right | move while pinched | turn the locked layer — release past the halfway point to commit, short of it to spring back |

Keyboard fallback: `R L U D F B` (+ Shift for the reverse turn), `-`/`=` to
resize, `S` scramble, `Z` reset, `H` swap left/right hand labels if your
camera reports them mirrored, `Q` quit.

## Requirements

- Python 3.9+
- A webcam
- A MediaPipe hand-landmarker model file (`hand_landmarker.task`) — [download here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

## Setup

```bash
git clone https://github.com/<Parth-fintech>/gesture-cube.git
cd gesture-cube
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Grab the hand-tracking model and tell the app where it lives
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
cp .env.example .env   # defaults to ./hand_landmarker.task — edit if you put it elsewhere
```

## Run

```bash
python gesture_cube.py
```

## Testing

```bash
pip install pytest
pytest test_gesture_cube.py -v
```

`test_gesture_cube.py` covers the puzzle state machine (moves, inverses,
scramble/solve, drag-settle commit), the rotation math, the box→face→cell
gesture mapping, and rendering — all without a webcam. The camera loop
itself lives behind `if __name__ == "__main__":`, so importing the script
for tests never opens a window or touches hardware.

## Known trade-offs

- The video feed is scaled to cover the full screen rather than
  letterboxed, so the far left/right (or top/bottom) edge of the camera
  frame is cropped on aspect ratios that don't match your display.
- The touchpad's active face is re-evaluated every frame from camera
  orientation only (no ray-casting against the rendered cube), so a tile
  too edge-on to the camera won't lock — reorient with the left hand and
  try again.

## License

MIT — see [LICENSE](LICENSE).
