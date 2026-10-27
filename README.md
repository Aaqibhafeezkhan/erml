<div align="center">
  <img src="docs/assets/erml-banner.png" alt="ERML Banner" width="100%" />
</div>

##### A pip-installable Python SDK for facial emotion recognition. Drop it into your own project, pass it a frame, get back structured emotion data — no camera or display logic included.

```python
from erml import EmotionDetector, FacePrediction, format_results

detector = EmotionDetector()
results = detector.analyze("photo.jpg")

# Object access — full IDE autocomplete
print(results[0].emotion, results[0].confidence)

# Or export to plain dict
print(format_results(results))
```

```
Face 1  [x=85 y=67 w=259 h=259]
  Emotion    : Neutral
  Confidence : 42.9%
  All scores :
    neutral    42.9%  ████████
    sad        18.2%  ███
    fear       15.4%  ███
    angry      10.2%  ██
    happy       7.5%  █
    surprise    5.2%  █
    disgust     0.1%
```

## Install

```bash
pip install erml
```

Or from source:

```bash
git clone https://github.com/sid-lakhani/erml
cd erml
uv venv --python 3.12
source .venv/bin/activate.fish   # or: source .venv/bin/activate (bash/zsh)
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
```

> Requires Python 3.8–3.12. TensorFlow does not yet support Python 3.13+.
> To train, additionally install: `uv pip install -r requirements-train.txt`

## Usage

### Basic

```python
from erml import EmotionDetector

detector = EmotionDetector()
results = detector.analyze("photo.jpg")
```

### Input types

```python
import cv2
import numpy as np
from PIL import Image

# File path
results = detector.analyze("photo.jpg")

# OpenCV BGR array
frame = cv2.imread("photo.jpg")
results = detector.analyze(frame)

# PIL Image
pil_img = Image.open("photo.jpg")
results = detector.analyze(pil_img)
```

### Output

`analyze()` returns a list of dicts — one per detected face:

```python
[
  FacePrediction(
    emotion="happy",        # top predicted emotion
    confidence=0.87,        # float 0–1
    all={                   # scores for all 7 emotions
      "angry": 0.01,
      "disgust": 0.00,
      "fear": 0.02,
      "happy": 0.87,
      "sad": 0.03,
      "surprise": 0.05,
      "neutral": 0.02
    },
    bbox=BoundingBox(x=120, y=80, w=64, h=64)
  )
]
```

Access fields directly with full IDE autocomplete:
```python
result = results[0]
print(result.emotion)        # "happy"
print(result.bbox.x)         # 120
```

Or export to a plain dictionary (e.g. for JSON logging):
```python
raw = results[0].model_dump()
raw_list = [r.model_dump() for r in results]
```

Returns `[]` if no face is detected. Never raises on empty input.

### Batch processing

Process multiple images in a single call using `analyze_batch()`:

```python
frames = ["face1.jpg", "face2.jpg", "group.jpg"]
batch_results = detector.analyze_batch(frames)

for image_path, faces in zip(frames, batch_results):
    print(f"{image_path}: {len(faces)} faces detected")
    for face in faces:
        print(f"  {face.emotion} ({face.confidence:.2f})")
```

`analyze_batch()` batches ONNX emotion inference across detected face crops while preserving input image ordering. If an image contains no faces, its corresponding entry is `[]`.

### Edge cases and low-confidence predictions

See [Edge cases and interpreting predictions](docs/edge-cases.md) for no-face and multi-face behavior, a confidence-threshold example, and responsible interpretation of expression labels.


### Human-readable output

```python
from erml import EmotionDetector, format_results

detector = EmotionDetector()
print(format_results(detector.analyze("photo.jpg")))
```

## Emotions

`angry` · `disgust` · `fear` · `happy` · `sad` · `surprise` · `neutral`

Trained on [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).


## CPU inference benchmark

Run the benchmark on the machine you want to evaluate, using an image representative of your workload:

~~~bash
python scripts/benchmark_cpu.py tests/sample_faces/happy_face.jpg
~~~

By default, the script benchmarks 320×240, 640×480, and 1280×720 images with 5 warm-up runs and 30 timed runs per size. Adjust the image, sizes, and iteration counts to match your workload:

~~~bash
python scripts/benchmark_cpu.py path/to/image.jpg --iterations 50 --warmup 5 --sizes 320x240 640x480 1920x1080 --json benchmark-results.json
~~~

The script uses ONNX Runtime's CPU execution provider and limits OpenCV to one thread. It reports minimum, median, 95th-percentile, and maximum latency in milliseconds, along with CPU, OS, Python, OpenCV, and ONNX Runtime details. Timings cover analyze() (face detection plus inference for detected faces); model initialization and first-run downloads are excluded. Each input is resized to the target dimensions, so use a representative image and record its source dimensions and detected face count. Run on an otherwise idle machine. Results from different machines or images are not directly comparable.

## Training

To retrain from scratch, download FER-2013 and place it at `dataset/` organized by emotion subfolder, then:

```bash
python training/train.py
```

Trains for up to 20 epochs with early stopping, saves best weights to `erml/assets/erml_v1.h5`.

## Quick inference script

```bash
python examples/test_inference.py photo.jpg
```

## Webcam demo

```bash
python examples/webcam_demo.py
python examples/webcam_demo.py --camera 1  # alternate camera index
```

Press **Q** to quit. Draws bounding boxes and emotion labels in real time.

## Tests

```bash
pytest tests/ -v
```

All tests run fully headless — no camera, no display, no trained model required.

## Project structure

```
erml/
├── erml/
│   ├── __init__.py       # public API: EmotionDetector, FacePrediction, format_results
│   ├── constants.py      # EMOTION_LABELS and shared constants
│   ├── detector.py       # EmotionDetector class (ONNX & YuNet runtime)
│   ├── download.py       # atomic auto-downloader for ONNX weights
│   ├── preprocess.py     # face ROI preprocessing
│   ├── schemas.py        # Pydantic output models (FacePrediction, BoundingBox)
│   ├── utils.py          # format_results helper
│   └── assets/           # ONNX weights cached here automatically
├── training/
│   ├── model.py          # TF/Keras CNN architecture definition
│   └── train.py          # training script (requires: pip install erml[train])
├── tests/
├── scripts/
│   └── export_onnx.py    # utility to export trained Keras models to ONNX
├── examples/
└── pyproject.toml
```

## Real-world Applications

ERML is designed as a foundational **computer-vision perception SDK** for developers to plug into broader AI systems and tools.

| Application | Value |
| --- | --- |
| 🤖 **AI Agents** | Multimodal agents understanding facial signals |
| 🎮 **Games** | NPC/player emotion → adaptive gameplay |
| 🧑‍💻 **Video Calls** | Engagement, attention, and reaction signals |
| 🧪 **HCI Research** | Studying human-computer interaction |
| 🏥 **Research** | Controlled affective-computing experiments |
| 🎥 **Content Creation**| Automatic reaction/emotion metadata |
| 🛍️ **UX Research** | Aggregate reactions during usability testing |
| 🧠 **Robotics** | Robot perception of human affect |
| 🖥️ **Accessibility** | Alternative interaction signals |

*Note: Facial expression ≠ actual emotional state. ERML detects facial expressions to provide a structured signal for your downstream application.*

## Versioning

| Version | Status | Notes |
|---------|--------|-------|
| v0.1.0 | legacy | FER-2013, basic CNN, ~52% val accuracy |
| v1.0.1 | current | Stable public API, ONNX runtime, PyPI published |

## Roadmap

ERML is evolving from an emotion classifier into a comprehensive **human visual perception SDK**. 

- **Phase 1 (Current):** Stable facial expression recognition SDK with local ONNX inference.
- **Phase 2 (Planned):** Expand perception to include face landmarks, head pose, and gaze tracking.
- **Phase 3 (Planned):** Expose structured multi-modal signals natively designed for AI agent consumption.

## Contributing

ERML is an open-source project and we actively welcome contributions! Whether it's fixing a bug, adding a new feature (like webcam streaming support), or improving documentation, your help is appreciated.

1. Check the [Issues](https://github.com/sid-lakhani/erml/issues) tab for `good first issue` or `help wanted` tags.
2. Read our [Contributing Guide](CONTRIBUTING.md) for setup instructions.
3. Open a Pull Request!

## License

MIT
