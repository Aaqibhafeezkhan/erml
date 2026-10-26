# Edge cases and interpreting predictions

ERML returns one FacePrediction for each face detected in an image. The result describes a model's facial-expression classification, not a person's internal emotional state.

## No face detected

If YuNet does not detect a face, EmotionDetector.analyze() returns an empty list ([]). It does not return a neutral prediction or a zero-confidence placeholder. Check for an empty result before indexing or iterating:

~~~python
from erml import EmotionDetector

detector = EmotionDetector()
results = detector.analyze("photo.jpg")

if not results:
    print("No faces detected. Try a clearer image with a visible face.")
else:
    for face in results:
        print(face.emotion, face.confidence)
~~~

An empty list can mean no face was found at the configured detector threshold; it does not prove that the image contains no face. Image quality, pose, occlusion, lighting, and face size can affect detection.

## Multiple faces detected

The returned list contains one prediction per detected face. Each prediction includes a bounding box in pixel coordinates. Process every result rather than assuming there is exactly one face:

~~~python
results = detector.analyze("group-photo.jpg")

for index, face in enumerate(results, start=1):
    print(index, face.bbox, face.emotion, face.confidence)
~~~

Do not use list position as a person's identity, and do not assume detection order is stable between images or runs.

## Low-confidence predictions

confidence is the score for the top predicted class. It is not a guarantee that the prediction is correct, and it should not be treated as a calibrated probability that a person feels that emotion. Choose a review threshold for your application and validate it against suitable data. The value below (0.60) is illustrative only, not a recommended or validated ERML threshold.

~~~python
REVIEW_THRESHOLD = 0.60  # Example only; validate for your use case.

for face in detector.analyze("photo.jpg"):
    if face.confidence < REVIEW_THRESHOLD:
        print("Review prediction:", face.emotion, face.confidence)
        continue

    print("Candidate expression label:", face.emotion)
~~~

For a low-confidence result, preserve the scores and bounding box for review rather than silently converting it to neutral. If your workflow cannot tolerate uncertain results, abstain or ask for human review.

## Responsible interpretation

ERML classifies visible facial-expression patterns into its seven trained labels. It cannot establish what a person actually feels, intends, believes, or is experiencing. Avoid making consequential decisions about people based only on these predictions. Consider consent, privacy, bias, image conditions, and training-data limitations when designing an application.

## CPU inference benchmark

See the CPU inference benchmark section in the README and run python scripts/benchmark_cpu.py --help. Results depend on the CPU, operating system, dependency versions, image content, and runtime conditions; record measurements from the machine being evaluated rather than relying on numbers from another environment.
