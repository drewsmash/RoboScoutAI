# Potato mode — no-AI tracking fallback

When YOLO is missing, API keys are unset, or the machine is underpowered, RoboScoutAI
still tracks robots with **dumb OpenCV** methods. Nothing downloads a model. Nothing
calls the cloud.

## Modes

| Mode | What runs | Needs |
| --- | --- | --- |
| **potato** | Motion blobs + bumper color + optical flow | OpenCV only |
| **motion** | Background subtract + optical flow | OpenCV only |
| **color** | HSV red/blue bumpers + motion | OpenCV only |
| **auto** | YOLO / OpenAI / Gemini *if available*, else potato | Whatever is installed |

`auto` always appends the potato strategies, so empty paths from a missing model are
avoided by design.

## Browser fallback

If the server returns few or no samples (or you picked **Potato**), the web UI can run
a **JavaScript frame-diff** tracker on the match `<video>` element in the browser.
No neural net, no API key — only canvas pixel differencing.

Samples are posted to `POST /api/jobs/{id}/browser-tracks` so exports stay consistent.

## Accuracy note

Potato / browser tracks are approximate motion estimates. Confirm before pick lists.
For better boxes, install Ultralytics YOLO or set an OpenAI / Gemini key.
