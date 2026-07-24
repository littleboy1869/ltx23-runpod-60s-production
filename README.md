# RunPod LTX-2.3 60-second story worker

This repository generates one approximately 60-second vertical MP4 with native
LTX-2.3 audio. It renders six ten-second shots, conditions every shot after the
first on the exact retained boundary frame, removes the duplicate first frame
at each join, normalizes the combined audio, and encodes the result at
720x1280/24 fps.

## Why six ten-second shots

The tested 20-second output became almost motionless for several seconds and
eventually left the frame empty. Shorter shots give LTX-2.3 fewer timed events
to remember. Every production prompt is therefore split into five explicit
two-second beats. The handler also measures frame-to-frame movement for every
one-second window and retries only a weak segment.

This cannot mathematically guarantee that a generative model will follow every
semantic action or preserve identity perfectly. It does prevent silent
workflow failures, wrong dimensions, missing audio, missing frames, long
low-motion stretches that the detector can see, and broken final merges.

## Prompt handling

`TextGenerateLTX2Prompt` is intentionally not in the executed graph. The
request already contains carefully written timed prompts, so another generator
could rewrite dialogue or remove actions. The Gemma model is still required:
LTX-2.3 uses Gemma as its text encoder even when prompt enhancement is disabled.

## Repository files

- `Dockerfile`: pins the known-working ComfyUI/custom-node revisions and models.
- `handler.py`: six-shot orchestration, motion QA, continuation, audio/video
  validation, merge, delivery, and the RunPod serverless entry point.
- `segment-template.json`: internal API-format ComfyUI graph copied into the
  image. Do not submit this file as the 60-second request.
- `runpod-ltx23-10s-validation-request.json`: inexpensive deployment test.
- `runpod-ltx23-60s-request.json`: production one-minute brainrot story.

The old `api-workflow.json` and `workflow.json` are intentionally removed.
They contain the obsolete loop workflow and can be submitted accidentally.

## Deploy

1. Replace the repository contents with these files and commit them to `main`.
2. Do not rename `handler.py` or `segment-template.json`.
3. Wait until RunPod reports the new build as completed and the new release as
   active. Do not test an older active release.
4. Use one `PRO 6000 48GB`, `L40`, `L40S`, or `6000 Ada PRO 48GB` GPU per
   worker. Keep max workers at `1` while testing.
5. Set execution timeout to at least `3600` seconds. Keep FlashBoot enabled.
   If initialization ever exceeds seven minutes, set the environment variable
   `RUNPOD_INIT_TIMEOUT=800`.

The large model-download layer appears before the template and handler copies,
so normal handler/template updates should reuse the cached model layer.

## Test in this order

### 1. Ten-second validation

Submit `runpod-ltx23-10s-validation-request.json` to the asynchronous `/run`
request form. Wait for `COMPLETED`. In the response, use the item in `videos`
whose `has_video` and `has_audio` values are both `true`.

Confirm:

- duration is about `10.04` seconds;
- dimensions are exactly `448x768`;
- frame rate is `24`;
- speech is audible;
- Zorp and the camera keep moving throughout.

Do not run the full minute if this test fails.

### 2. Full production request

Submit `runpod-ltx23-60s-request.json` once through `/run`. The handler will:

1. generate each ten-second native audio/video segment;
2. reject malformed, wrong-size, silent, or short files;
3. score motion in each second and retry a weak segment once;
4. pass frame 239 into the next generation;
5. retain 240 frames from every segment, removing duplicate join frames;
6. concatenate exactly 1,440 frames;
7. crop to 9:16, scale to 720x1280, normalize audio, and validate the result.

The final response includes `motion_qa`. `all_segments_passed: true` means every
selected segment passed the automated low-motion check. A warning means the
best retry was used but the segment did not meet the strict motion threshold.

## Quality and output size

The AI source preset is `448x768`, which is substantially larger than the old
256x512 result and is valid for the workflow's half-resolution first pass.
The final TikTok file is 720x1280.

RunPod queue responses have a practical size limit. Without object storage, the
handler caps the final video near 750 kb/s so its base64 payload remains safe.
With compatible object storage configured, the request's 2500 kb/s setting is
used and the response returns a URL. Therefore:

- inline base64: reliable delivery, more compression;
- object storage: noticeably better final compression quality.

The generated visual detail comes from the 448x768 LTX source. Scaling to
720x1280 produces the correct TikTok canvas but does not invent true 720p
detail.

## Current reliability choices

- The known-working distilled FP8 checkpoint remains pinned. Do not change to
  the newer 1.1 checkpoint until this pipeline passes end to end; that model
  uses different calibration and would force a large image rebuild.
- Motion retry is set to `1`. Raising it can improve the worst segment but can
  also increase cost substantially.
- Dialogue is limited to one short quoted line per segment and happens during
  visible movement. This is more reliable than asking the model for continuous
  conversation.
- Audio seams receive 15 ms fades and final loudness normalization.

Cost optimization comes after the full pipeline passes. The first levers will
be lowering the source preset to 384x640, reducing retries for consistently
good prompts, and benchmarking standard 48GB versus 48GB Pro using execution
time rather than cold-start delay.
