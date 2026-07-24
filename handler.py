"""RunPod ComfyUI handler for dense 60-second LTX-2.3 stories.

Two request modes are supported:

1. Normal worker-comfyui API requests containing ``input.workflow``.
2. A local 60-second story mode containing ``input.segment_prompts`` or
   ``input.story_prompt``. Story mode runs six ten-second native audio/video
   workflows sequentially, carries the exact boundary frame into the next
   segment, checks each result for extended low-motion periods, and joins the
   clips into one 720x1280 MP4.

The supplied segment template encodes prompts directly; it does not run a
second prompt-writing model. Gemma remains loaded because LTX-2.3 requires it
as its text encoder. No hosted model or text API is used. The only network
operation after generation is an optional upload of the finished MP4 to
user-provided S3-compatible storage.
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, TypedDict

import runpod
from runpod.serverless.utils import rp_upload

import base_handler


OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output")).resolve()
INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/comfyui/input")).resolve()
SEGMENT_TEMPLATE = Path(
    os.environ.get(
        "LTX23_SEGMENT_TEMPLATE",
        "/opt/runpod-ltx23/segment-template.json",
    )
).resolve()
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".gif"}

# A 7 MiB binary file becomes about 9.34 MiB after base64 encoding, leaving
# headroom under RunPod's 10 MB asynchronous /run response limit.
MAX_BASE64_VIDEO_BYTES = int(
    os.environ.get("MAX_BASE64_VIDEO_BYTES", str(7 * 1024 * 1024))
)
MEDIA_COMMAND_TIMEOUT = int(os.environ.get("MEDIA_COMMAND_TIMEOUT", "900"))
STORY_SEGMENTS = 6
SEGMENT_SECONDS = 10
STORY_SECONDS = STORY_SEGMENTS * SEGMENT_SECONDS
STORY_FPS = 24
SEGMENT_FRAMES = STORY_FPS * SEGMENT_SECONDS + 1
MERGED_FRAMES_PER_SEGMENT = STORY_FPS * SEGMENT_SECONDS

# The two-stage x2 workflow needs both final dimensions divisible by 64 so its
# half-resolution first stage is still divisible by LTX-2.3's required 32.
DEFAULT_SOURCE_WIDTH = 448
DEFAULT_SOURCE_HEIGHT = 768
MAX_STORY_PIXELS = 512 * 896
FINAL_WIDTH = 720
FINAL_HEIGHT = 1280

DEFAULT_MOTION_THRESHOLD = 1.25
DEFAULT_MOTION_RETRIES = 1
MAX_MOTION_RETRIES = 2


class FileState(TypedDict):
    mtime_ns: int
    size: int


class MotionReport(TypedDict):
    mean_score: float
    per_second: list[float]
    low_motion_seconds: list[int]
    longest_low_motion_run: int
    passed: bool


class StoryRequestError(RuntimeError):
    """A user-correctable or workflow execution error in story mode."""


def _video_snapshot() -> dict[Path, FileState]:
    """Return the current state of all supported video files."""
    if not OUTPUT_DIR.exists():
        return {}

    snapshot: dict[Path, FileState] = {}
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        snapshot[path.resolve()] = {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    return snapshot


def _changed_videos(
    before: dict[Path, FileState], after: dict[Path, FileState]
) -> list[Path]:
    """Find video files created or overwritten by the current job."""
    return sorted(
        (path for path, state in after.items() if before.get(path) != state),
        key=lambda path: after[path]["mtime_ns"],
    )


def _storage_config(job: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    """Resolve request-level or environment-level S3-compatible storage."""
    requested = job.get("s3Config")
    if requested is not None:
        required = ("endpointUrl", "accessId", "accessSecret", "bucketName")
        if not isinstance(requested, dict) or any(
            not requested.get(key) for key in required
        ):
            raise StoryRequestError(
                "s3Config must contain endpointUrl, accessId, accessSecret, "
                "and bucketName."
            )
        credentials = {
            "endpointUrl": str(requested["endpointUrl"]),
            "accessId": str(requested["accessId"]),
            "accessSecret": str(requested["accessSecret"]),
        }
        return credentials, str(requested["bucketName"])

    endpoint = os.environ.get("BUCKET_ENDPOINT_URL")
    access_id = os.environ.get("BUCKET_ACCESS_KEY_ID")
    access_secret = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
    if any((endpoint, access_id, access_secret)) and not all(
        (endpoint, access_id, access_secret)
    ):
        raise StoryRequestError(
            "S3 environment configuration is incomplete. Set "
            "BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID, and "
            "BUCKET_SECRET_ACCESS_KEY together."
        )
    if endpoint and access_id and access_secret:
        return None, os.environ.get("BUCKET_NAME")
    return None, None


def _has_environment_storage() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "BUCKET_ENDPOINT_URL",
            "BUCKET_ACCESS_KEY_ID",
            "BUCKET_SECRET_ACCESS_KEY",
        )
    )


def _return_video(job: dict[str, Any], path: Path) -> dict[str, Any]:
    """Upload a video to configured storage or safely return it as base64."""
    relative_name = str(path.relative_to(OUTPUT_DIR))
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    size = path.stat().st_size
    if size == 0:
        raise StoryRequestError(f"{relative_name} was created but is empty.")
    media = _probe_video(path)
    media_fields = {
        "duration_seconds": round(media["duration"], 3),
        "has_video": media["has_video"],
        "has_audio": media["has_audio"],
        "width": media["width"],
        "height": media["height"],
        "fps": media["fps"],
    }

    bucket_creds, bucket_name = _storage_config(job)
    using_env_storage = _has_environment_storage()
    if bucket_creds is not None or using_env_storage:
        job_id = str(job.get("id", "job"))
        url = rp_upload.upload_file_to_bucket(
            file_name=path.name,
            file_location=str(path),
            bucket_creds=bucket_creds,
            bucket_name=bucket_name,
            prefix=job_id,
            extra_args={"ContentType": mime_type},
        )
        return {
            "filename": relative_name,
            "type": "s3_url",
            "mime_type": mime_type,
            "size": size,
            "data": url,
            **media_fields,
        }

    if size > MAX_BASE64_VIDEO_BYTES:
        raise StoryRequestError(
            f"{relative_name} is {size} bytes, larger than the safe inline "
            f"limit of {MAX_BASE64_VIDEO_BYTES} bytes. Add top-level s3Config "
            "to the request or configure S3-compatible endpoint variables."
        )

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "filename": relative_name,
        "type": "base64",
        "mime_type": mime_type,
        "size": size,
        "data": encoded,
        **media_fields,
    }


def _safe_job_token(job: dict[str, Any]) -> str:
    raw = str(job.get("id", "local-job"))
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return (token or "local-job")[:80]


def _integer_input(
    values: dict[str, Any], name: str, default: int
) -> int:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise StoryRequestError(f"{name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StoryRequestError(f"{name} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise StoryRequestError(f"{name} must be an integer.")
    return parsed


def _float_input(
    values: dict[str, Any], name: str, default: float
) -> float:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise StoryRequestError(f"{name} must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StoryRequestError(f"{name} must be a number.") from exc
    if not parsed == parsed or parsed in (float("inf"), float("-inf")):
        raise StoryRequestError(f"{name} must be a finite number.")
    return parsed


def _story_dimensions(job_input: dict[str, Any]) -> tuple[int, int]:
    width = _integer_input(job_input, "width", DEFAULT_SOURCE_WIDTH)
    height = _integer_input(job_input, "height", DEFAULT_SOURCE_HEIGHT)
    if width < 256 or height < 256 or width % 64 or height % 64:
        raise StoryRequestError(
            "width and height must each be at least 256 and divisible by 64. "
            "Use 384x640 for the balanced preset or 448x768 for quality."
        )
    if width * height > MAX_STORY_PIXELS:
        raise StoryRequestError(
            f"width × height must not exceed {MAX_STORY_PIXELS} pixels on the "
            "48 GB reliability preset. Use 448x768 or less."
        )
    return width, height


def _story_prompts(job_input: dict[str, Any]) -> tuple[str, list[str]]:
    """Return an identity bible and six explicitly timed local prompts."""
    global_story = str(job_input.get("story_prompt", "")).strip()
    raw_segments = job_input.get("segment_prompts")

    if raw_segments is not None:
        if (
            not isinstance(raw_segments, list)
            or len(raw_segments) != STORY_SEGMENTS
            or not all(isinstance(item, str) and item.strip() for item in raw_segments)
        ):
            raise StoryRequestError(
                "segment_prompts must be an array of exactly six non-empty "
                "strings, one for each ten-second segment."
            )
        segments = [item.strip() for item in raw_segments]
    elif global_story:
        blocks = [
            block.strip()
            for block in re.split(r"\n\s*---+\s*\n", global_story)
            if block.strip()
        ]
        if len(blocks) != STORY_SEGMENTS:
            raise StoryRequestError(
                "A single story_prompt must contain exactly six ten-second "
                "blocks separated by a line containing ---. For stronger "
                "identity consistency, use story_prompt as the global identity "
                "bible and provide six segment_prompts."
            )
        # Six --- separated blocks are a fully local, deterministic split.
        # Do not repeat the entire six-part story inside every model prompt.
        segments = blocks
        global_story = ""
    else:
        raise StoryRequestError(
            "Provide either story_prompt or exactly six segment_prompts."
        )

    if len(global_story) > 12000 or any(len(prompt) > 4000 for prompt in segments):
        raise StoryRequestError("The story or one of its segment prompts is too long.")

    beat_pattern = re.compile(
        r"(?<!\d)\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*"
        r"\d+(?:\.\d+)?\s*(?:s|sec|second)?",
        re.IGNORECASE,
    )
    for index, prompt in enumerate(segments):
        if len(beat_pattern.findall(prompt)) < 4:
            raise StoryRequestError(
                f"segment_prompts[{index}] needs at least four explicit time "
                "ranges such as [0-2s], [2-4s], [4-6s], [6-8s], and [8-10s]. "
                "Timed beats prevent the model from spending half the clip idle."
            )
    return global_story, segments


def _load_segment_template() -> dict[str, Any]:
    try:
        payload = json.loads(SEGMENT_TEMPLATE.read_text(encoding="utf-8"))
        workflow = payload["input"]["workflow"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise StoryRequestError(
            f"The built-in LTX-2.3 segment template is invalid: {exc}"
        ) from exc

    required_nodes = {
        "1742",  # native audio decode
        "1746",  # audio VAE
        "1756",  # initial audio/video latent
        "1758",  # final audio/video split
        "1772",  # negative conditioning
        "1780",  # direct positive conditioning
        "1784",  # blank frame or continuation image
        "1798",  # MP4 output
    }
    missing = sorted(required_nodes.difference(workflow))
    if missing:
        raise StoryRequestError(
            f"The built-in segment template is missing nodes: {', '.join(missing)}"
        )
    if any(
        node.get("class_type") == "TextGenerateLTX2Prompt"
        for node in workflow.values()
        if isinstance(node, dict)
    ):
        raise StoryRequestError(
            "The segment template still contains TextGenerateLTX2Prompt. "
            "Deploy the corrected direct-prompt template."
        )
    return workflow


def _segment_prompt(
    global_story: str,
    local_prompt: str,
    index: int,
    has_reference: bool,
    attempt: int,
) -> str:
    start = index * SEGMENT_SECONDS
    end = start + SEGMENT_SECONDS
    continuity = (
        "Start from the supplied continuation frame. Preserve the exact same "
        "characters, clothes, objects, lighting, location, and camera direction."
        if has_reference
        else "Establish all recurring characters and visual details clearly."
    )
    parts = []
    if global_story and local_prompt != global_story:
        parts.append(f"GLOBAL STORY AND IDENTITY BIBLE:\n{global_story}")
    parts.extend(
        [
            f"SEGMENT {index + 1} OF {STORY_SEGMENTS} — STORY TIME "
            f"{start}-{end} SECONDS:\n"
            f"{local_prompt}",
            continuity,
            "Execute every timed beat in order. Begin motion in the first frame. "
            "Every one-second interval must show obvious purposeful movement by "
            "the main character plus movement by the camera, a prop, or the "
            "background. Never hold a pose, wait, stare, freeze, or leave the "
            "frame empty. Keep the main character clearly visible throughout. "
            "Any dialogue must be spoken by the visible character while its "
            "mouth, head, hands, and body continue moving.",
            "Generate one uninterrupted ten-second shot with no cut, montage, "
            "restart, fade, title, caption, logo, time jump, or camera teleport. "
            "The final frame must contain clear ongoing motion that continues "
            "naturally into the next segment.",
            "Generate synchronized native audio throughout: intelligible exact "
            "dialogue where quoted, action-matched sound effects, and continuous "
            "ambient sound. Do not output silence, narration, or subtitles.",
        ]
    )
    if attempt:
        parts.append(
            f"MOTION RETRY {attempt}: the earlier take contained an extended "
            "low-motion passage. Increase subject translation, limb motion, prop "
            "motion, parallax, and forward camera tracking in every timed beat. "
            "Do not slow down for dialogue or the ending."
        )
    return "\n\n".join(parts)


def _prepare_segment_workflow(
    template: dict[str, Any],
    *,
    global_story: str,
    local_prompt: str,
    index: int,
    width: int,
    height: int,
    seed: int,
    attempt: int,
    continuation_name: str | None,
    output_prefix: str,
) -> dict[str, Any]:
    workflow = copy.deepcopy(template)
    has_reference = continuation_name is not None

    workflow["2120"]["inputs"]["value"] = SEGMENT_SECONDS
    workflow["2123"]["inputs"]["value"] = STORY_FPS
    workflow["2137"]["inputs"]["value"] = width
    workflow["2138"]["inputs"]["value"] = height
    workflow["1780"]["inputs"]["text"] = _segment_prompt(
        global_story,
        local_prompt,
        index,
        has_reference,
        attempt,
    )
    workflow["1772"]["inputs"]["text"] = (
        "static, frozen, still frame, held pose, idle character, empty frame, "
        "subject exits frame, no motion, slideshow, cut, jump cut, montage, fade, "
        "camera teleport, identity change, clothing change, prop change, text, "
        "subtitles, captions, logo, watermark, blurry, low detail, pixelated, "
        "compression artifacts, bad anatomy, extra limbs, distorted face, "
        "garbled speech, silent audio, offscreen dialogue, desynchronized sound"
    )
    seed_offset = index * 100 + attempt * 10000
    workflow["1760"]["inputs"]["noise_seed"] = seed + seed_offset
    workflow["1796"]["inputs"]["noise_seed"] = seed + seed_offset + 1
    workflow["1798"]["inputs"]["filename_prefix"] = output_prefix
    workflow["1798"]["inputs"]["trim_to_audio"] = False
    workflow["1798"]["inputs"]["crf"] = 18
    workflow["1798"]["inputs"]["save_metadata"] = False
    workflow["1784"]["inputs"]["width"] = width
    workflow["1784"]["inputs"]["height"] = height

    if has_reference:
        workflow["1784"] = {
            "inputs": {"image": continuation_name},
            "class_type": "LoadImage",
            "_meta": {"title": "Previous segment continuation frame"},
        }
        workflow["2116"]["inputs"]["value"] = False
    else:
        workflow["2116"]["inputs"]["value"] = True

    return workflow


def _run_command(
    command: list[str], description: str
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=MEDIA_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise StoryRequestError(
            f"{description} exceeded {MEDIA_COMMAND_TIMEOUT} seconds."
        ) from exc
    if completed.returncode:
        details = (completed.stderr or completed.stdout).strip()[-3000:]
        raise StoryRequestError(f"{description} failed: {details}")
    return completed


def _probe_video(path: Path) -> dict[str, Any]:
    completed = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate,"
            "nb_read_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        f"Validation of {path.name}",
    )
    try:
        probe = json.loads(completed.stdout)
        duration = float(probe["format"]["duration"])
        streams = probe["streams"]
        stream_types = {stream["codec_type"] for stream in streams}
        video_stream = next(
            (stream for stream in streams if stream["codec_type"] == "video"),
            None,
        )
        audio_stream = next(
            (stream for stream in streams if stream["codec_type"] == "audio"),
            None,
        )
        width = int(video_stream["width"]) if video_stream is not None else None
        height = int(video_stream["height"]) if video_stream is not None else None
        frame_count = (
            int(video_stream["nb_read_frames"])
            if video_stream is not None
            and str(video_stream.get("nb_read_frames", "")).isdigit()
            else None
        )
        fps = (
            float(Fraction(video_stream["r_frame_rate"]))
            if video_stream is not None
            else None
        )
        sample_rate = (
            int(audio_stream["sample_rate"])
            if audio_stream is not None
            and str(audio_stream.get("sample_rate", "")).isdigit()
            else None
        )
        channels = (
            int(audio_stream["channels"])
            if audio_stream is not None and audio_stream.get("channels") is not None
            else None
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ) as exc:
        raise StoryRequestError(f"Could not parse media details for {path.name}.") from exc
    return {
        "duration": duration,
        "has_video": "video" in stream_types,
        "has_audio": "audio" in stream_types,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def _extract_continuation_frame(video: Path, destination: Path) -> None:
    """Extract the exact frame retained at the outgoing ten-second boundary."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{MERGED_FRAMES_PER_SEGMENT - 1})",
            "-fps_mode",
            "vfr",
            "-frames:v",
            "1",
            str(destination),
        ],
        f"Boundary-frame extraction from {video.name}",
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise StoryRequestError(
            f"Boundary-frame extraction from {video.name} produced no image."
        )


def _motion_report(path: Path, threshold: float) -> MotionReport:
    """Measure frame-to-frame luminance change in one-second windows."""
    completed = _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            "tblend=all_mode=difference,signalstats,"
            "metadata=mode=print:key=lavfi.signalstats.YAVG:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ],
        f"Motion analysis of {path.name}",
    )

    sums = [0.0] * SEGMENT_SECONDS
    counts = [0] * SEGMENT_SECONDS
    current_time: float | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("frame:"):
            match = re.search(r"pts_time:([0-9]+(?:\.[0-9]+)?)", line)
            current_time = float(match.group(1)) if match else None
            continue
        if current_time is None or not line.startswith("lavfi.signalstats.YAVG="):
            continue
        second = min(int(current_time), SEGMENT_SECONDS - 1)
        try:
            score = float(line.partition("=")[2])
        except ValueError:
            continue
        sums[second] += score
        counts[second] += 1

    if any(count == 0 for count in counts):
        raise StoryRequestError(
            f"Motion analysis of {path.name} did not cover all ten seconds."
        )

    per_second = [total / count for total, count in zip(sums, counts)]
    low_motion_seconds = [
        second for second, score in enumerate(per_second) if score < threshold
    ]
    longest_run = 0
    current_run = 0
    low_set = set(low_motion_seconds)
    for second in range(SEGMENT_SECONDS):
        if second in low_set:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return {
        "mean_score": sum(per_second) / len(per_second),
        "per_second": per_second,
        "low_motion_seconds": low_motion_seconds,
        "longest_low_motion_run": longest_run,
        # One isolated quiet second is tolerable. Two consecutive quiet
        # seconds or more than two quiet seconds total triggers a retry.
        "passed": longest_run <= 1 and len(low_motion_seconds) <= 2,
    }


def _motion_rank(report: MotionReport) -> tuple[int, int, int, float]:
    """Sort passing, continuously moving takes ahead of weaker takes."""
    return (
        0 if report["passed"] else 1,
        report["longest_low_motion_run"],
        len(report["low_motion_seconds"]),
        -report["mean_score"],
    )


def _concat_segments(
    segments: list[Path],
    destination: Path,
    video_bitrate_kbps: int,
) -> None:
    """Join six clips, remove duplicate boundary frames, and master audio."""
    if len(segments) != STORY_SEGMENTS:
        raise StoryRequestError(
            f"Expected {STORY_SEGMENTS} segments, received {len(segments)}."
        )

    maxrate = max(video_bitrate_kbps + 100, int(video_bitrate_kbps * 1.2))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for segment in segments:
        command.extend(["-i", str(segment)])

    filters: list[str] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []
    frame_seconds = 1 / STORY_FPS
    for index in range(STORY_SEGMENTS):
        start_frame = 0 if index == 0 else 1
        end_frame = start_frame + MERGED_FRAMES_PER_SEGMENT
        filters.append(
            f"[{index}:v]trim=start_frame={start_frame}:end_frame={end_frame},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        audio_start = 0.0 if index == 0 else frame_seconds
        filters.append(
            f"[{index}:a]atrim=start={audio_start:.9f}:"
            f"duration={SEGMENT_SECONDS},asetpts=PTS-STARTPTS,"
            "aresample=48000,"
            "aformat=sample_fmts=fltp:sample_rates=48000:"
            "channel_layouts=stereo,"
            "afade=t=in:st=0:d=0.015,"
            f"afade=t=out:st={SEGMENT_SECONDS - 0.015}:d=0.015"
            f"[a{index}]"
        )
        video_labels.append(f"[v{index}]")
        audio_labels.append(f"[a{index}]")

    filters.append(
        "".join(
            label
            for pair in zip(video_labels, audio_labels)
            for label in pair
        )
        + f"concat=n={STORY_SEGMENTS}:v=1:a=1[vcat][acat]"
    )
    filters.append(
        "[vcat]crop=trunc(ih*9/16/2)*2:ih:(iw-ow)/2:0,"
        f"scale={FINAL_WIDTH}:{FINAL_HEIGHT}:flags=lanczos,"
        f"fps={STORY_FPS},setsar=1[vout]"
    )
    filters.append(
        "[acat]loudnorm=I=-16:TP=-1.5:LRA=11,"
        "aresample=48000[aout]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            str(STORY_SECONDS),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-b:v",
            f"{video_bitrate_kbps}k",
            "-maxrate",
            f"{maxrate}k",
            "-bufsize",
            f"{maxrate * 2}k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    _run_command(command, "Final 60-second audio/video merge")


def _progress(job: dict[str, Any], message: str) -> None:
    try:
        runpod.serverless.progress_update(job, message)
    except Exception:
        # Progress reporting must never fail a paid generation.
        pass


def _run_story(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        raise StoryRequestError("input must be a JSON object.")

    width, height = _story_dimensions(job_input)
    global_story, prompts = _story_prompts(job_input)
    seed = _integer_input(job_input, "seed", 42)
    if seed < 0 or seed > 2**63 - 1:
        raise StoryRequestError("seed must be between 0 and 9223372036854775807.")

    motion_threshold = _float_input(
        job_input,
        "motion_threshold",
        DEFAULT_MOTION_THRESHOLD,
    )
    if motion_threshold < 0.25 or motion_threshold > 5.0:
        raise StoryRequestError("motion_threshold must be between 0.25 and 5.0.")
    motion_retries = _integer_input(
        job_input,
        "motion_retries",
        DEFAULT_MOTION_RETRIES,
    )
    if motion_retries < 0 or motion_retries > MAX_MOTION_RETRIES:
        raise StoryRequestError(
            f"motion_retries must be between 0 and {MAX_MOTION_RETRIES}."
        )

    # Inline mode deliberately uses a modest bitrate so base64 stays under
    # /run's 10 MB response limit. Object storage allows a higher-quality file.
    bucket_creds, _ = _storage_config(job)
    using_storage = bucket_creds is not None or _has_environment_storage()
    requested_bitrate = _integer_input(
        job_input, "output_video_bitrate_kbps", 2500
    )
    if requested_bitrate < 350 or requested_bitrate > 5000:
        raise StoryRequestError(
            "output_video_bitrate_kbps must be between 350 and 5000."
        )
    final_bitrate = requested_bitrate if using_storage else min(requested_bitrate, 750)

    template = _load_segment_template()
    token = _safe_job_token(job)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    continuation = INPUT_DIR / f"ltx23_{token}_continuation.png"
    final_path = OUTPUT_DIR / f"ltx23_{token}_60s_with_audio.mp4"
    retry_path = final_path.with_name(f"{final_path.stem}_inline.mp4")
    segment_paths: list[Path] = []
    generated_paths: set[Path] = set()
    motion_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        for index, local_prompt in enumerate(prompts):
            attempts: list[tuple[Path, MotionReport, dict[str, Any], int]] = []
            for attempt in range(motion_retries + 1):
                prefix = (
                    f"ltx23_{token}_segment_{index + 1:02d}"
                    f"_take_{attempt + 1:02d}"
                )
                workflow = _prepare_segment_workflow(
                    template,
                    global_story=global_story,
                    local_prompt=local_prompt,
                    index=index,
                    width=width,
                    height=height,
                    seed=seed,
                    attempt=attempt,
                    continuation_name=continuation.name if index else None,
                    output_prefix=prefix,
                )
                _progress(
                    job,
                    f"Generating segment {index + 1}/{STORY_SEGMENTS}, "
                    f"take {attempt + 1}/{motion_retries + 1}, with native audio",
                )
                before = _video_snapshot()
                result = base_handler.handler(
                    {
                        "id": (
                            f"{job.get('id', 'job')}-segment-{index + 1}"
                            f"-take-{attempt + 1}"
                        ),
                        "input": {"workflow": workflow},
                    }
                )
                if not isinstance(result, dict):
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} returned an "
                        "invalid worker result."
                    )
                if result.get("error"):
                    details = result.get("details") or [result["error"]]
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} failed: "
                        f"{'; '.join(map(str, details))}"
                    )

                changed = _changed_videos(before, _video_snapshot())
                candidates = [path for path in changed if prefix in path.name]
                generated_paths.update(candidates)
                if not candidates:
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} completed but "
                        "produced no video file."
                    )

                valid_candidates: list[tuple[Path, dict[str, Any]]] = []
                for candidate in reversed(candidates):
                    try:
                        candidate_media = _probe_video(candidate)
                    except StoryRequestError:
                        continue
                    if (
                        candidate_media["has_video"]
                        and candidate_media["has_audio"]
                    ):
                        valid_candidates.append((candidate, candidate_media))
                if not valid_candidates:
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} produced no "
                        "file containing both video and audio."
                    )

                segment, media = valid_candidates[0]
                if not 9.9 <= media["duration"] <= 10.2:
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} is "
                        f"{media['duration']:.3f}s; expected about 10.042s."
                    )
                if media["width"] != width or media["height"] != height:
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} encoded at "
                        f"{media['width']}x{media['height']}, not the requested "
                        f"{width}x{height}. Both source dimensions must be "
                        "divisible by 64."
                    )
                if media["fps"] is None or abs(media["fps"] - STORY_FPS) > 0.01:
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} is not 24 fps."
                    )
                if (
                    media["frame_count"] is not None
                    and media["frame_count"] < SEGMENT_FRAMES
                ):
                    raise StoryRequestError(
                        f"Segment {index + 1}, take {attempt + 1} contains only "
                        f"{media['frame_count']} frames; expected {SEGMENT_FRAMES}."
                    )

                report = _motion_report(segment, motion_threshold)
                attempts.append((segment, report, media, attempt))
                _progress(
                    job,
                    f"Segment {index + 1} motion check: "
                    f"{'pass' if report['passed'] else 'retry needed'}",
                )
                if report["passed"]:
                    break

            segment, report, media, selected_attempt = min(
                attempts,
                key=lambda item: _motion_rank(item[1]),
            )
            segment_paths.append(segment)
            motion_summaries.append(
                {
                    "segment": index + 1,
                    "selected_take": selected_attempt + 1,
                    "takes_generated": len(attempts),
                    "passed": report["passed"],
                    "mean_score": round(report["mean_score"], 3),
                    "low_motion_seconds": report["low_motion_seconds"],
                    "longest_low_motion_run": report[
                        "longest_low_motion_run"
                    ],
                    "per_second_scores": [
                        round(score, 3) for score in report["per_second"]
                    ],
                }
            )
            if not report["passed"]:
                warnings.append(
                    f"Segment {index + 1} still had low motion after "
                    f"{len(attempts)} take(s); the best take was used."
                )
            if index < STORY_SEGMENTS - 1:
                _extract_continuation_frame(segment, continuation)

        _progress(
            job,
            "Merging six aligned segments into one 720x1280 MP4 with audio",
        )
        _concat_segments(segment_paths, final_path, final_bitrate)
        final_media = _probe_video(final_path)
        if (
            not final_media["has_video"]
            or not final_media["has_audio"]
            or not 59.9 <= final_media["duration"] <= 60.1
            or final_media["width"] != FINAL_WIDTH
            or final_media["height"] != FINAL_HEIGHT
            or final_media["fps"] is None
            or abs(final_media["fps"] - STORY_FPS) > 0.01
        ):
            raise StoryRequestError(
                "Final validation failed: expected a 60-second, 720x1280, "
                "24 fps file containing video and audio."
            )
        if not using_storage and final_path.stat().st_size > MAX_BASE64_VIDEO_BYTES:
            # One deterministic lower-bitrate retry prevents an otherwise
            # successful generation from failing only at the response boundary.
            _concat_segments(segment_paths, retry_path, 550)
            retry_path.replace(final_path)
            final_media = _probe_video(final_path)
            if final_path.stat().st_size > MAX_BASE64_VIDEO_BYTES:
                raise StoryRequestError(
                    "The final MP4 is still too large for inline delivery. Add "
                    "top-level s3Config and retry."
                )

        delivered_video = _return_video(job, final_path)
        return {
            "videos": [delivered_video],
            "duration_seconds": round(final_media["duration"], 3),
            "segments": STORY_SEGMENTS,
            "audio": True,
            "continuity": (
                "exact retained boundary frame conditions each next segment; "
                "duplicate first frames are removed during merge"
            ),
            "source_width": width,
            "source_height": height,
            "width": final_media["width"],
            "height": final_media["height"],
            "fps": STORY_FPS,
            "motion_qa": {
                "threshold": motion_threshold,
                "all_segments_passed": all(
                    item["passed"] for item in motion_summaries
                ),
                "segments": motion_summaries,
            },
            "warnings": warnings,
        }
    finally:
        continuation.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        retry_path.unlink(missing_ok=True)
        for path in generated_paths:
            path.unlink(missing_ok=True)


def _is_story_request(job: dict[str, Any]) -> bool:
    job_input = job.get("input")
    return isinstance(job_input, dict) and (
        "story_prompt" in job_input or "segment_prompts" in job_input
    )


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Run a six-segment story or a normal worker-comfyui workflow."""
    if _is_story_request(job):
        try:
            return _run_story(job)
        except StoryRequestError as exc:
            return {"error": "60-second story generation failed", "details": [str(exc)]}
        except Exception as exc:
            return {
                "error": "60-second story generation failed unexpectedly",
                "details": [f"{type(exc).__name__}: {exc}"],
            }

    try:
        _storage_config(job)
    except StoryRequestError as exc:
        return {"error": "Invalid video delivery configuration", "details": [str(exc)]}

    before = _video_snapshot()
    result = base_handler.handler(job)
    if not isinstance(result, dict) or result.get("error"):
        return result

    changed = _changed_videos(before, _video_snapshot())
    if not changed:
        return result

    try:
        videos = [_return_video(job, path) for path in changed]
    except Exception as exc:
        return {
            "error": "Video was generated but could not be returned",
            "details": [str(exc)],
        }

    result.pop("status", None)
    if result.get("images") == []:
        result.pop("images", None)
    result["videos"] = videos
    return result


if __name__ == "__main__":
    print("worker-comfyui-ltx23 - Starting audio/video story handler...")
    runpod.serverless.start({"handler": handler})
