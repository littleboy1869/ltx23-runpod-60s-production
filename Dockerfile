# RunPod's ComfyUI base already contains handler.py and
# runpod.serverless.start(...). Do not override CMD or ENTRYPOINT.
FROM runpod/worker-comfyui:5.8.4-base

WORKDIR /comfyui

# Install the exact custom-node revisions detected from the workflow.
# A missing revision fails the build instead of silently using an unknown HEAD.
RUN set -eux; \
    git clone https://github.com/kijai/ComfyUI-KJNodes /comfyui/custom_nodes/ComfyUI-KJNodes; \
    git -C /comfyui/custom_nodes/ComfyUI-KJNodes checkout debc47061f722e93fc9885c19facefcf59b6426a; \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /comfyui/custom_nodes/ComfyUI-VideoHelperSuite; \
    git -C /comfyui/custom_nodes/ComfyUI-VideoHelperSuite checkout 8923bd836bdab8b7bbdf4ed104b7d045e70c66e2; \
    git clone https://github.com/yolain/ComfyUI-Easy-Use /comfyui/custom_nodes/ComfyUI-Easy-Use; \
    git -C /comfyui/custom_nodes/ComfyUI-Easy-Use checkout b11c63487243d183a967261bdf019c5afcec2cf7

# A raw git clone does not install Python dependencies.
RUN set -eux; \
    for package in \
        /comfyui/custom_nodes/ComfyUI-KJNodes \
        /comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
        /comfyui/custom_nodes/ComfyUI-Easy-Use; \
    do \
        if [ -f "$package/requirements.txt" ]; then \
            python -m pip install --no-cache-dir -r "$package/requirements.txt"; \
        fi; \
    done

# Download the seven exact files referenced by the workflow.
# All repositories below are public, so no build-time token is required.
RUN set -eux; \
    download_model() { \
        url="$1"; \
        relative_path="$2"; \
        filename="$3"; \
        mkdir -p "/comfyui/$relative_path"; \
        attempt=1; \
        for delay in 0 10 20 30 60; do \
            if [ "$delay" -gt 0 ]; then sleep "$delay"; fi; \
            if comfy model download \
                --url "$url" \
                --relative-path "$relative_path" \
                --filename "$filename"; \
            then \
                return 0; \
            fi; \
            rm -f "/comfyui/$relative_path/$filename"; \
            echo "Download attempt $attempt failed for $filename" >&2; \
            attempt=$((attempt + 1)); \
        done; \
        echo "Download failed after 5 attempts: $filename" >&2; \
        return 1; \
    }; \
    download_model \
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors" \
        "models/vae" \
        "LTX23_video_vae_bf16.safetensors"; \
    download_model \
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
        "models/latent_upscale_models" \
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"; \
    download_model \
        "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
        "models/text_encoders" \
        "gemma_3_12B_it_fp8_scaled.safetensors"; \
    download_model \
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
        "models/text_encoders" \
        "ltx-2.3_text_projection_bf16.safetensors"; \
    download_model \
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors" \
        "models/vae" \
        "LTX23_audio_vae_bf16.safetensors"; \
    download_model \
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors" \
        "models/vae/vae_approx" \
        "taeltx2_3.safetensors"; \
    download_model \
        "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors" \
        "models/diffusion_models/LTXVideo/v2" \
        "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"

# Keep this compatibility update after the large model-download layer. That
# preserves Docker layer caching, so changing the code below does not force
# RunPod to download the 34 GB model layer again.
#
# ComfyUI v0.20.1 supplies the current audio nodes used by LTX-2.3. The pinned
# KJNodes revision includes the LTX-2.3 audio-VAE loader fix, NAG fixes, and
# the later ComfyUI audio-VAE compatibility fix.
RUN set -eux; \
    git -C /comfyui fetch origin 64b8457f55cd7fb54ca7a956d9c73b505e903e0c --depth=1; \
    git -C /comfyui checkout 64b8457f55cd7fb54ca7a956d9c73b505e903e0c; \
    python -m pip install --no-cache-dir -r /comfyui/requirements.txt; \
    git -C /comfyui/custom_nodes/ComfyUI-KJNodes fetch origin 6f9c24a5354b53c2f4b9170f8ac8d3d0c1883ee3 --depth=1; \
    git -C /comfyui/custom_nodes/ComfyUI-KJNodes checkout 6f9c24a5354b53c2f4b9170f8ac8d3d0c1883ee3; \
    python -m pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt; \
    python -m pip install --no-cache-dir "transformers>=4.50.3,<5" "huggingface-hub<1.0"; \
    command -v ffmpeg; \
    command -v ffprobe

# The custom handler uses this flat, API-format workflow as the template for
# each ten-second audio/video segment. It sends prompts directly to the LTX
# text encoder; no prompt-enhancer generation node runs inside the workflow.
RUN mkdir -p /opt/runpod-ltx23
COPY segment-template.json /opt/runpod-ltx23/segment-template.json
RUN python -c "import json; p=json.load(open('/opt/runpod-ltx23/segment-template.json')); w=p['input']['workflow']; assert '1798' in w and '1780' in w; assert all(n.get('class_type') != 'TextGenerateLTX2Prompt' for n in w.values())"

# Fail the image build if ComfyUI or a custom node cannot import. This catches
# dependency drift before a paid GPU worker starts.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Keep RunPod's tested ComfyUI execution logic, then replace its image-only
# entry point with a wrapper that also returns VideoHelperSuite output files.
RUN cp /handler.py /base_handler.py
COPY handler.py /handler.py
RUN python -m py_compile /handler.py

# /start.sh from the base image launches /handler.py.
WORKDIR /
