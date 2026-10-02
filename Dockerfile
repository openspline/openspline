FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
COPY packages/node/package.json packages/node/package.json
COPY packages/browser/package.json packages/browser/package.json
COPY packages/react/package.json packages/react/package.json
RUN npm ci
COPY packages/node packages/node
COPY packages/browser packages/browser
COPY packages/react packages/react
COPY scripts/copy-browser.mjs scripts/copy-browser.mjs
RUN npm run build

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY packages/server /app/packages/server
COPY --from=frontend /build/packages/server/src/openspline_server/static/browser /app/packages/server/src/openspline_server/static/browser
COPY requirements/server.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && pip install --no-cache-dir --no-deps /app/packages/server
COPY workers.yaml /app/workers.yaml
ENV OPENSPLINE_MODEL_DIR=/models/avatar OPENSPLINE_AUDIO_MODEL_DIR=/models/audio OPENSPLINE_RUNTIME_DIR=/runtime OPENSPLINE_GPU_LOCK_DIR=/gpu-locks HF_HOME=/models/cache
EXPOSE 7860
ENTRYPOINT ["openspline"]
CMD ["serve", "--host", "0.0.0.0", "--config", "/app/workers.yaml"]
