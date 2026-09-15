# python:3.12-slim: TensorFlow 2.21 ships wheels only up to cp313, and 3.12
# is the safest, most widely-supported intersection with scikit-learn 1.9.0
# and Keras 3.15 (see REPORT.md for the full version-pinning rationale).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Layer-cache dependency installation separately from source code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# chmod explicitly rather than relying on the executable bit surviving every
# checkout/clone: NTFS (Windows) doesn't track the Unix +x bit, so a commit
# made from a Windows checkout can silently lose it even with correct LF
# line endings, leaving the container failing at startup with
# "permission denied" on the entrypoint.
RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/artifacts /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# 127.0.0.1, not "localhost": avoids the container resolving to the ::1
# (IPv6) /etc/hosts entry and getting a spurious connection-refused against
# a server that's only actually bound on the IPv4 side, on images/tools
# where that ordering bites (see the matching note in frontend/Dockerfile).
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
