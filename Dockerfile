FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small and needs no GPU on the host -- the app's
# ~45M-param model runs fine on CPU for single-request, real-time inference.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Requires checkpoints/app_bundle.pt and tokenizer/vocab/ to already exist in the build
# context -- run `python train_and_export.py` locally (ideally on a GPU) before building.
# Training runs fine on CPU too, but is meaningfully slower; either way it should happen
# once, not on every image build -- CPU training inside the build step was tried and is
# too slow to be a reliable build step on most CI/PaaS build-time limits.
COPY checkpoints/app_bundle.pt checkpoints/app_bundle.pt
COPY tokenizer/vocab/ tokenizer/vocab/
COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
