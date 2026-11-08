"""Fetch WeSpeaker's ResNet34 speaker-embedding model as ONNX (VoxCeleb, large-margin fine-tuned) into the HF cache and
print its path and input/output names, for scripts/release_speakers.py. RUNS ON THE BOX.

    HF_HOME=$MELATI_DATA/hf python scripts/fetch_wespeaker.py
"""
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"

files = list_repo_files(REPO)
print(files)
onnx = next(f for f in files if f.endswith(".onnx"))
path = hf_hub_download(REPO, onnx)
import onnxruntime as ort  # noqa: E402

s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print(path, [(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])
