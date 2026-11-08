"""One Hugging Face checkpoint from a Melati head: the encoder of openai/whisper-large-v3 (float16) and the head
(float32), loaded by AutoModel with trust_remote_code (melati_hf/modeling_melati_head.py). Both the non-verbal head
(nonverbal/train.py) and the emotion heads (emotion/probe.py) save {"labels", "layers", "dim", "state"} with the same
layout (melati.ser.head_module: mix, net.0 LayerNorm, net.1 Linear, net.4 Linear). A frame-level detector
(nonverbal/frames.py's checkpoint: {"classes", "layers", "fps", "state"}) can ride on the same encoder."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_merged(head: Path, out: Path, frames: Path | None = None, frame_thresholds=None, **config):
    """Write model.safetensors, config.json (with `config` added), the remote code, and the feature extractor's
    config into `out`; with `frames`, the frame-level detector too (its thresholds in config.frame_head)."""
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
    sys.path.insert(0, str(HERE))
    from configuration_melati_head import MelatiHeadConfig
    from modeling_melati_head import MelatiHeadModel
    ck = torch.load(head, map_location="cpu")
    base = WhisperModel.from_pretrained("openai/whisper-large-v3", dtype=torch.float32)
    enc = {k: v for k, v in base.config.to_dict().items()
           if k not in ("architectures", "_name_or_path", "model_type", "transformers_version", "torch_dtype", "dtype")}
    fck = torch.load(frames, map_location="cpu") if frames else None
    if fck:
        convs = sorted({k.split(".")[1] for k in fck["state"] if k.startswith("convs.")}, key=int)
        config["frame_head"] = {"layers": list(fck["layers"]), "fps": int(fck["fps"]), "classes": list(fck["classes"]),
                                "hidden": int(fck["state"]["inp.weight"].shape[0]),
                                "kernel": int(fck["state"]["convs.0.weight"].shape[-1]),
                                "dilations": [2 ** int(i) for i in convs],
                                "thresholds": dict(frame_thresholds or {})}
    cfg = MelatiHeadConfig(encoder=enc, labels=list(ck["labels"]), hidden=256, **config)
    MelatiHeadConfig.register_for_auto_class()
    MelatiHeadModel.register_for_auto_class("AutoModel")
    model = MelatiHeadModel(cfg)
    model.encoder.load_state_dict(base.encoder.state_dict())
    st = ck["state"]
    assert int(ck["layers"]) == model.layer_mix.shape[0] and int(ck["dim"]) == model.head_norm.weight.shape[0]
    with torch.no_grad():
        model.layer_mix.copy_(st["mix"])
        model.head_norm.weight.copy_(st["net.0.weight"]); model.head_norm.bias.copy_(st["net.0.bias"])
        model.head_fc1.weight.copy_(st["net.1.weight"]); model.head_fc1.bias.copy_(st["net.1.bias"])
        model.head_out.weight.copy_(st["net.4.weight"]); model.head_out.bias.copy_(st["net.4.bias"])
        if fck:
            missing = model.frame_head.load_state_dict(fck["state"], strict=True)
            assert not missing.missing_keys and not missing.unexpected_keys, missing
    model.encoder.half()
    model.save_pretrained(str(out), safe_serialization=True)
    WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3").save_pretrained(str(out))
    del base, model
