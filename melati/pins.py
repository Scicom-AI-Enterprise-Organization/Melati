"""The revision of every model and checkout Melati downloads.

A repository's main branch can change between two runs: Who-Spoke-When's weights changed on 2026-10-22, three days
before the full run began, so a box set up the week before diarized with another model. The servers load their models
by name when they start, so each name is pinned here once: the loaders pass `rev(name)`,
the setup scripts download and check out the same revisions (`python3 melati/pins.py <name>` prints one), and the
cluster configs give vLLM its models' (tests/test_pins.py holds them to this table).

Each is the revision the published dataset was made with, what main held when the full run began on 2026-10-25,
except Melati's own releases, pinned where the pipeline was checked against them. The pip packages whose version
changes what is computed are pinned in the setup scripts beside the rest of their environment.

    python3 melati/pins.py openai/whisper-large-v3
"""
import sys

HUB = {      # Hugging Face model repositories
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "facebook/w2v-bert-2.0": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
    "microsoft/wavlm-large": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    "MIT/ast-finetuned-audioset-10-10-0.4593": "f826b80d28226b62986cc218e5cec390b1096902",
    "emotion2vec/emotion2vec_plus_large": "6c303ba987b86b93193de93e34bb2b077a6bedc4",
    "MERaLiON/MERaLiON-SER-v1": "7e3ee6fa4534dea8316e8ca43e377e2fbb58496b",
    "JusperLee/DBS-TIGER-DnR": "22da7d4cd3ea63f9365181008bff1c0d2ab6fb2b",
    "JusperLee/DBS-BSRNN-DnR": "5af343f57f106fbb95efe8a60f1a3999eeb0aef8",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "26291f793822fb6be9555850f06dfe95f2d7e695",
    "Scicom-intl/Who-Spoke-When-Over-the-Phone-Whisper-Large-V3-Merged": "6b30eed0c28a4dc0003b8dc1f11ee95ed2637a4a",
    "Scicom-intl/Where-Words-Begin-and-End-Whisper-Large-V3": "618763b1ed3ae17cb9a2062d6d4d0de436828c7a",
    "Scicom-intl/SOTA-MOS": "bffd4249cad3eb0fa8624fc5076325f6b956be01",
    "Scicom-intl/Melati-Whisper-Large-V3": "cdbb981c6ce8d87b338ed9503ef9a51f24a2a6aa",
    "Scicom-intl/Melati-NonVerbal-Whisper-Large-V3": "24e22aab38b70a73c901784ed407fec083772d4e",
    "Scicom-intl/Melati-Emotion-Heads": "ea304af6989e54228323e29ed942b3ec76ee6f90",
    "Scicom-intl/Melati-AudioSet-EAT-Large": "78248ff107120278b14465c301a20b57223d6b28",       # v4
    "Scicom-intl/Melati-AudioSet-Ensemble": "b3175658f59fea64ecddee98661ec1dabb0f59a5",
    "diarizers-community/voxconverse": "3acfa1b45ca4b7419aee999d67d94c617f9c9d47",   # a dataset: box_smoke.sh
}
GIT = {      # code the setup scripts check out
    "https://github.com/JusperLee/Dynamic-Band-Splitting": "ff526463a58ee0278e7bd05c4b8bce112b743e34",
    "https://github.com/Scicom-AI-Enterprise-Organization/SOTA-MOS": "72771fbc157afca19f89c9f5c03a36e6039cde41",
    "https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2": "8986ccb04f7b26666e873c93ff8ba777e25a07e4",
    "https://github.com/fschmid56/PretrainedSED": "1aa47e482f7e89904cba2338999345025d8b4e36",
}


def rev(name: str):
    """The pinned revision of a Hub repository or a git URL; None for anything else, such as a local folder."""
    return HUB.get(name) or GIT.get(name.removesuffix(".git"))


def local(repo: str) -> str:
    """The repository at its pinned revision, as a folder, for a loader that cannot be given a revision: FunASR's
    download from the Hub drops the one it is passed."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev(repo))


if __name__ == "__main__":
    if len(sys.argv) != 2 or rev(sys.argv[1]) is None:
        sys.exit(f"usage: python3 melati/pins.py <name>; pinned: {', '.join([*HUB, *GIT])}")
    print(rev(sys.argv[1]))
