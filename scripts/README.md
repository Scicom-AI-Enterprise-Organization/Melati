# scripts/

The measurements and audits behind `results/`, the dataset's build and release, and the checks that went with them.
Each script's docstring says what it measures and how to run it; most run on the GPU machine against a work directory
of a run or the published dataset (`python scripts/<name>.py --help`).

## A run in progress

| script | what it reports |
|---|---|
| `full_progress.py` | how far a run has got: recordings past each stage, hours finished, failures, the rate |
| `full_failures.py` | the recordings that failed and have not finished since, with their stage and error |
| `full_timeline.py` | a run's progress over time, from its own files (the paper's throughput figure) |
| `cluster_load.py` | where a running cluster spends its GPU time, model by model |
| `empty_text_audit.py`, `probe_empty_asr.py`, `asr_empty_drops.py` | empty transcripts: the model hearing nothing, against the server failing, and what export drops for them |

## The dataset

| script | what it does |
|---|---|
| `build_hf_dataset.py` | pack the run's output into the dataset's parquet shards, block by block |
| `check_hf_dataset.py` | load the dataset the way a user would and check its columns and audio |
| `hf_dataset_upload.py` | upload the dataset folder, resumably, and publish it once it reads back |
| `dataset_card.py`, `stats_for_repo.py` | the dataset card and its figures; its stats for `results/` without transcripts |
| `release_audit.py` | every published shard's columns checked for rows that are wrong rather than noisy |
| `release_audio_check.py`, `release_pairing.py`, `release_speakers.py` | the released audio decodes as its row says, carries its own transcript, and a speaker id holds one voice |
| `release_cut_and_sources.py`, `release_language_fallback.py`, `release_left_out.py`, `release_squeezed.py`, `release_labels.py` | the audit's follow-ups: the cut, the language fallback, what the release leaves out and why, squeezed words, emotion spans against laughs |
| `tag_rates.py`, `count_events.py` | tags an hour of exported speech, and events by kind |

## Melati against Emilia-Pipe

| script | what it measures |
|---|---|
| `make_pilot.py`, `dev_voxconverse.py` | the recordings both pipelines process: a pilot of the podcasts, and VoxConverse |
| `compare_emilia.py`, `show_compare.py` | the two pipelines on the same recordings, and the comparison as tables |
| `audit_audio.py`, `audit_segments.py` | what each export does to the audio, and segments a segmentation defect damaged |
| `nv_injection.py` | where separators put a laugh, and what a quality filter thinks of it |
| `score_diar.py` | the diarization stage's error rate against reference turns |

## Emotion

| script | what it measures |
|---|---|
| `eval_emotion.py` | the emotion candidates on labelled speech in many languages |
| `compare_emotion_ensembles.py` | two seven-model averages side by side, per held-out corpus |
| `distill_gaps.py` | where the released emotion and non-verbal models fall short, class by class |
| `check_ssl_batching.py`, `profile_tag.py` | the tag server's batched encoders against one clip at a time, and their CPU cost |

## Non-verbal sounds and fillers

| script | what it measures |
|---|---|
| `eval_nonverbal.py`, `eval_nonverbal_gaps.py`, `eval_nonverbal_llm.py` | non-verbal candidates and tags against NonverbalTTS's, by acoustics and by an audio LLM |
| `eval_fillers.py` | fillers, laughs and breaths against PodcastFillers' hand labels |
| `audit_nv_tags.py` | the tags PodcastFillers cannot score, judged by an independent ear |
| `gap_curve.py`, `nv_threshold_curve.py`, `frames_diag.py`, `gap_false_tags.py` | precision and recall at every threshold, and what a head's wrong tags are |
| `pf_licences.py`, `podcastfillers_stats.py` | PodcastFillers' episodes by licence, and its events per split |

## Sounds

| script | what it does |
|---|---|
| `sounds_pilot.py` | the sounds stage on a few hours of the full run before it joins it |
| `sounds_listen.py`, `sounds_check.template.html` | the pilot's inline sound tags as clips to listen to, on one page |
| `sounds_frames_peek.py` | a few segments' music frames, as text |

## Hugging Face and the servers

| script | what it does |
|---|---|
| `hf_fetch.py`, `hf_can_read.py`, `hf_repo_layout.py`, `hf_card_meta.py` | fetch a repo's files in parallel; whether the token reads a gated repo; what a repo holds; its card's metadata |
| `hf_repo_check.py`, `hf_public_smoke.py` | a model repo as the public sees it, and its card's snippets run with no token |
| `check_card_figures.py`, `push_card_figures.py` | what a collection's cards show, and uploading redrawn figures alone |
| `probe_vllm_whisper.py`, `check_dnsmos_threads.py`, `fetch_wespeaker.py` | what vLLM's Whisper returns with no language, DNSMOS on one thread, and the speaker model `release_speakers.py` uses |
