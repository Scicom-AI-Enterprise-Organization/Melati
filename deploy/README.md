# deploy/

The shell scripts that set Melati up and run it, and every run behind the released models and the paper, kept as
they were run on one machine with eight GPUs. Each sources `env.sh` (where the repo, the environments and the data
live: the main README's Getting started) and runs as `bash deploy/<name>.sh`. Most name their GPUs in an argument or
inside the script, so read one before running it on your own machine.

## Start here

| script | what it does |
|---|---|
| `box_melati_setup.sh` | `.venv-melati` (stages, servers, tagging) and `.venv-vllm` (the ASR), and every model the pipeline downloads, at the revisions `melati/pins.py` names |
| `box_diar_setup.sh` | `.venv-diar`: the vLLM build the diarization model needs |
| `box_align_setup.sh` | `.venv-align`: the word aligner's environment |
| `box_sed_setup.sh` | `.venv-sed`: PretrainedSED's AudioSet taggers |
| `box_smoke.sh` | check a setup: one VoxConverse recording through every stage, one replica a service, in a few minutes |
| `box_pilot.sh` | bring the model servers up, then run the pipeline over a source list |
| `box_full.sh` | Melati v1: every podcast through the GPU stages, restartable |
| `box_full_publish_loop.sh` | pack, check and publish what the full run has finished, every 6 hours and once at the end |

## Running and changing a cluster

| script | what it does |
|---|---|
| `box_pilot_front.sh`, `box_pilot_rest.sh` | a run split in two: the stages before the ASR, then the rest for one ASR model |
| `box_sounds_pilot.sh` | the sounds stage on a few hours of the full run before it joins it |
| `box_roll.sh` | roll services onto the code in the repo, one replica at a time, while a run uses them |
| `box_retire_replica.sh` | stop one replica for good after its entry in the cluster config was set to null |
| `box_full_stop_driver.sh` | stop the full run's driver and clients, leaving the model servers up |
| `box_full_clients_restart.sh` | restart the full run's clients onto the code now in the repo |
| `box_full_publish.sh` | one publish of the full run (what `box_full_publish_loop.sh` repeats) |
| `box_repack.sh` | re-pack and publish every shard after a change to the export |
| `box_refront.sh`, `box_reseg.sh` | re-run the front or the seg stage of a run, invalidating only what changed after it |
| `full_run_events.txt` | changes made to the cluster during the full run, marked on the paper's throughput figure |

## Data

| script | what it fetches |
|---|---|
| `box_fetch_podcast.sh` | Melati v1's source audio and the two earlier Emilia outputs of the same recordings |
| `box_fetch_podcastfillers.sh` | PodcastFillers, the filler and non-verbal benchmark in the podcast domain |
| `box_fetch_mer2023.sh` | MER2023 (gated) and its labelled clips |
| `box_audioset_fetch.sh`, `box_audioset_strong_fetch.sh` | AudioSet's audio, and its strongly labelled subset |
| `box_beats_fetch.sh` | BEATs fine-tuned on AudioSet-2M, a teacher for the AudioSet tagger |

## The ASR model: Melati-Whisper-Large-V3

| script | what it does |
|---|---|
| `box_asr_setup.sh` | the training and benchmark environment |
| `box_asr_corpus.sh` | the hallucination training corpus and the Common Voice 22 replay pool |
| `box_asr_smoke.sh` | a few steps of every trainer mode |
| `box_asr_train.sh` | the first sweep: mixes, then one job a GPU |
| `box_asr_sweep2.sh`, `box_asr_sweep3.sh`, `box_asr_sweep4.sh` | the second sweep's phases A to C |
| `box_asr_merge_screen.sh` | one merged model, screened on the arms that separate the fine-tunes |
| `box_asr_rebench.sh`, `box_asr_rescore.sh`, `box_asr_rescore_eval.sh` | checkpoints scored again after fixes to the benchmark |
| `box_asr_suppress_check.sh`, `box_asr_forced_lang.sh`, `box_asr_base_cv22.sh` | the benchmark's controls: token suppression, forced language, base whisper-large-v3 on the same CV22 set |
| `box_hf_release.sh` | build, smoke-test and publish the released model and its card |

## Emotion

| script | what it does |
|---|---|
| `box_emo_setup.sh` | the environment the emotion heads are trained, scored and served in |
| `box_emo_retrain.sh` | the five heads from scratch: the corpora (about 28 GB), features, heads, the seven-model average and its thresholds |
| `box_emo_release_heads.sh` | publish the five heads as Melati-Emotion-Heads and read them back |
| `box_emo_release_head.sh` | the Whisper head trained on the licence-clean corpora alone, released as Melati-Emotion-Whisper-Large-V3 |
| `box_emo_distill.sh`, `box_emo_mer2023.sh` | experiments: a distilled head, and MER2023 added to training |
| `box_my_sample.sh` | the Malaysian emotion labelling sample and its page |

## Non-verbal sounds: Melati-NonVerbal-Whisper-Large-V3

The released heads are `box_nv_rel4.sh` (gap head) and `box_nv_frames3.sh` (frame head); `box_nv_ship_rel4.sh` puts
them into the pipeline and re-runs what the paper and the card read. The rest is how they were reached:

| script | what it tried |
|---|---|
| `box_nv_heads.sh`, `box_nv_heads_v2.sh` | the first heads on three encoders, then with real gap candidates |
| `box_nv_novs.sh`, `box_nv_release_head.sh`, `box_nv_release_head2.sh`, `box_nv_ship_release.sh` | the first releases, as each training corpus's licence was checked |
| `box_nv_rel3.sh`, `box_nv_rel5.sh`, `box_nv_rel6.sh`, `box_nv_variant.sh` | the other ways out of NonverbalTTS's and PodcastFillers' licence conflict |
| `box_nv_rule.sh`, `box_nv_rule_apply.sh` | a looser gap finder for breaths, and the head retrained on what it finds |
| `box_nv_frames.sh`, `box_nv_frames2.sh`, `box_nv_frames4.sh` | the frame detector's other versions |
| `box_nv_pf_eval.sh`, `box_nv_pilot.sh`, `box_nv_smoke.sh` | one head through the whole pipeline: PodcastFillers' test episodes, the 200-podcast pilot, one recording |
| `box_my_sounds.sh` | the sounds labelling set, sampled from the full run |

## AudioSet: Melati-AudioSet-EAT-Large and the ensemble

| script | what it does |
|---|---|
| `box_audioset_teachers.sh`, `box_audioset_targets.sh` | the evaluation set decoded and every teacher scored on it; the teachers' probabilities on the training sets |
| `box_audioset_train.sh` | distil the tagger on a pool of GPUs |
| `box_audioset_control.sh`, `box_audioset_ship_v4.sh` | the control on v3's gain, and v4 |
| `box_audioset_v5.sh`, `box_audioset_ship_v5.sh` | more teachers, and v5 |
| `box_audioset_frames.sh` | the frame head behind the inline sound tags (`FRAMES_CKPT`) |
| `box_audioset_windows.sh`, `box_audioset_short_eval.sh`, `box_audioset_eval.sh` | where the tagger hears each sound, how it does on segment-length cuts, and PretrainedSED class by class |
| `box_audioset_pass.sh` | PretrainedSED's taggers over every finished recording |
| `box_audioset_card.sh` | re-write a released tagger's card |
| `box_audioset_ensemble.sh`, `box_audioset_ensemble_update.sh`, `box_ensemble_fresh.sh` | the ensemble, a new version of it, and loading it the way someone else would |

## Comparisons and cards

| script | what it does |
|---|---|
| `box_emilia_baseline.sh`, `box_emilia_resume.sh` | upstream Emilia-Pipe on the same recordings, unchanged |
| `box_uvr_setup.sh` | the UVR-MDX-Net service Emilia-Pipe separates with, for the comparisons |
| `box_cards_redraw.sh` | every Hugging Face card's figures redrawn in the paper's style |
