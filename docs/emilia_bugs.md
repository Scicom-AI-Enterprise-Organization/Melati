# What is wrong with Emilia-Pipe

A check of Emilia-Pipe and of the fixes in its [mesolitica fork](https://github.com/mesolitica/Emilia): every
defect found in the upstream code (`open-mmlab/Amphion/preprocessors/Emilia`, commit `26f6883`, vendored unchanged in
`baselines/emilia_upstream/`), what the mesolitica fork changed, what the Scicom fork
([Scicom-AI-Enterprise-Organization/Emilia](https://github.com/Scicom-AI-Enterprise-Organization/Emilia)) changed
after that, and what Melati does instead. Line numbers are upstream's. Measured effects go in `results/emilia/` and are
summarised at the end once they exist; nothing below is an estimate.

Lineage: `mesolitica/Emilia` (2025-05, four commits) is the fork that produced `mesolitica/Malaysian-Emilia-v2`, whose
card says it "should fixed https://github.com/open-mmlab/Amphion/issues/436".
`Scicom-AI-Enterprise-Organization/Emilia` starts from mesolitica's history and adds a multi-GPU orchestrator, remote
ASR and the post-processing chain.

## 1. Every exported segment is 32 kbit/s MP3, band-limited at about 8 kHz (Amphion #436)

`utils/tool.py:65`: `audio.export(path, format="mp3")`, pydub with no bitrate, so ffmpeg's libmp3lame default applies.
The issue's reporter measured 32 kbit/s, and LAME low-passes at that rate: the "24 kHz" dataset carries nothing above
about 8 kHz, the bandwidth of 16 kHz audio. The issue is still open upstream.

- **mesolitica:** `write_mp3` became `sf.write(path, x, sr)`: libsndfile's MP3 encoder, at its own default quality.
- **Melati:** FLAC, 16-bit. No lossy encoder between the source and the dataset (`melati/audio.py`).
- **Measured:** `scripts/audit_audio.py` on Malaysian-Emilia v1 (upstream export) and v2 (mesolitica's) segments of
  the same podcasts, and on the official Emilia release.

## 2. Every segment is peak-normalised to full scale

`utils/tool.py:58`: `x = np.int16(x / np.max(np.abs(x)) * 32767)`. Each segment's loudest sample is put at 0 dBFS, so a
segment's loudness follows its loudest transient (a laugh, a plosive, a cough) rather than the voice. Before that,
`main.py:75-84` computes a loudness gain towards -20 dBFS, clamps it to +-3 dB, applies it, and then divides the whole
waveform by its peak (`waveform /= max_amplitude`, line 84), which undoes the loudness step. A silent segment divides
by zero.

- **mesolitica:** `sf.write` removes the per-segment normalisation; the whole-file peak normalisation stays.
- **Melati:** one gain per recording (EBU R128 -23 LUFS with a -1 dBFS peak ceiling); within a recording a whisper
  stays quieter than a shout; each segment's loudness and peak are measured and stored, never imposed.

## 3. Voice activity detection only runs on turns longer than 20 seconds

`models/silero_vad.py:13,146`: `VAD_THRESHOLD = 20`, and `if end - start <= VAD_THRESHOLD:` appends the diarization
turn as it is. For every turn of up to 20 s, the segment's edges are pyannote's turn edges, with whatever silence,
breath or noise pyannote included, and Silero never looks at it.

- **Melati:** Silero runs on every recording (`stage_seg.py`); segments are built from its speech islands.

## 4. Overlapping turns stay in both speakers' segments; contained turns vanish into the other speaker's

`models/silero_vad.py:137-140`: `if end <= last_end: continue; last_end = end`. Turns arrive sorted by start. A turn
that ends before the previous one ends (a "yeah" inside someone else's sentence) is skipped, but its audio is still
inside the longer turn's segment. A turn that only partly overlaps the previous one keeps the overlap, which is then
in two segments, one per speaker. The Emilia paper lists "segments may contain multiple speakers or overlap" among its
limitations; this is the line that does it.

- **Melati:** overlaps between any two speakers' turns (plus 0.1 s) are removed from every segment
  (`stage_seg.overlaps`), and how much was removed is a column.

## 5. A segment longer than 30 s is cut every 30 s exactly

`main.py:196`: `vad["end"] = current_start + MAX_SEGMENT_LENGTH`, in a loop. A speech island longer than 30 s is chopped
wherever 30 s lands, mid-word if that is where it lands.

- **Melati:** segments are cut only in pauses; an island longer than 29.5 s with no 100 ms pause is cut at its
  lowest VAD-probability frame, and flagged `forced_cut`.

## 6. Short same-speaker pieces are glued across gaps of up to 2 s

`main.py:178,214`: `MERGE_GAP = 2`. A piece shorter than 3 s is appended to the previous segment of the same speaker if
the gap is under 2 s, and the gap's audio goes with it.

- **Melati:** islands are joined across pauses of at most 1.0 s, and a segment closes at the first pause of 0.3 s or
  more once it is 15 s long.

## 7. Code-switched speech is filtered out by language-ID confidence

`main.py:278`: a segment is kept only if Whisper's language ID says a supported language with probability > 0.8. A
sentence that switches between Malay and English is exactly the kind Whisper is unsure about.

- **mesolitica:** lowered to 0.5.
- **Melati:** no language-ID filter; the language the ASR model reports is a column.

## 8. Quality is DNSMOS at 16 kHz, applied before export

`utils/tool.py:260`: `min_dnsmos=3` on DNSMOS P.835 OVRL, computed on a 16 kHz copy, and segments below it are never
written. DNSMOS cannot hear above 8 kHz, and the threshold cannot be revisited without re-running everything.

- **mesolitica:** lowered to 2.8 (and `min_duration` to 2 s).
- **Melati:** SOTA-MOS at 24 kHz and DNSMOS, on both the mix and the separated speech, kept as columns; a subset is a
  query. `scripts/nv_injection.py` measures what each score does to a segment when a laugh or a cough is added to it.

## 9. Separation is a music separator, and it is applied to everything

`main.py` step 1: UVR-MDX-Net-Inst_HQ_3 splits vocals from instruments. It knows nothing about sound effects, and the
vocals stem is exported whether or not there was anything to remove.

- **Scicom fork:** separation can be switched off; measured there, off loses 10% of hours and leaves 80% of clips
  with no segments at all, against 45% with it on.
- **Melati:** DBS-TIGER (speech / music / effects); the export keeps the original mix where the separator found
  little to remove and SOTA-MOS does not prefer its output. `scripts/nv_injection.py` measures which stem each
  separator files a laugh, cough or sigh under.

## 10. The transcription prompt asks for fillers, and a prompt is what Whisper repeats over silence

`main.py:540`: every segment is decoded with `initial_prompt="Um, Uh, Ah. Like, you know. I mean, right. ... Emm. So.
Oh. 嗯,呃,..."`, so that Whisper writes disfluencies. A prompt is also text Whisper can emit when it hears nothing,
and nothing downstream checks for a transcript with no speech behind it.

- **Melati:** an ASR model trained not to write text over voice-free audio (`asr/`), and fillers recovered from the
  audio between aligned words (`melati/pipeline.py`, stage `fill`), with their own times.

## 11. Robustness: one bad file ends the run, nothing resumes

Upstream `main.py:573` loops over files with no `try`, writes no resume marker, and `filter()` divides by the segment
count, so a file with no segments raises `ZeroDivisionError` (line 385).

Measured on Melati's 200-recording podcast pilot: the run died on recording 97 after 52 minutes ("average MOS: nan",
then the division), leaving 103 recordings unprocessed. `deploy/box_emilia_resume.sh` finishes such a run with
upstream's code unchanged and records each recording it died on (`emilia_crashed.jsonl`) as producing nothing.

- **mesolitica:** resume by finished JSON, `try/except` per file, files over 5 h skipped. Its `get_length` used
  `soundfile.read`, which returns nothing for MP3 on libsndfile < 1.1 (every MP3 skipped as "too long"), and its
  `--global-size` sharding (`len // global_size`) silently drops the remainder.
- **Scicom fork:** both fixed (a decode fallback chain; balanced `chunks()`).
- **Melati:** every stage resumes by output existence and writes through a temporary file and a rename.
