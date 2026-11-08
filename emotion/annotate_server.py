#!/usr/bin/env python3
"""The labelling pages for Melati's Malaysian test sets. RUNS ON THE BOX; the audio never leaves it.

Annotators open a page through a port forward, type their name once, and label one clip at a time:
  /         emotion: the seven emotions, or "unclear" (emotion/sample_my_test.py)
  /sounds   non-verbal: what is heard at a marked moment of a clip, never told what was tagged there
            (nonverbal/sample_my_sounds.py; breaths, laughs, coughs, fillers and untagged gaps, shuffled)
Each clip is shown to up to --per-clip annotators, never twice to the same one, and every answer is appended to
its task's labels file with who gave it and when, so agreement can be measured and nothing is overwritten.

    python emotion/annotate_server.py --sample $MELATI_DATA/emo/my_test/sample.jsonl \\
        --labels $MELATI_DATA/emo/my_test/labels.jsonl --port 17900 \\
        --sounds-sample $MELATI_DATA/nv/my_sounds/sample.jsonl --sounds-labels $MELATI_DATA/nv/my_sounds/labels.jsonl
    # then the pages are on the box's port 17900: http://<box>:17900 (emotion) and http://<box>:17900/sounds
"""
import argparse, json, threading, time
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

LABELS = ("neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted")
SOUNDS = ("laugh", "breath", "cough", "filler (um, erm, eh)", "speech (a word)", "other sound", "nothing")

STYLE = """<style>
:root{--bg:#fafafa;--fg:#18181b;--muted:#6b7280;--line:#e5e7eb;--accent:#183b8d;--mark:#183b8d33}
@media (prefers-color-scheme:dark){:root{--bg:#18181b;--fg:#f4f4f5;--muted:#a1a1aa;--line:#3f3f46;--accent:#f58220;--mark:#f5822044}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px}
button{font:inherit;padding:10px 14px;margin:4px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent)}.muted{color:var(--muted)}#text{margin:12px 0;font-size:17px}
audio{width:100%;margin:12px 0}input{font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg)}
#bar{position:relative;height:28px;border:1px solid var(--line);border-radius:6px;margin:4px 0 8px}
#mark{position:absolute;top:0;bottom:0;background:var(--mark);border-left:2px solid var(--accent);border-right:2px solid var(--accent)}
#head{position:absolute;top:0;bottom:0;width:2px;background:var(--fg)}
</style>"""

COMMON_JS = """
let cur=null;const who=document.getElementById('who');
try{who.value=localStorage.getItem('who')||''}catch(e){}
who.onchange=()=>{try{localStorage.setItem('who',who.value)}catch(e){};next()};
const b=document.getElementById('btns');
L.forEach((l,i)=>{const x=document.createElement('button');x.textContent=(i+1)+' '+l;x.onclick=()=>send(l);b.appendChild(x)});
const u=document.createElement('button');u.textContent='0 unclear';u.onclick=()=>send('unclear');b.appendChild(u);
async function next(){if(!who.value){document.getElementById('msg').textContent='Type your name first.';return}
 const r=await fetch(BASE+'next?who='+encodeURIComponent(who.value));const j=await r.json();
 document.getElementById('prog').textContent=j.done+' labelled by you, '+j.left+' left';
 if(!j.id){document.getElementById('clip').style.display='none';document.getElementById('msg').textContent='All done. Thank you.';return}
 cur=j.id;const a=document.getElementById('a');a.src=BASE+'audio/'+encodeURIComponent(j.id);a.play().catch(()=>{});
 document.getElementById('text').textContent=j.text||'';document.getElementById('msg').textContent='';show(j)}
async function send(l){if(!cur)return;await fetch(BASE+'label',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({id:cur,who:who.value,label:l})});next()}
document.onkeydown=e=>{if(e.target===who)return;if(e.key===' '){e.preventDefault();const a=document.getElementById('a');a.currentTime=0;a.play()}
 const k=parseInt(e.key);if(k>=1&&k<=L.length)send(L[k-1]);if(e.key==='0')send('unclear')};
next();
"""

EMOTION_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Melati emotion labels</title>""" + STYLE + """</head><body>
<h2>Which emotion does the speaker's voice express?</h2>
<p class="muted">Listen to the voice, not only the words. Most podcast speech is neutral: pick neutral unless the emotion is clear.
Keys 1-7 pick a label, 0 is unclear, space replays. <a href="/sounds/">Sounds task</a></p>
<p>Your name: <input id="who" placeholder="name"> <span id="prog" class="muted"></span></p>
<div id="clip"><audio id="a" controls></audio><div id="text"></div><div id="btns"></div></div>
<p id="msg" class="muted"></p>
<script>
const L=%LABELS%;const BASE='/';function show(j){}
""" + COMMON_JS + "</script></body></html>"

SOUNDS_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Melati sound labels</title>""" + STYLE + """</head><body>
<h2>What do you hear in the marked part?</h2>
<p class="muted">The bar is the clip and the shaded part is the moment to judge; the line moves as it plays. Judge only the shaded
part: a laugh, a breath, a cough, a filler (um, erm, eh), a spoken word, some other sound, or nothing. The text is what was said
around it, with &#9646; where the shaded part starts (fillers are left out of the text). Keys 1-7 pick, 0 is unclear, space replays.
<a href="/">Emotion task</a></p>
<p>Your name: <input id="who" placeholder="name"> <span id="prog" class="muted"></span></p>
<div id="clip"><audio id="a" controls></audio><div id="bar"><div id="mark"></div><div id="head"></div></div>
<div id="text"></div><div id="btns"></div></div>
<p id="msg" class="muted"></p>
<script>
const L=%LABELS%;const BASE='/sounds/';let dur=1;
function show(j){dur=j.clip_s||1;const m=document.getElementById('mark');
 m.style.left=(100*j.mark[0]/dur)+'%';m.style.width=Math.max(0.8,100*(j.mark[1]-j.mark[0])/dur)+'%'}
(function tick(){const a=document.getElementById('a');document.getElementById('head').style.left=(100*Math.min(1,a.currentTime/dur))+'%';
 requestAnimationFrame(tick)})();
""" + COMMON_JS + "</script></body></html>"


class Label(BaseModel):
    id: str
    who: str
    label: str


def task(sample: Path, labels: Path, per_clip: int, choices, page: str, extra=()) -> APIRouter:
    """One labelling task: its clips, its answers file, its page; mounted under a prefix."""
    clips = [json.loads(l) for l in sample.open(encoding="utf-8")]
    by_id = {c["id"]: c for c in clips}
    lock = threading.Lock()
    done = defaultdict(set)                       # clip id -> annotators who labelled it
    if labels.exists():
        for l in labels.open(encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]].add(r["who"])
    r = APIRouter()

    @r.get("/", response_class=HTMLResponse)
    def page_():
        return page.replace("%LABELS%", json.dumps(list(choices)))

    @r.get("/next")
    def nxt(who: str):
        with lock:
            mine = sum(who in v for v in done.values())
            todo = [c for c in clips if who not in done[c["id"]] and len(done[c["id"]]) < per_clip]
            # fewest answers first, so every clip reaches its quota before any clip gets more
            todo.sort(key=lambda c: len(done[c["id"]]))
            c = todo[0] if todo else None
            return dict({"id": c and c["id"], "text": c and c.get("text"), "done": mine, "left": len(todo)},
                        **{k: c and c.get(k) for k in extra})

    @r.get("/audio/{cid:path}")
    def audio(cid: str):
        c = by_id.get(cid)
        if not c:
            raise HTTPException(404)
        return FileResponse(c["audio"])

    @r.post("/label")
    def label(x: Label):
        if x.id not in by_id or (x.label not in choices and x.label != "unclear") or not x.who.strip():
            raise HTTPException(400)
        with lock:
            if x.who in done[x.id]:
                return {"ok": True, "dup": True}
            done[x.id].add(x.who)
            with labels.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": x.id, "who": x.who.strip(), "label": x.label, "t": time.time()}) + "\n")
        return {"ok": True}

    @r.get("/progress")
    def progress():
        with lock:
            return {"clips": len(clips), "with_any": sum(bool(v) for v in done.values()),
                    "complete": sum(len(v) >= per_clip for v in done.values()),
                    "annotators": sorted({w for v in done.values() for w in v})}
    return r


def build(sample: Path, labels: Path, per_clip: int, sounds_sample: Path | None = None,
          sounds_labels: Path | None = None):
    app = FastAPI()
    if sounds_sample:
        app.include_router(task(sounds_sample, sounds_labels, per_clip, SOUNDS, SOUNDS_PAGE, ("mark", "clip_s")),
                           prefix="/sounds")
    app.include_router(task(sample, labels, per_clip, LABELS, EMOTION_PAGE))
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--sounds-sample", type=Path, default=None)
    ap.add_argument("--sounds-labels", type=Path, default=None)
    ap.add_argument("--per-clip", type=int, default=3)
    ap.add_argument("--port", type=int, default=17900)
    args = ap.parse_args()
    uvicorn.run(build(args.sample, args.labels, args.per_clip, args.sounds_sample, args.sounds_labels),
                host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
