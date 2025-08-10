# ====== Synchronizations ======
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import uuid4
import threading, time
from engine import Sync, WhenPattern, Engine, Frame



def make_syncs() -> List[Sync]:
    syncs: List[Sync] = []
    # On Ticker.tick("capture") → Camera.capture
    def tick_to_capture_where(eng, fr):
        tick = fr.last("Ticker", "tick")
        if tick is None:
            return False
        cap = fr.last("Camera", "capture")
        # Fire only once per tick (idempotent): allow if no capture yet or the last capture happened before this tick
        return (cap is None) or (cap.t < tick.t)

    syncs.append(Sync(
        name="TickToCapture",
        when=[WhenPattern(concept="Ticker", action="tick", inputs={"key": "capture"})],
        where=tick_to_capture_where,
        then=lambda fr: [("Camera", "capture", {"device": "0"})],
    ))
    # After capture → detect
    def DetectOnCapture_where(eng: Engine, fr: Frame) -> bool:
        cap = fr.last("Camera","capture")
        if not cap or not cap.output or not cap.output.get("frame"):
            return False
        frame_id = cap.output["frame"]
        img = eng.query("Camera","_getFrame", frame=frame_id).get("data")
        # Robustly validate that img is a non-empty numpy array (avoid ambiguous comparisons)
        try:
            import numpy as np
            if img is None:
                return False
            if not isinstance(img, np.ndarray):
                return False
            if img.size == 0:
                return False
        except Exception:
            return False
        fr.bind("frame_id", frame_id)
        fr.bind("img", img)
        return True
    def DetectOnCapture_then(fr: Frame):
        return [("Detector","detect", {"frame": fr.get("frame_id"), "img": fr.get("img"), "conf": 0.25, "iou": 0.45})]
    # Every 10s (Ticker.tick("detect")) → run detect on the latest frame
    def TickToDetect_where(eng: Engine, fr: Frame) -> bool:
        # get latest frame from camera
        cap_id = eng.query("Camera","_latest").get("frame")
        if not cap_id:
            return False
        cam = eng.query("Camera","_getFrame", frame=cap_id)
        img = cam.get("data")
        try:
            import numpy as np
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                return False
        except Exception:
            return False
        fr.bind("frame_id", cap_id)
        fr.bind("img", img)
        return True
    def TickToDetect_then(fr: Frame):
        return [("Detector","detect", {"frame": fr.get("frame_id"), "img": fr.get("img"), "conf": 0.25, "iou": 0.45})]
    syncs.append(Sync(
        name="TickToDetect",
        when=[WhenPattern(concept="Ticker", action="tick", inputs={"key":"detect"})],
        where=TickToDetect_where,
        then=TickToDetect_then,
    ))

    # After detect → associate
    def Associate_where(eng: Engine, fr: Frame) -> bool:
        det = fr.last("Detector","detect")
        if not det:
            return False
        detections = det.output.get("detections")
        boxes = eng.query("Detector","_get", detections=detections).get("boxes", [])
        frame_id = eng.query("Detector","_frameOf", detections=detections).get("frame")
        if frame_id is None:
            return False
        fr.bind("frame_id", frame_id)
        fr.bind("boxes", boxes)
        return True
    def Associate_then(fr: Frame):
        return [("Associator","assign", {"frame": fr.get("frame_id"), "boxes": fr.get("boxes"), "thresholds": {"torsoBand":[0.3,0.9],"tIoU":0.05,"tDist":0.2}})]
    syncs.append(Sync(
        name="AssociateAfterDetect",
        when=[WhenPattern(concept="Detector", action="detect", inputs={}, outputs={"detections":"det"})],
        where=Associate_where,
        then=Associate_then,
    ))
    # After associate → render + count
    def Render_where(eng: Engine, fr: Frame) -> bool:
        assoc = fr.last("Associator","assign")
        if not assoc:
            return False
        associations = assoc.output.get("associations")
        got = eng.query("Associator","_get", associations=associations)
        frame_id = got.get("frame")
        img = eng.query("Camera","_getFrame", frame=frame_id).get("data") if frame_id else None
        if img is None:
            return False
        fr.bind("frame_id", frame_id)
        fr.bind("img", img)
        fr.bind("persons", got.get("persons", []))
        fr.bind("phones", got.get("phones", []))
        fr.bind("matches", got.get("matches", []))
        fr.bind("using", got.get("using", []))
        return True
    def Render_then(fr: Frame):
        return [
            ("Renderer","render", {"frame": fr.get("frame_id"), "img": fr.get("img"), "persons": fr.get("persons"), "phones": fr.get("phones"), "matches": fr.get("matches"), "using": fr.get("using")}),
            ("Counter","update", {"frame": fr.get("frame_id"), "using": fr.get("using")}),
        ]
    syncs.append(Sync(
        name="RenderAfterAssociate",
        when=[WhenPattern(concept="Associator", action="assign", inputs={}, outputs={"associations":"as"})],
        where=Render_where,
        then=Render_then,
    ))
    # API routes: /frame.jpg
    def GetFrame_where(eng: Engine, fr: Frame) -> bool:
        req = fr.last("API","request")
        cap = eng.query("Camera","_latest").get("frame")
        if not (req and cap):
            return False
        # Prefer the latest rendered frame, but fall back to raw camera frame if render not ready yet
        render_id = eng.query("Renderer","_latestByFrame", frame=cap).get("render")
        if render_id:
            img = eng.query("Renderer","_getImage", render=render_id).get("image")
            if img and len(img) > 0:
                fr.bind("img", img)
                fr.bind("req_id", req.output.get("request"))
                return True
        # Fallback: encode the raw camera frame as JPEG so the page at least shows something
        cam = eng.query("Camera","_getFrame", frame=cap)
        raw = cam.get("data")
        try:
            import cv2, numpy as np
            if isinstance(raw, np.ndarray) and raw.size > 0:
                ok, buf = cv2.imencode('.jpg', raw, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    fr.bind("img", buf.tobytes())
                    fr.bind("req_id", req.output.get("request"))
                    return True
        except Exception:
            pass
        return False
    def GetFrame_then(fr: Frame):
        return [("API","respond", {"request": fr.get("req_id"), "body": {"image": fr.get("img")}, "contentType": "image/jpeg"})]
    syncs.append(Sync(
        name="GetFrame",
        when=[WhenPattern(concept="API", action="request", inputs={"path":"/frame.jpg","method":"GET","params":"$p"}, outputs={"request":"r"})],
        where=GetFrame_where,
        then=GetFrame_then,
    ))
    # API routes: /count
    def GetCount_where(eng: Engine, fr: Frame) -> bool:
        req = fr.last("API","request")
        n = eng.query("Counter","_get").get("count", 0)
        fr.bind("n", n)
        fr.bind("req_id", req.output.get("request"))
        return True
    def GetCount_then(fr: Frame):
        return [("API","respond", {"request": fr.get("req_id"), "body": {"using": fr.get("n")}, "contentType": "application/json"})]
    syncs.append(Sync(
        name="GetCount",
        when=[WhenPattern(concept="API", action="request", inputs={"path":"/count","method":"GET","params":"$p"}, outputs={"request":"r"})],
        where=GetCount_where,
        then=GetCount_then,
    ))
    # API routes: /
    def Index_where(eng: Engine, fr: Frame) -> bool:
        req = fr.last("API","request")
        n = eng.query("Counter","_get").get("count", 0)
        fr.bind("n", n)
        fr.bind("req_id", req.output.get("request"))
        return True
    def Index_then(fr: Frame):
        html_template = """
        <html>
          <head>
            <meta charset='utf-8'>
            <meta name='viewport' content='width=device-width, initial-scale=1'>
            <title>Classroom Phone Monitor</title>
            <style>
              :root { --bg:#0f1115; --card:#151924; --muted:#96a0b5; --accent:#3b82f6; }
              body{margin:0;background:var(--bg);color:#e7ecf5;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,'Noto Sans','Apple Color Emoji','Segoe UI Emoji';}
              .wrap{max-width:1100px;margin:24px auto;padding:0 16px}
              .header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
              .badge{background:var(--card);border:1px solid #22283a;border-radius:12px;padding:8px 12px;display:inline-flex;gap:10px;align-items:center}
              .dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent)}
              .count{font-weight:600}
              .muted{color:var(--muted);font-size:14px}
              .stage{position:relative;background:#000;border:1px solid #22283a;border-radius:12px;overflow:hidden}
              .stage img{display:block;width:100%;height:auto;}
              /* Crossfade with two layers */
              .layer{position:absolute;inset:0;transition:opacity .25s ease;}
              .layer.hidden{opacity:0}
              .layer.visible{opacity:1}
            </style>
          </head>
          <body>
            <div class='wrap'>
              <div class='header'>
                <div class='badge'>
                  <div class='dot'></div>
                  <div>
                    <div class='count'>Students using phone: <span id='count'>__N__</span></div>
                    <div class='muted'>Updating ~2–4×/sec</div>
                  </div>
                </div>
              </div>
              <div class='stage' id='stage'>
                <img id='imgA' class='layer visible' src='/frame.jpg?ts=__TS__' alt='frame A'/>
                <img id='imgB' class='layer hidden'  src='' alt='frame B'/>
              </div>
            </div>
            <script>
              const imgA = document.getElementById('imgA');
              const imgB = document.getElementById('imgB');
              const countEl = document.getElementById('count');
              let showingA = true;
              function swapNext(){
                const next = showingA ? imgB : imgA;
                const ts = Date.now();
                next.src = '/frame.jpg?ts=' + ts;
                next.onload = () => {
                  // crossfade
                  if(showingA){ imgA.classList.add('hidden'); imgA.classList.remove('visible');
                                 imgB.classList.add('visible'); imgB.classList.remove('hidden'); }
                  else{ imgB.classList.add('hidden'); imgB.classList.remove('visible');
                        imgA.classList.add('visible'); imgA.classList.remove('hidden'); }
                  showingA = !showingA;
                };
              }
              async function refreshCount(){
                try{
                  const r = await fetch('/count', {cache:'no-store'});
                  const j = await r.json();
                  if(typeof j.using === 'number') countEl.textContent = j.using;
                }catch(_){/* noop */}
              }
              // cadence
              setInterval(swapNext, 400);  // ~2.5 FPS on the page, smoother than full refresh
              setInterval(refreshCount, 500);
            </script>
          </body>
        </html>
        """
        html = html_template.replace("__N__", str(fr.get('n'))).replace("__TS__", str(int(time.time()*1000)))
        return [("API","respond", {"request": fr.get("req_id"), "body": {"html": html}, "contentType": "text/html"})]
    syncs.append(Sync(
        name="IndexPage",
        when=[WhenPattern(concept="API", action="request", inputs={"path":"/","method":"GET","params":"$p"}, outputs={"request":"r"})],
        where=Index_where,
        then=Index_then,
    ))
    return syncs
