from __future__ import annotations
from typing import Any, Dict, Tuple, Callable
from flask import Flask, Response, jsonify
import threading, time
from concepts import Ticker, Camera, Detector, Associator, Renderer, Counter, API
from engine import Sync, WhenPattern, Engine, Frame
from sync import make_syncs


# ====== Build & Run ======

def build_engine() -> Engine:
    eng = Engine()
    eng.register_concept(Ticker("Ticker"))
    eng.register_concept(Camera("Camera"))
    eng.register_concept(Detector("Detector"))
    eng.register_concept(Associator("Associator"))
    eng.register_concept(Renderer("Renderer"))
    eng.register_concept(Counter("Counter"))
    eng.register_concept(API("API"))
    for s in make_syncs():
        eng.register_sync(s)
    return eng

# Background ticker thread to drive capture at ~3 FPS
class TickerThread(threading.Thread):
    def __init__(self, eng: Engine, key: str = "capture", fps: float = 3.0):
        super().__init__(daemon=True)
        self.eng = eng
        self.key = key
        self.period = 1.0/max(0.1, fps)
        self._stop = threading.Event()
    def run(self):
        while not self._stop.is_set():
            t0 = time.time()
            self.eng.invoke("Ticker","tick", {"key": self.key})
            elapsed = time.time()-t0
            time.sleep(max(0.0, self.period - elapsed))
    def stop(self):
        self._stop.set()


def make_app(eng: Engine) -> Flask:
    app = Flask(__name__)
    def _await_response(invoker: Callable[[], None]) -> Tuple[bytes, str]:
        # Wait for API.respond callback
        ev = threading.Event()
        payload: Dict[str, Any] = {"body": None, "ctype": "application/octet-stream"}
        def cb(req_id: str, body: Dict[str, Any], ctype: str):
            payload["body"], payload["ctype"] = body, ctype
            ev.set()
        # Fire API.request → syncs will eventually call API.respond which triggers cb
        invoker_cb = lambda: eng.invoke("API","request", {"callback": cb, "path": path, "method": method, "params": {}})
        invoker_cb()  # but we need path/method from outer scope; handled in each route
        ev.wait(timeout=2.0)
        return payload.get("body"), payload.get("ctype")

    @app.get("/frame.jpg")
    def frame_jpg():
        ev = threading.Event()
        out = {"data": b"", "ctype": "image/jpeg"}
        def cb(req_id: str, body: Dict[str, Any], ctype: str):
            out["data"], out["ctype"] = body.get("image", b""), ctype
            ev.set()
        eng.invoke("API","request", {"callback": cb, "path": "/frame.jpg", "method": "GET", "params": {}})
        ev.wait(timeout=2.0)
        return Response(out["data"], mimetype=out["ctype"])  # can be empty until first render

    @app.get("/count")
    def count_json():
        ev = threading.Event()
        out = {"data": {}, "ctype": "application/json"}
        def cb(req_id: str, body: Dict[str, Any], ctype: str):
            out["data"], out["ctype"] = body, ctype
            ev.set()
        eng.invoke("API","request", {"callback": cb, "path": "/count", "method": "GET", "params": {}})
        ev.wait(timeout=2.0)
        return jsonify(out["data"])  # {"using": N}

    @app.get("/")
    def index_page():
        # Simple page that shows a stable MJPEG stream + live count (no flicker)
        html = """
        <html>
          <head>
            <meta charset='utf-8'>
            <meta name='viewport' content='width=device-width, initial-scale=1'>
            <title>Classroom Phone Monitor</title>
            <style>
              body{margin:0;background:#0f1115;color:#e7ecf5;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial}
              .wrap{max-width:1100px;margin:24px auto;padding:0 16px}
              .badge{background:#151924;border:1px solid #22283a;border-radius:12px;padding:8px 12px;display:inline-flex;gap:10px;align-items:center;margin-bottom:12px}
              .dot{width:10px;height:10px;border-radius:50%;background:#3b82f6;box-shadow:0 0 12px #3b82f6}
              .stage{background:#000;border:1px solid #22283a;border-radius:12px;overflow:hidden}
              img{display:block;width:100%;height:auto}
            </style>
          </head>
          <body>
            <div class='wrap'>
              <div class='badge'>
                <div class='dot'></div>
                <div>Students using phone: <span id='count'>0</span></div>
              </div>
              <div class='stage'>
                <img src='/stream' alt='live stream'/>
              </div>
            </div>
            <script>
              async function refreshCount(){
                try{ const r = await fetch('/count', {cache:'no-store'}); const j = await r.json();
                      if(typeof j.using === 'number') document.getElementById('count').textContent = j.using; }
                catch(e){}
              }
              setInterval(refreshCount, 1000);
              refreshCount();
            </script>
          </body>
        </html>
        """
        return Response(html, mimetype="text/html")
    @app.get("/stream")
    def stream():
        from time import sleep
        import cv2, numpy as np
        boundary = b"--frame"
        def gen():
            while True:
                cap_id = eng.query("Camera","_latest").get("frame")
                if cap_id:
                    cam = eng.query("Camera","_getFrame", frame=cap_id)
                    raw = cam.get("data")
                    if isinstance(raw, np.ndarray) and raw.size > 0:
                        # Overlay latest detections (if any) on the live frame
                        latest = eng.query("Associator","_latest")
                        persons = latest.get("persons", [])
                        phones = latest.get("phones", [])
                        matches = latest.get("matches", [])
                        using = latest.get("using", [])
                        frame_bytes = b""
                        if persons or phones:
                            over = eng.query("Renderer","_overlay", img=raw, persons=persons, phones=phones, matches=matches, using=using).get("image")
                            frame_bytes = over if over else b""
                        if not frame_bytes:
                            ok, buf = cv2.imencode('.jpg', raw, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            frame_bytes = buf.tobytes() if ok else b""
                        if frame_bytes:
                            yield (boundary + b"\r\n"
                                   + b"Content-Type: image/jpeg\r\n"
                                   + b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                                   + frame_bytes + b"\r\n")
                sleep(0.2)  # ~5 FPS visual stream
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app
