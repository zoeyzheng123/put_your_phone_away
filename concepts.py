
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import uuid4
import time

# ====== Concepts ======
class Concept:
    def __init__(self, name: str):
        self.name = name
    def perform(self, action: str, input_map: Dict[str, Any]) -> Dict[str, Any]:
        fn = getattr(self, action, None)
        if fn is None:
            raise AttributeError(f"{self.name}.{action} not found")
        return fn(**input_map)
    def query(self, qname: str, input_map: Dict[str, Any]) -> Dict[str, Any]:
        if not qname.startswith("_"):
            raise ValueError("Query names must start with '_' to be pure")
        fn = getattr(self, qname, None)
        if fn is None:
            raise AttributeError(f"{self.name}.{qname} not found")
        return fn(**input_map)


# 1) Ticker: external cadence source
class Ticker(Concept):
    def __init__(self, name: str):
        super().__init__(name)
    def tick(self, key: str) -> Dict[str, Any]:
        return {"key": key}

# 2) Camera: webcam capture
import cv2
class Camera(Concept):
    def __init__(self, name: str):
        super().__init__(name)
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_id: Optional[str] = None
        self._frames: Dict[str, Dict[str, Any]] = {}
    def _ensure_open(self, device: str) -> None:
        idx = int(device)
        if self._cap is None:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"Webcam {idx} not found")
            self._cap = cap
            # Optional: set baseline resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    def capture(self, device: str) -> Dict[str, Any]:
        self._ensure_open(device)
        ok, frame = self._cap.read()
        if not ok:
            return {"frame": None}
        fid = str(uuid4())
        self._frames[fid] = {"data": frame, "ts": time.time(), "device": device}
        self._latest_id = fid
        return {"frame": fid}
    def _latest(self) -> Dict[str, Any]:
        return {"frame": self._latest_id} if self._latest_id else {}
    def _getFrame(self, frame: str) -> Dict[str, Any]:
        rec = self._frames.get(frame)
        return {"data": rec["data"], "ts": rec["ts"]} if rec else {}

# 3) Detector: YOLOv8n on COCO (person + cell phone)
from ultralytics import YOLO
class Detector(Concept):
    def __init__(self, name: str, model_name: str = "yolo11n.pt"):
        super().__init__(name)
        self.model = YOLO(model_name)
        # Map COCO ids to names; ultralytics provides model.names
        self.names = self.model.names
        # Precompute class ids for person and cell phone
        self.person_id = [k for k,v in self.names.items() if v == 'person'][0]
        self.phone_id = [k for k,v in self.names.items() if v in ('cell phone','mobile phone')][0]
        self._by_frame: Dict[str, Dict[str, Any]] = {}
        self._det_to_frame: Dict[str, str] = {}  # new: map detection id -> frame id
    def detect(self, frame: str, img: Any, conf: float=0.2, iou: float=0.45) -> Dict[str, Any]:
        # Run on original image; Ultralytics rescales internally and maps back to original coords
        res = self.model.predict(img, conf=conf, iou=iou, verbose=False)
        boxes_out: List[Dict[str, Any]] = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                if cls_id not in (self.person_id, self.phone_id):
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                boxes_out.append({
                    "xyxy": [x1,y1,x2,y2],
                    "cls": self.names[cls_id],
                    "conf": float(b.conf.item())
                })
        det_id = str(uuid4())
        self._by_frame[frame] = {"id": det_id, "boxes": boxes_out}
        self._det_to_frame[det_id] = frame  # record inverse mapping
        return {"detections": det_id}
    def _get(self, detections: str) -> Dict[str, Any]:
        for f, rec in self._by_frame.items():
            if rec["id"] == detections:
                return {"boxes": rec["boxes"]}
        return {"boxes": []}
    def _frameOf(self, detections: str) -> Dict[str, Any]:
        f = self._det_to_frame.get(detections)
        return {"frame": f} if f else {}

# 4) Associator: phone→person per-frame association
import math
class Associator(Concept):
    def __init__(self, name: str):
        super().__init__(name)
        self._by_frame: Dict[str, Dict[str, Any]] = {}
        self._latest_rec: Dict[str, Any] = {}  # cache most recent associations for overlay
    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1, iy1 = max(ax1,bx1), max(ay1,by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw*ih
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        union = max(area_a+area_b-inter, 1)
        return inter/union
    @staticmethod
    def _point_to_rect_dist(cx: float, cy: float, rect: List[int]) -> float:
        x1,y1,x2,y2 = rect
        dx = max(x1 - cx, 0, cx - x2)
        dy = max(y1 - cy, 0, cy - y2)
        return math.hypot(dx, dy)
    def assign(self, frame: str, boxes: List[Dict[str, Any]], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        persons = [b for b in boxes if b["cls"] == "person"]
        phones  = [b for b in boxes if b["cls"] in ("cell phone","mobile phone")]
        # thresholds
        t_iou = float(thresholds.get("tIoU", 0.05))
        t_dist_norm = float(thresholds.get("tDist", 0.2))
        band_lo, band_hi = thresholds.get("torsoBand", [0.3, 0.9])

        # Build candidate matches with a cost (smaller is better). We'll then do a greedy
        # one-to-one assignment so that each phone and each person is used at most once.
        candidates: List[Tuple[float,int,int]] = []  # (cost, hi, pi)
        person_metrics: List[Tuple[List[int], float, List[int]]] = []
        for p in persons:
            px1,py1,px2,py2 = p["xyxy"]
            ph = max(1, py2 - py1)
            band = [px1, int(py1 + band_lo*ph), px2, int(py1 + band_hi*ph)]
            p_diag = math.hypot(px2-px1, py2-py1)
            person_metrics.append((p["xyxy"], max(p_diag, 1.0), band))

        for hi, h in enumerate(phones):
            hx1,hy1,hx2,hy2 = h["xyxy"]
            cx, cy = (hx1+hx2)/2, (hy1+hy2)/2
            for pi, (p_box, p_diag, band) in enumerate(person_metrics):
                inside = (band[0] <= cx <= band[2]) and (band[1] <= cy <= band[3])
                iou = self._iou(h["xyxy"], p_box) if not inside else 0.0
                dist = self._point_to_rect_dist(cx, cy, band)/p_diag
                # accept if plausibly connected
                if inside or iou >= t_iou or dist <= t_dist_norm:
                    # cost prefers inside band, then closer distance, then higher IoU bonus
                    cost = (0.0 if inside else 1.0) + dist - 0.2*iou
                    candidates.append((cost, hi, pi))

        # Greedy one-to-one: sort by cost asc, then take pairs if both phone/person unused
        candidates.sort(key=lambda x: x[0])
        used_phones: set[int] = set()
        used_persons: set[int] = set()
        matches: List[Tuple[int,int]] = []
        for cost, hi, pi in candidates:
            if hi in used_phones or pi in used_persons:
                continue
            used_phones.add(hi)
            used_persons.add(pi)
            matches.append((hi, pi))

        using = [persons[pi] for _, pi in matches]
        assoc_id = str(uuid4())
        rec = {"id": assoc_id, "frame": frame, "using": using, "matches": matches, "persons": persons, "phones": phones}
        self._by_frame[frame] = rec
        self._latest_rec = rec
        return {"associations": assoc_id}
    def _get(self, associations: str) -> Dict[str, Any]:
        for f, rec in self._by_frame.items():
            if rec["id"] == associations:
                return {"frame": rec.get("frame"), "using": rec["using"], "matches": rec["matches"], "persons": rec.get("persons", []), "phones": rec.get("phones", [])}
        return {"frame": None, "using": [], "matches": [], "persons": [], "phones": []}
    def _latest(self) -> Dict[str, Any]:
        rec = self._latest_rec or {}
        return {"frame": rec.get("frame"), "using": rec.get("using", []), "matches": rec.get("matches", []), "persons": rec.get("persons", []), "phones": rec.get("phones", [])}

# 5) Counter
class Counter(Concept):
    def __init__(self, name: str):
        super().__init__(name)
        self.value = 0
    def update(self, frame: str, using: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.value = int(len(using))
        return {"count": self.value}
    def _get(self) -> Dict[str, Any]:
        return {"count": self.value}

# 6) Renderer
import numpy as np
class Renderer(Concept):
    def __init__(self, name: str):
        super().__init__(name)
        self._latest_by_frame: Dict[str, Dict[str, Any]] = {}
    def render(self, frame: str, img: Any, persons: List[Dict[str, Any]], phones: List[Dict[str, Any]], matches: List[Tuple[int,int]], using: List[Dict[str, Any]]) -> Dict[str, Any]:
        ann = img.copy()
        using_set = {pi for _, pi in matches}
        for i, p in enumerate(persons):
            x1,y1,x2,y2 = p["xyxy"]
            color = (0,0,255) if i in using_set else (0,255,0)
            cv2.rectangle(ann, (x1,y1), (x2,y2), color, 2)
            label = "using" if i in using_set else "person"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(ann, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
            cv2.putText(ann, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        for h in phones:
            x1,y1,x2,y2 = h["xyxy"]
            cv2.rectangle(ann, (x1,y1), (x2,y2), (255,0,0), 2)
        for (hi, pi) in matches:
            hx1,hy1,hx2,hy2 = phones[hi]["xyxy"]
            px1,py1,px2,py2 = persons[pi]["xyxy"]
            cx, cy = int((hx1+hx2)/2), int((hy1+hy2)/2)
            cv2.line(ann, (cx,cy), (int((px1+px2)/2), int(py1 + 0.7*(py2-py1))), (255,255,0), 2)
        ok, buf = cv2.imencode('.jpg', ann, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        img_bytes = buf.tobytes() if ok else b''
        rid = str(uuid4())
        self._latest_by_frame[frame] = {"id": rid, "image": img_bytes}
        return {"render": rid}
    def _latestByFrame(self, frame: str) -> Dict[str, Any]:
        rec = self._latest_by_frame.get(frame)
        return {"render": rec["id"]} if rec else {}
    def _getImage(self, render: str) -> Dict[str, Any]:
        for f, rec in self._latest_by_frame.items():
            if rec.get("id") == render:
                return {"image": rec.get("image", b"")}
        return {"image": b""}
    def _overlay(self, img: Any, persons: List[Dict[str, Any]], phones: List[Dict[str, Any]], matches: List[Tuple[int,int]], using: List[Dict[str, Any]]) -> Dict[str, Any]:
        ann = img.copy()
        using_set = {pi for _, pi in matches}
        for i, p in enumerate(persons):
            x1,y1,x2,y2 = p["xyxy"]
            color = (0,0,255) if i in using_set else (0,255,0)
            cv2.rectangle(ann, (x1,y1), (x2,y2), color, 2)
            label = "using" if i in using_set else "person"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(ann, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
            cv2.putText(ann, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        for h in phones:
            x1,y1,x2,y2 = h["xyxy"]
            cv2.rectangle(ann, (x1,y1), (x2,y2), (255,0,0), 2)
        for (hi, pi) in matches:
            hx1,hy1,hx2,hy2 = phones[hi]["xyxy"]
            px1,py1,px2,py2 = persons[pi]["xyxy"]
            cx, cy = int((hx1+hx2)/2), int((hy1+hy2)/2)
            cv2.line(ann, (cx,cy), (int((px1+px2)/2), int(py1 + 0.7*(py2-py1))), (255,255,0), 2)
        ok, buf = cv2.imencode('.jpg', ann, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return {"image": buf.tobytes() if ok else b""}

# 7) API (bootstrap) — request/response with callback (bootstrap) — request/response with callback
class API(Concept):
    def __init__(self, name: str):
        super().__init__(name)
        self._req: Dict[str, Dict[str, Any]] = {}
    def request(self, callback: Callable[[str, Dict[str, Any], str], None], path: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        rid = str(uuid4())
        self._req[rid] = {"callback": callback, "path": path, "method": method, "params": params}
        return {"request": rid}
    def respond(self, request: str, body: Dict[str, Any], contentType: str) -> Dict[str, Any]:
        req = self._req.get(request)
        if req and callable(req.get("callback")):
            try:
                req["callback"](request, body, contentType)
            except Exception:
                pass
        return {"request": request}
