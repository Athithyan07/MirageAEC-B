"""
ai_client.py - HTTP client that calls the MirageAEC Colab AI server.
Wraps all 5 model endpoints with retry logic, timeout handling,
and graceful fallback to heuristic methods when AI server is offline.
"""

import io, base64, time, logging
from typing import Dict, List, Any, Optional
import requests
from PIL import Image

log = logging.getLogger("ai_client")


def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class AIClient:
    """
    Communicates with the Colab AI inference server.
    Automatically falls back to None if server is unreachable.
    """

    def __init__(self, base_url: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.timeout = timeout
        self._online: Optional[bool] = None

    @property
    def available(self) -> bool:
        if not self.base_url:
            return False
        if self._online is None:
            self._online = self._ping()
        return self._online

    def _ping(self) -> bool:
        try:
            headers = {"ngrok-skip-browser-warning": "true"}
            r = requests.get(f"{self.base_url}/health", headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                log.info(f"[AI] Server online: {data.get('models', [])} | VRAM={data.get('vram_used_gb',0)}GB")
                return True
        except Exception as e:
            log.warning(f"[AI] Server offline: {e}")
        return False

    def _post(self, endpoint: str, payload: dict) -> Optional[dict]:
        if not self.base_url:
            return None
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            headers = {"ngrok-skip-browser-warning": "true"}
            t0 = time.time()
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            elapsed = time.time() - t0
            log.info(f"[AI] {endpoint} -> {elapsed:.1f}s")
            self._online = True
            return r.json()
        except requests.exceptions.Timeout:
            log.error(f"[AI] {endpoint} timed out after {self.timeout}s")
            self._online = False
            # Ngrok URL may have changed — force re-discovery on next request
            reset_client()
        except Exception as e:
            log.error(f"[AI] {endpoint} error: {e}")
            self._online = False
            # If connection refused, Colab may have restarted — force re-discovery
            if "Connection" in str(e) or "refused" in str(e).lower():
                reset_client()
        return None

    # ── Stage 2: GroundingDINO detection ──────────────────────
    def detect_elements(
        self,
        image_bytes: bytes,
        prompt: str = "wall . door . window . column . staircase . bathroom . kitchen . room",
        threshold: float = 0.30
    ) -> Optional[List[Dict]]:
        """Returns list of {label, bbox:[x1,y1,x2,y2], conf} in pixel coords."""
        result = self._post("detect", {
            "image_b64": _encode_image(image_bytes),
            "prompt": prompt,
            "threshold": threshold
        })
        if result and "detections" in result:
            return result["detections"]
        return None

    # ── Stage 3: SAM2 segmentation ────────────────────────────
    def segment_masks(
        self,
        image_bytes: bytes,
        boxes: List[List[float]],
        labels: List[str]
    ) -> Optional[List[Dict]]:
        """Returns list of {label, polygon_px, bbox, score} from SAM2."""
        result = self._post("segment", {
            "image_b64": _encode_image(image_bytes),
            "boxes": boxes,
            "labels": labels
        })
        if result and "masks" in result:
            return result["masks"]
        return None

    # ── Stage 4: Florence-2 room classification ───────────────
    def classify_rooms(
        self,
        image_bytes: bytes,
        room_crops: List[Dict]
    ) -> Optional[List[Dict]]:
        """Returns list of {id, bbox, room_type, caption} per room."""
        result = self._post("classify-rooms", {
            "image_b64": _encode_image(image_bytes),
            "room_crops": room_crops
        })
        if result and "classified_rooms" in result:
            return result["classified_rooms"]
        return None

    # ── Stage 5: DeepSeek-R1 MEP calculations ─────────────────
    def calculate_mep(
        self,
        rooms: List[Dict],
        climate_zone: str = "mixed_humid",
        building_type: str = "residential",
        voltage: int = 230,
        phases: int = 1
    ) -> Optional[Dict]:
        """Returns full MEP engineering specs per room (HVAC/Elec/Plumbing/Fire)."""
        result = self._post("calculate-mep", {
            "rooms": rooms,
            "climate_zone": climate_zone,
            "building_type": building_type,
            "voltage": voltage,
            "phases": phases
        })
        if result and "mep_specs" in result:
            return result["mep_specs"]
        return None

    # ── Stage 6: Qwen2.5-VL equipment placement ───────────────
    def place_equipment(
        self,
        image_bytes: bytes,
        rooms: List[Dict],
        plan_width_m: float,
        plan_depth_m: float
    ) -> Optional[Dict]:
        """Returns optimal {ahu, electrical_db, plumbing_riser, fire_main} coordinates."""
        result = self._post("place-equipment", {
            "image_b64": _encode_image(image_bytes),
            "rooms": rooms,
            "plan_width_m": plan_width_m,
            "plan_depth_m": plan_depth_m
        })
        if result and "equipment_locations" in result:
            return result["equipment_locations"]
        return None

    # ── Stage 8: DeepSeek-R1 compliance check ─────────────────
    def check_compliance(
        self,
        mep_layout: Dict
    ) -> Optional[Dict]:
        """Returns compliance report against ASHRAE/NEC/UPC/NFPA."""
        result = self._post("check-compliance", {
            "mep_layout": mep_layout
        })
        if result and "compliance_report" in result:
            return result["compliance_report"]
        return None


# Singleton — import this in all services
_client: Optional[AIClient] = None

def get_ai_client() -> AIClient:
    """
    Returns an AIClient pointed at the current Colab AI server URL.
    URL is auto-discovered from HuggingFace Hub — no manual config needed.
    Falls back to static AI_SERVER_URL in config if HF Hub is unreachable.
    """
    global _client

    # Try to auto-discover URL from HF Hub (Colab publishes it there on startup)
    discovered_url = ""
    try:
        from services.ai_url_discovery import get_ai_server_url
        discovered_url = get_ai_server_url() or ""
    except Exception:
        pass

    # Fall back to static config if discovery fails
    if not discovered_url:
        try:
            from config import settings
            discovered_url = settings.AI_SERVER_URL or ""
        except Exception:
            pass

    # Re-create client if URL changed (Colab restarted with new ngrok URL)
    if _client is None or _client.base_url != discovered_url.rstrip("/"):
        if discovered_url:
            log.info(f"[AI] Connecting to AI server: {discovered_url}")
        _client = AIClient(
            base_url=discovered_url,
            timeout=120
        )

    return _client

def reset_client():
    """Force re-discovery of AI server URL on next request."""
    global _client
    _client = None
    try:
        from services.ai_url_discovery import invalidate_ai_url
        invalidate_ai_url()
    except Exception:
        pass

