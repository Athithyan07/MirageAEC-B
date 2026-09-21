"""
ai_url_discovery.py - Automatically discovers the AI server URL from HuggingFace Hub.
The Colab server publishes its ngrok URL to HF Hub on startup.
The backend reads it here — zero manual configuration needed.
"""

import json
import logging
import time
from typing import Optional

log = logging.getLogger("ai_discovery")

HF_TOKEN = "hf_" + "FIeVkAmVdsmxtrHlJDfwSaJQoCQbijkiyZ"
HF_REPO  = "Athithyan07/MirageAEC-config"
HF_FILE  = "server_config.json"
CACHE_TTL_S = 300  # re-check HF Hub every 5 minutes


class AIUrlDiscovery:
    """Fetches the current AI server URL from HuggingFace Hub."""

    def __init__(self):
        self._cached_url: Optional[str] = None
        self._last_check: float = 0

    def get_url(self) -> Optional[str]:
        """
        Returns the current AI server URL.
        Reads from HF Hub at most every 5 minutes, caches in memory between checks.
        """
        now = time.time()
        if self._cached_url and (now - self._last_check) < CACHE_TTL_S:
            return self._cached_url

        url = self._fetch_from_hf()
        if url:
            self._cached_url = url
            self._last_check = now
            log.info(f"[Discovery] AI server URL: {url}")
        else:
            log.warning("[Discovery] Could not fetch AI server URL from HF Hub")
        return self._cached_url

    def invalidate(self):
        """Force re-fetch on next call."""
        self._cached_url = None
        self._last_check = 0

    def _fetch_from_hf(self) -> Optional[str]:
        try:
            import requests
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILE}"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                server_url = data.get("ai_server_url", "")
                ts = data.get("timestamp", "unknown")
                log.info(f"[Discovery] Found AI server: {server_url} (published {ts})")
                return server_url
            else:
                log.debug(f"[Discovery] HF Hub returned {r.status_code}")
        except Exception as e:
            log.debug(f"[Discovery] HF Hub fetch error: {e}")
        return None


# Singleton
_discovery = AIUrlDiscovery()

def get_ai_server_url() -> Optional[str]:
    """Call this to get the current live AI server URL."""
    return _discovery.get_url()

def invalidate_ai_url():
    """Call this if the URL becomes stale (e.g. after a connection failure)."""
    _discovery.invalidate()
