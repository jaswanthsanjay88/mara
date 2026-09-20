"""
Thread-safe LRU Document Prefix KV-Cache for Mara.
Allows sub-5ms decision evaluation against pre-computed document states.
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Optional, Tuple, List
import torch


class PrefixKVCache:
    """Thread-safe LRU cache for document prefix KV states."""
    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self.lock = threading.Lock()
        self._cache: OrderedDict[str, Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def hash_state(state_text: str) -> str:
        """Computes deterministic SHA-256 hash of document state text."""
        return hashlib.sha256(state_text.strip().encode("utf-8")).hexdigest()

    def get(self, state_hash: str) -> Optional[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]]:
        with self.lock:
            if state_hash in self._cache:
                self.hits += 1
                self._cache.move_to_end(state_hash)
                kv_cache, state_len = self._cache[state_hash]
                # Return cloned tensors for branch isolation
                cloned = [(k.clone(), v.clone()) for k, v in kv_cache]
                return cloned, state_len
            self.misses += 1
            return None

    def put(
        self,
        state_hash: str,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        state_len: int,
    ) -> None:
        with self.lock:
            if state_hash in self._cache:
                self._cache.move_to_end(state_hash)
            else:
                if len(self._cache) >= self.max_entries:
                    self._cache.popitem(last=False)
            # Store detached, cloned tensors in CPU or persistent device memory
            stored = [(k.detach().clone(), v.detach().clone()) for k, v in kv_cache]
            self._cache[state_hash] = (stored, state_len)

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }
