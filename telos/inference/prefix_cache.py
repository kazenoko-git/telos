"""Bounded exact-prefix KV cache reuse for multi-turn and shared-prompt inference."""

from collections import OrderedDict


def _common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def slice_kv_cache(kv_cache, length: int):
    """Return a non-destructive prefix view of a PyTorch, MLX, or Qwen3 KV cache."""
    sliced = []
    for layer in kv_cache:
        truncate = getattr(layer, "truncate", None)
        if truncate is not None:
            sliced.append(truncate(length))
        else:
            keys, values = layer
            sliced.append((keys[:, :, :length], values[:, :, :length]))
    return sliced


class PrefixKVCache:
    """Bounded LRU cache storing immutable prompt KV prefixes by exact token IDs."""

    def __init__(self, max_entries: int = 4):
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative.")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], object] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, token_ids: list[int] | tuple[int, ...]):
        """Find the longest reusable prefix strictly shorter than token_ids.

        At least one suffix token is always left uncached so the model computes
        the final prompt hidden state and logits without storing hidden states.
        """
        if self.max_entries == 0 or len(token_ids) <= 1 or not self._entries:
            return 0, None
        query = tuple(token_ids)
        max_reusable = len(query) - 1
        best_key = None
        best_len = 0
        for key in self._entries:
            matched = min(_common_prefix_length(key, query), max_reusable)
            if matched > best_len:
                best_len = matched
                best_key = key
                if best_len == max_reusable:
                    break
        if best_key is None or best_len == 0:
            return 0, None
        self._entries.move_to_end(best_key)
        return best_len, slice_kv_cache(self._entries[best_key], best_len)

    def put(self, token_ids: list[int] | tuple[int, ...], kv_cache) -> None:
        """Store a prompt's KV cache for future exact-prefix reuse."""
        if self.max_entries == 0 or not token_ids or kv_cache is None:
            return
        key = tuple(token_ids)
        self._entries[key] = slice_kv_cache(kv_cache, len(key))
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
