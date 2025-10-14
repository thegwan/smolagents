from ast import Tuple
import copy
import json
import re
import time
from typing import Any, Optional, OrderedDict

from smolagents.src.smolagents.monitoring import TokenUsage


class AgenticPlanCache:
    """
    Plan cache that follows "Agentic Plan Caching" (keyword-based exact match).
    - Keys are keywords (strings) extracted by the small model.
    - A cache hit requires an exact keyword match.
    - Insertions store a structured plan template + metadata.
    - Uses LRU eviction via an OrderedDict `access_order`.
    """

    def __init__(self, cache_size_cap: int = 1024, persist_path: Optional[str] = None):
        """
        Args:
            cache_size_cap: max number of entries.
            persist_path: optional file path to persist cache as JSON.
            enable_embeddings: optional (disabled by default) — if True, you can enable semantic fallback.
        """
        self.cache_size_cap = int(cache_size_cap)
        self.cache = {}  # keyword -> plan_entry (dict)
        self.access_order = OrderedDict()  # keyword -> None (for LRU)
        self.cachegen_overhead = {"input": 0, "output": 0}
        self.persist_path = persist_path

        # if persistence path provided, try to load
        if self.persist_path:
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # restore cache and access_order from file (if present)
                self.cache = {k: v for k, v in loaded.get("cache", {}).items()}
                # ensure access_order contains same keys (oldest first)
                for k in loaded.get("access_order", []):
                    if k in self.cache:
                        self.access_order[k] = None
            except Exception:
                # start with empty cache if load fails
                self.cache = {}
                self.access_order = OrderedDict()

    # ------------------------
    # Utilities
    # ------------------------
    @staticmethod
    def normalize_keyword(raw_keyword: str) -> str:
        """Normalize raw keyword text to canonical cache key form."""
        if raw_keyword is None:
            return ""
        s = str(raw_keyword).strip().lower()

        # Remove parentheses and content inside them
        s = re.sub(r"\s*\(.*?\)\s*", " ", s)

        # Remove punctuation except whitespace
        s = re.sub(r"[^\w\s]", " ", s)

        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _save(self):
        """Persist cache if a path is provided."""
        if not self.persist_path:
            return
        try:
            payload = {
                "cache": self.cache,
                "access_order": list(self.access_order.keys()),
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            # Swallow persistence errors (non-fatal)
            pass

    def _touch(self, keyword: str):
        """Mark entry as recently used (LRU)."""
        if keyword in self.access_order:
            self.access_order.move_to_end(keyword)
        else:
            self.access_order[keyword] = None

    def insert(self, keyword: str, plan_entry: dict[str, Any]) -> None:
        """
        Insert a plan template under the provided keyword.
        plan_entry should be a JSON-serializable dict with fields like:
           {"keyword": kw, "plan_template": "..."}
        """
        if not keyword:
            return
        keyword = self.normalize_keyword(keyword)
        # store
        self.cache[keyword] = plan_entry
        self._touch(keyword)
        # evict if needed
        while len(self.cache) > self.cache_size_cap:
            self.evict("LRU")
        # persist
        self._save()

    def evict(self, strategy: str = "LRU"):
        """Evict entries from cache. Default LRU."""
        if strategy == "LRU":
            if not self.access_order:
                return
            lru_key, _ = self.access_order.popitem(last=False)
            try:
                del self.cache[lru_key]
            except KeyError:
                pass
        else:
            raise NotImplementedError(f"Unknown eviction strategy: {strategy}")
        self._save()

    def get_exact(self, keyword: str) -> Optional[dict]:
        """Return cached entry for exact keyword match, or None."""
        if not keyword:
            return None
        key = self.normalize_keyword(keyword)
        entry = self.cache.get(key)
        if entry is not None:
            self._touch(key)
        return copy.deepcopy(entry) if entry is not None else None


    def extract_keyword(self, small_model, task_text: str) -> Tuple[str, TokenUsage]:
        """
        Use the provided small_model to extract a single short keyword for the task.
        small_model should implement `.generate(messages)` and return an object with `.content` and `.token_usage`.
        Returns: (filtered_keyword, token_usage)
        """
        # short prompt that asks the small model for a single keyword only (paper)
        prompt = [
            {"role": "system", "content": "You are a compact keyword extractor. Reply with a single short keyword or short phrase (1-3 words) that captures the core semantic intent of the user's task. Reply with the keyword(s) only."},
            {"role": "user", "content": task_text},
        ]
        # call small model
        resp = small_model.generate(prompt)
        # extract content + token usage
        raw = getattr(resp, "content", "") or ""
        token_usage = getattr(resp, "token_usage", None) or TokenUsage(0, 0)
        filtered = self.normalize_keyword(raw)
        return filtered, token_usage

    def extract_keyword_and_search_for_hit(self, small_model, task_text: str, threshold: Optional[str] = None) -> Tuple[str, Optional[dict], TokenUsage, dict]:
        """
        Runs keyword extraction with the small model and does an exact-match cache lookup.
        Returns: (keyword, cached_entry_or_None, accumulated_token_usage, latency_info)
        - threshold is ignored for default exact-match behavior; kept for backwards compat / experiments.
        """
        latency_info: dict[str, float] = {}
        total_tokens = TokenUsage(0, 0)

        t0 = time.time()
        kw, kw_tokens = self.extract_keyword(small_model, task_text)
        t1 = time.time()
        # accumulate tokens
        total_tokens.input_tokens += getattr(kw_tokens, "input_tokens", 0)
        total_tokens.output_tokens += getattr(kw_tokens, "output_tokens", 0)

        latency_info["keyword_extraction"] = t1 - t0

        # per paper: exact-match lookup on keyword
        lookup_t0 = time.time()
        entry = self.get_exact(kw) if kw else None
        lookup_t1 = time.time()
        latency_info["cache_lookup"] = lookup_t1 - lookup_t0

        return kw, copy.deepcopy(entry) if entry is not None else None, total_tokens, latency_info


    def induce_and_insert(self, large_model, keyword: str, conversation_trace: dict[str, Any]) -> TokenUsage:
        """
        Use the large model to distill a structured, reusable plan template from a successful conversation trace.
        - keyword: the canonical keyword under which to store the template.
        - conversation_trace: a dict containing the dialogue/trace that led to the final solution.
        Returns the token usage (TokenUsage) consumed during cache induction (to be accounted for separately).
        """
        if not keyword:
            keyword = "unknown"
        key = self.normalize_keyword(keyword)

        # Extract a plain plan string from the conversation trace
        plan_string = self._extract_plan_template_as_text(conversation_trace)

        # Construct a safe instruction to the large model, with json format
        induction_prompt = (
            "You will be given the full planning/action trace of an agent solving a task. "
            "Produce a reusable plan template that abstracts away problem-specific details (names, dates, numeric values) "
            "and preserves the planning structure (facts survey + step-by-step plan). "
            "Return only a JSON object with keys: 'keyword', 'task', 'plan_template' (string). "
            "Make sure strings are JSON-safe and do not include execution-specific values."
            "\n\nTrace:\n"
            + plan_string
        )
        # call large_model
        resp = large_model.generate([{"role": "user", "content": induction_prompt}], response_format={"type": "json_object"})
        token_usage = getattr(resp, "token_usage", None) or TokenUsage(0, 0)

        # update induction overhead accounting
        try:
            self.cachegen_overhead["input"] += token_usage.input_tokens
            self.cachegen_overhead["output"] += token_usage.output_tokens
        except Exception:
            pass

        # parse content as JSON (robust)
        content = getattr(resp, "content", "") or ""
        try:
            parsed = self._extract_json(content)
        except Exception:
            # fallback: build naive template
            parsed = {
                "keyword": key,
                "task": plan_string[:300],
                "plan_template": plan_string,
            }

        # Validate basic structure and fix if needed
        parsed = self._validate_and_fix_template(parsed)

        # Insert into cache under the keyword (paper: keyword-based)
        self.insert(key, parsed)

        return token_usage

    def _extract_plan_template_as_text(self, conversation_trace: dict[str, Any]) -> str:
        """
        Extract a plain plan string from the conversation trace.
        """
        # if isinstance(conversation_trace, dict):
        #     if "plan" in conversation_trace and isinstance(conversation_trace["plan"], str):
        #         return conversation_trace["plan"]
        #     if "conversation" in conversation_trace:
        #         parts = []
        #         for entry in conversation_trace["conversation"]:
        #             out = entry.get("output") or entry.get("message") or entry.get("content")
        #             if out:
        #                 parts.append(str(out))
        #         return "\n\n".join(parts)
        # # otherwise stringify the traec
        return json.dumps(conversation_trace, ensure_ascii=False, indent=2)

    def _validate_and_fix_template(self, template: dict[str, Any]) -> dict:
        """Ensure required fields exist and apply minimal fixes."""
        if "keyword" not in template:
            template["keyword"] = "unknown"
        if "task" not in template:
            template["task"] = template.get("task", template.get("keyword", ""))
        if "plan_template" not in template:
            template["plan_template"] = ""
        # ensure keyword stored is normalized
        template["keyword"] = self.normalize_keyword(template["keyword"])
        return template

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Robust extraction of a JSON object from arbitrarily formatted text."""
        # try to parse directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # try to extract last JSON block in code fences
        code_block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        # try to extract last {...} substring
        brace_match = re.search(r"(\{(?:.|\n)*\})", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass
        # give up
        raise ValueError("Failed to extract JSON from text")

