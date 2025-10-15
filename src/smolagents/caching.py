
import copy
import json
import re
import time
from typing import Any, Optional, OrderedDict

from smolagents.monitoring import TokenUsage


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


    def extract_keyword(self, small_model, task_text: str, k: int = 1) -> (list[str], TokenUsage):
        """
        Use the provided small_model to extract k short keywords for the task.
        small_model should implement `.generate(messages)` and return an object with `.content` and `.token_usage`.
        Returns: (list of filtered_keywords, token_usage)
        """
        if k == 1:
            # short prompt that asks the small model for a single keyword only 
            prompt = [
                {"role": "system", "content": "You are a compact keyword extractor. Reply with a single short keyword or short phrase (1-3 words) that captures the core semantic intent of the user's task. Reply with the keyword(s) only. Do not include any specific details about the task, only the core semantic intent."},
                {"role": "user", "content": task_text},
            ]
        else:
            # prompt for multiple keywords
            prompt = [
                {"role": "system", "content": f"You are a compact keyword extractor. Reply with exactly {k} different short keywords or short phrases (1-3 words each) that capture different aspects of the core semantic intent of the user's task. Each keyword should be on a separate line. Reply with the keywords only. Do not include any specific details about the task, only the core semantic intent."},
                {"role": "user", "content": task_text},
            ]
        
        # call small model
        resp = small_model.generate(prompt)
        # extract content + token usage
        raw = getattr(resp, "content", "") or ""
        token_usage = getattr(resp, "token_usage", None) or TokenUsage(0, 0)
        
        if k == 1:
            filtered = self.normalize_keyword(raw)
            return [filtered] if filtered else [], token_usage
        else:
            # Split by lines and normalize each keyword
            keywords = []
            for line in raw.strip().split('\n'):
                line = line.strip()
                if line:
                    normalized = self.normalize_keyword(line)
                    if normalized and normalized not in keywords:
                        keywords.append(normalized)
            return keywords[:k], token_usage

    def extract_keyword_and_search_for_hit(self, small_model, task_text: str, k: int = 1) -> (str, Optional[dict], TokenUsage, dict, list[str]):
        """
        Runs keyword extraction with the small model and does an exact-match cache lookup.
        Returns: (best_keyword, cached_entry_or_None, accumulated_token_usage, latency_info, all_keywords)
        - k: number of keywords to extract and search for
        """
        latency_info: dict[str, float] = {}
        total_tokens = TokenUsage(0, 0)

        t0 = time.time()
        keywords, kw_tokens = self.extract_keyword(small_model, task_text, k)
        print(f"$$$$$ Extracted keywords: {keywords}")
        t1 = time.time()
        # accumulate tokens
        total_tokens.input_tokens += getattr(kw_tokens, "input_tokens", 0)
        total_tokens.output_tokens += getattr(kw_tokens, "output_tokens", 0)

        latency_info["keyword_extraction"] = t1 - t0

        
        lookup_t0 = time.time()
        best_keyword = None
        best_entry = None
        
        # Search through all keywords for a cache hit
        for keyword in keywords:
            if keyword:
                entry = self.get_exact(keyword)
                if entry is not None:
                    best_keyword = keyword
                    best_entry = entry
                    break  # Use first hit found
        
        # If no hit found, use the first keyword as the primary one
        if best_keyword is None and keywords:
            best_keyword = keywords[0]
            
        lookup_t1 = time.time()
        latency_info["cache_lookup"] = lookup_t1 - lookup_t0

        return best_keyword, copy.deepcopy(best_entry) if best_entry is not None else None, total_tokens, latency_info, keywords


    def induce_and_insert(self, large_model, keywords: list[str], conversation_trace: dict[str, Any]) -> TokenUsage:
        """
        Use the large model to distill a structured, reusable plan template from a successful conversation trace.
        - keywords: list of keywords under which to store the template.
        - conversation_trace: a dict containing the dialogue/trace that led to the final solution.
        Returns the token usage (TokenUsage) consumed during cache induction (to be accounted for separately).
        """
        if not keywords:
            keywords = ["unknown"]
        
        # Normalize all keywords
        normalized_keywords = [self.normalize_keyword(kw) for kw in keywords if kw]
        if not normalized_keywords:
            normalized_keywords = ["unknown"]

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
        resp = large_model.generate([{"role": "user", "content": induction_prompt}], response_format={"type": "json_object"}, stop_sequences=["<end_plan>"])
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
                "keyword": normalized_keywords[0],
                "task": plan_string[:300],
                "plan_template": plan_string,
            }

        # Validate basic structure and fix if needed
        parsed = self._validate_and_fix_template(parsed)

        # Insert into cache under all keywords 
        for keyword in normalized_keywords:
            # Create a copy of the template for each keyword
            template_copy = copy.deepcopy(parsed)
            template_copy["keyword"] = keyword
            self.insert(keyword, template_copy)

        return token_usage

    def _extract_plan_template_as_text(self, conversation_trace: dict[str, Any]) -> str:
        """
        Extract a plain plan string from the conversation trace.
        Handles ChatMessage objects by converting them to dictionaries.
        """
        def serialize_obj(obj):
            """Recursively serialize objects, handling ChatMessage and other non-serializable types."""
            if hasattr(obj, 'dict'):  # ChatMessage and other objects with dict() method
                return obj.dict()
            elif isinstance(obj, dict):
                return {k: serialize_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_obj(item) for item in obj]
            else:
                return obj
        
        # Serialize the conversation trace to handle ChatMessage objects
        serialized_trace = serialize_obj(conversation_trace)
        return json.dumps(serialized_trace, ensure_ascii=False, indent=2)

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

    def adapt_cached_plan(self, small_model, cached_template: dict[str, Any], task: str) -> tuple[str, TokenUsage]:
        """
        Adapt a cached plan template to a new task.
        """
        adapt_prompt = [
            {
                "role": "system", 
                "content": "You are a lightweight planner. Given a cached plan template and a new task, "
                          "adapt the template to the new task. Keep structure (facts survey + plan) and "
                          "replace problem-specific details. The adapted plan should be in the format of "
                          "a GAIA plan."
            },
            {
                "role": "user",
                "content": f"Task:\n{task}\n\nCached plan template:\n{cached_template.get('plan_template', '')}\n\nPlease output the adapted plan."
            }
        ]

        adapt_msg = small_model.generate(adapt_prompt, stop_sequences=["<end_plan>"])
        adapt_token_usage = getattr(adapt_msg, "token_usage", None) or TokenUsage(0, 0)
        adapted_plan = getattr(adapt_msg, "content", "") or ""
        
        return adapted_plan, adapt_token_usage
