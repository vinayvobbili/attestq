"""Does this question want documentary evidence at all?

`quality.assess` measures whether an answer is carried by the evidence it cites.
What it cannot measure is whether the question wanted evidence in the first
place. "What is the legal entity name?" has a correct answer that appears in no
policy document and never will, so the support ratio for a correct answer is
legitimately zero — and a flag on it is noise. Noise is not free: a badge that
fires on questions nobody can fix is a badge reviewers learn to scroll past,
which costs you the real catches the check exists for.

Measurement cannot settle this. It is a judgment about what kind of thing is
being asked, which is what a language model is for.

Three decisions worth stating, because each one is load-bearing:

  1. **It reads the question, not the answer.** The signature takes both because
     `quality.ClaimClassifier` does, but the answer is deliberately ignored.
     "Does this question require documentary evidence?" is a property of the
     question; feeding the model the answer invites it to reason from what was
     written to whether it needed backing, which is exactly the rationalization
     we are trying to catch. Ignoring it also makes the cache correct by
     construction — the same question classifies the same way for every
     respondent, and questionnaires ask the same questions for years.

  2. **A small, fast model is the right tool.** This is a two-way classification,
     not drafting. Pointing it at the same large model that writes your answers
     spends rate-limit budget to answer a question an 8B-class model gets right.
     Inject whatever `ChatFn` you like — it need not be the one on your `Engine`.

  3. **Unsure means the flag stays.** Every failure path — endpoint down, slow,
     unparseable reply — returns True (evidence required), leaving the verdict
     exactly as measurement found it. `assess` consults this only on answers
     already flagged and only ever downgrades, so the worst a wrong call here
     can do is show a badge you would have shown anyway.

Usage::

    from attestq import Engine, make_claim_classifier

    engine = Engine(chat=big_model, embed=embedder,
                    verify=True,
                    claim_classifier=make_claim_classifier(small_model))
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

from .protocols import ChatFn

logger = logging.getLogger(__name__)

# Question text is short and repeats across respondents, so an unbounded dict
# would still be small — but questionnaires do get bulk-run, so cap it.
DEFAULT_CACHE_LIMIT = 2000

PROMPT = """You are classifying a security questionnaire item for an assurance team.

Decide which kind of item it is:

EVIDENCE - it asks what the organization does, how a control works, or whether a
safeguard is in place. A good answer makes a claim that policy, audit, or
architecture documentation should be able to back up.

RECORD - it asks for a fact of record about the organization or the submission
itself: legal or trading name, address, jurisdiction, contact details, registration
or DUNS numbers, headcount, revenue, dates, certificate expiry, questionnaire
scope or routing. A good answer is looked up, not evidenced.

Item:
{question}

Reply with exactly one word: EVIDENCE or RECORD."""

# Reasoning models emit a <think> block. We ask for one word, but a served model
# can be swapped or a chat template can ignore the no-think flag, and a stray
# reasoning block would make every reply unparseable — which fails closed to
# "evidence required" and silently turns the classifier off.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_reply(reply: str) -> Optional[bool]:
    """EVIDENCE -> True, RECORD -> False, anything else -> None (do not cache).

    Deliberately strict about the two words rather than hunting for them
    anywhere in the reply: a model that wrote a paragraph did not follow the
    instruction, and a paragraph mentioning both words would resolve on token
    order rather than meaning.
    """
    head = _THINK_RE.sub("", reply or "").strip().strip(".*_`\"' ").split()
    if not head:
        return None
    word = head[0].upper()
    if word.startswith("EVIDENCE"):
        return True
    if word.startswith("RECORD"):
        return False
    return None


class LLMClaimClassifier:
    """A `quality.ClaimClassifier` backed by any chat model.

    Callable as ``(question, answer="") -> bool``, where True means "a good
    answer to this should be backed by documentation". Results are memoized per
    question, and every failure path returns True so the classifier can only
    ever leave a flag standing, never raise one.

    Args:
        chat: Any ``(prompt: str) -> str`` callable. A small, fast model is the
            right choice — see the module docstring.
        cache_limit: Max memoized questions before the cache is cleared wholesale.
        prompt: Override the classification prompt. Must instruct the model to
            reply with exactly ``EVIDENCE`` or ``RECORD``, or supply a matching
            ``parser``.
        parser: Override reply parsing; ``str -> True | False | None``, where
            None means "unparseable, do not cache".
    """

    def __init__(
        self,
        chat: ChatFn,
        *,
        cache_limit: int = DEFAULT_CACHE_LIMIT,
        prompt: str = PROMPT,
        parser=parse_reply,
    ):
        self._chat = chat
        self._cache_limit = cache_limit
        self._prompt = prompt
        self._parser = parser
        self._cache: dict = {}
        self._lock = threading.Lock()

    @staticmethod
    def _cache_key(question: str) -> str:
        return " ".join(question.lower().split())

    def __call__(self, question: str, answer: str = "") -> bool:
        """True when a good answer to `question` should be backed by documentation.

        `answer` is accepted to match the `ClaimClassifier` signature and is not
        used — see the module docstring.
        """
        question = (question or "").strip()
        if not question:
            return True

        key = self._cache_key(question)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        try:
            reply = self._chat(self._prompt.format(question=question[:2000]))
            verdict = self._parser(reply or "")
        except Exception as e:
            logger.warning("[claim_classifier] classification failed: %s", e)
            return True

        if verdict is None:
            # Unparseable is a model problem, not a question property — keep the
            # flag and let the next call try again rather than caching a guess.
            logger.info("[claim_classifier] unparseable reply for: %s", question[:120])
            return True

        with self._lock:
            if len(self._cache) >= self._cache_limit:
                self._cache.clear()
            self._cache[key] = verdict
        return verdict

    def reset_cache(self) -> None:
        """Drop memoized classifications (tests, and after a prompt change)."""
        with self._lock:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)


def make_claim_classifier(chat: ChatFn, **kwargs) -> LLMClaimClassifier:
    """Build an `LLMClaimClassifier` — the callable `assess` and `Engine` expect."""
    return LLMClaimClassifier(chat, **kwargs)
