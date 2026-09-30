"""Hook title options and post captions for Viral Hook Mode.

One model call per finished clip, made with the clip's *final* transcript —
after boundary refinement and opening trims — so the model only sees words the
viewer will actually hear. Its reply is then checked in code, because "never
claim anything the speaker doesn't say" is too important to leave to a prompt:

- every number in a title must be a number the speaker says;
- every content word in a title must appear in the clip (or in the
  user-supplied speaker facts), allowing only generic hook framing words like
  "reveals" or "shocking";
- a title is at most 8 words, in Title Case or ALL CAPS, ending in one emoji.

An option that fails is dropped, not repaired. If fewer than three survive,
the gap is filled with titles cut verbatim from the clip's own sentences, which
are grounded by construction. The best survivor becomes the clip's hook title
and the next two are kept as alternatives.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..providers.base import DetectionConfig, LLMProvider, extract_json_object, load_prompt
from .hooktitle import EMOJI_RE, split_runs
from .transcript import Transcript

log = logging.getLogger(__name__)

PROMPT = "hook_copy_v1"
MAX_TITLE_WORDS = 8
OPTIONS = 3

#: Emoji appended when a title arrives without one, by topic.
TOPIC_EMOJI: dict[str, str] = {
    "money": "💰",
    "dating": "💔",
    "men vs women": "😳",
    "high-value": "👀",
    "status": "👑",
    "stats & studies": "📊",
    "controversial take": "😱",
}
DEFAULT_EMOJI = "🤔"

#: Hashtags used to top up a caption the model gave too few for.
TOPIC_HASHTAGS: dict[str, list[str]] = {
    "money": ["#money", "#finance", "#success"],
    "dating": ["#dating", "#relationships", "#datingadvice"],
    "men vs women": ["#men", "#women", "#relationships"],
    "high-value": ["#highvalueman", "#dating", "#selfimprovement"],
    "status": ["#status", "#success", "#mindset"],
    "stats & studies": ["#facts", "#study", "#psychology"],
    "controversial take": ["#debate", "#hottake", "#opinion"],
}
GENERIC_HASHTAGS = ["#podcast", "#podcastclips", "#viral"]


# --------------------------------------------------------------------------
# Words and numbers
# --------------------------------------------------------------------------

#: Words a title may use without the speaker saying them: grammar, and the
#: generic framing every hook pattern relies on. Nothing here asserts a fact.
FRAMING_WORDS = {
    # function words and pronouns
    "the", "and", "but", "for", "nor", "not", "you", "your", "you're", "yours", "this",
    "that", "these", "those", "them", "they", "their", "his", "her", "him", "she",
    "its", "it's", "with", "from", "into", "about", "than", "then", "what", "what's",
    "why", "how", "who", "when", "where", "which", "does", "did", "doing", "are", "was",
    "were", "is", "has", "have", "had", "can", "could", "would", "should", "will",
    "must", "here", "here's", "there", "every", "all", "any", "some", "just", "only",
    "very", "really", "actually", "even", "still", "ever", "never", "most", "more",
    "less", "out", "off", "over", "our", "we", "i'm", "i", "my", "me", "if", "or",
    "vs", "versus", "one", "thing", "things", "way", "ways", "much", "many", "lot",
    # hook framing verbs and intensifiers
    "reveals", "reveal", "explains", "explain", "says", "say", "said", "tells",
    "admits", "breaks", "down", "learn", "know", "need", "want", "watch", "wait",
    "shocking", "surprising", "brutal", "honest", "harsh", "truth", "real", "secret",
    "nobody", "everyone", "anyone", "people", "happens", "happened", "exactly",
}  # fmt: skip

#: Words that count as the same claim. A title may say "study" if the speaker
#: says "research"; it may not say "study" if the speaker says nothing of the kind.
SYNONYMS: list[set[str]] = [
    {"study", "research", "researcher", "survey", "scientist"},
    {"poll", "survey", "vote", "voter"},
    {"men", "man", "guy", "male", "husband", "boyfriend"},
    {"women", "woman", "girl", "female", "wife", "girlfriend"},
    {"money", "income", "salary", "earn", "earnings", "dollar", "rich", "wealth", "wealthy"},
    {"date", "dating", "relationship"},
]

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "k": 1_000,
    "grand": 1_000,
    "million": 1_000_000,
    "m": 1_000_000,
    "mil": 1_000_000,
    "billion": 1_000_000_000,
    "b": 1_000_000_000,
}
_SPECIAL = {"half": 50, "dozen": 12}
_NUMERIC = re.compile(r"^\$?(\d+(?:[.,]\d+)*)(k|m|b)?(%|\+)?$", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    text = EMOJI_RE.sub(" ", text)
    # Keep $, %, + and decimal points attached to numbers; split on the rest.
    return [t for t in re.split(r"[^\w$%+.,'’-]+", text.lower()) if t]


def _clean(token: str) -> str:
    return token.replace("’", "'").strip(".,'-")


def stem(word: str) -> str:
    """A deliberately crude stemmer: it only has to agree with itself, so that
    "date", "dates", "dated" and "dating" all land on the same key."""
    word = _clean(word).lower()
    if word.endswith("'s"):
        word = word[:-2]
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) >= 6:
        word = word[:-3]
    elif word.endswith("ed") and len(word) >= 5:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) >= 4:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]  # running -> runn -> run
    if len(word) > 3:
        word = word.rstrip("e")
    return word


def extract_numbers(text: str) -> set[float]:
    """Every number stated in ``text``, digits or words: "50k", "fifty thousand",
    "69%", "sixty-nine percent" and "$50,000" all come out as plain values."""
    values: set[float] = set()
    tokens = [_clean(t) for t in _tokens(text)]
    tokens = [p for t in tokens for p in (t.split("-") if not _NUMERIC.match(t) else [t]) if p]

    current: float | None = None
    total = 0.0

    def flush() -> None:
        nonlocal current, total
        if current is not None:
            values.add(total + current)
        current, total = None, 0.0

    for token in tokens:
        match = _NUMERIC.match(token)
        if match:
            flush()
            number = float(match.group(1).replace(",", ""))
            if match.group(2):
                number *= _SCALES[match.group(2).lower()]
            values.add(number)
            current, total = number, 0.0
            continue
        if token in _UNITS:
            current = (current or 0) + _UNITS[token]
        elif token in _SCALES and current is None and len(token) > 1:
            current = float(_SCALES[token])  # "a hundred", "a million"
        elif token in _SCALES and current is not None:
            scale = _SCALES[token]
            if scale == 100:
                current *= 100
            else:
                total += current * scale
                current = 0
                values.add(total)
        elif token in _SPECIAL:
            flush()
            values.add(_SPECIAL[token])
        elif token in ("and", "a") and current is not None:
            continue
        else:
            flush()
    flush()
    return values


def _is_number_token(token: str) -> bool:
    token = _clean(token)
    return bool(_NUMERIC.match(token)) or token in _UNITS or token in _SCALES or token in _SPECIAL


def _content_words(title: str) -> list[str]:
    words: list[str] = []
    for token in _tokens(title):
        for part in _clean(token).split("-"):
            part = _clean(part)
            if len(part) < 3 or _is_number_token(part) or part in ("percent", "%"):
                continue
            if part in FRAMING_WORDS:
                continue
            words.append(part)
    return words


def _vocabulary(text: str) -> set[str]:
    stems: set[str] = set()
    for token in _tokens(text):
        for part in [_clean(token), *_clean(token).split("-")]:
            if part:
                stems.add(stem(part))
    return stems


def _supported(word: str, vocabulary: set[str]) -> bool:
    word_stem = stem(word)
    if word_stem in vocabulary:
        return True
    # Compounds: "Multi-Millionaire" against a spoken "multimillionaire".
    if len(word_stem) >= 5 and any(word_stem in v for v in vocabulary):
        return True
    return any(
        word_stem in {stem(s) for s in group} and vocabulary & {stem(s) for s in group}
        for group in SYNONYMS
    )


def content_words(text: str) -> list[str]:
    """Words in ``text`` that carry meaning: no framing words, numbers, or short words."""
    return _content_words(text)


def vocabulary(text: str) -> set[str]:
    """Every stemmed word form in ``text``."""
    return _vocabulary(text)


@dataclass
class GroundingResult:
    ok: bool
    unsupported_words: list[str] = field(default_factory=list)
    unsupported_numbers: list[float] = field(default_factory=list)


def check_grounding(title: str, clip_text: str, facts: str = "") -> GroundingResult:
    """Whether every number and content word in ``title`` is backed by the clip."""
    source = f"{clip_text} {facts}"
    numbers = extract_numbers(source)
    title_numbers = extract_numbers(title)
    if not re.search(r"(?<![\d.])1(?![\d.])", title):
        # "One Thing Women Want" is phrasing, not a statistic.
        title_numbers.discard(1)
    missing_numbers = sorted(n for n in title_numbers if n not in numbers)
    vocabulary = _vocabulary(source)
    missing_words = [w for w in _content_words(title) if not _supported(w, vocabulary)]
    return GroundingResult(
        ok=not missing_numbers and not missing_words,
        unsupported_words=missing_words,
        unsupported_numbers=missing_numbers,
    )


# --------------------------------------------------------------------------
# Title formatting
# --------------------------------------------------------------------------

_SMALL_WORDS = {"a", "an", "the", "and", "but", "or", "nor", "for", "on", "at", "to", "by",
                "of", "in", "vs", "vs.", "as"}  # fmt: skip


def title_case(text: str) -> str:
    words = text.split()
    out: list[str] = []
    for index, word in enumerate(words):
        core = re.sub(r"[^\w]", "", word)
        if any(c.isdigit() for c in word) or (core.isupper() and len(core) > 1):
            out.append(word)  # 69%, $100K, USA
        elif 0 < index < len(words) - 1 and word.lower() in _SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append("-".join(p[:1].upper() + p[1:].lower() for p in word.split("-")))
    return " ".join(out)


def word_count(title: str) -> int:
    return len([w for w in EMOJI_RE.sub(" ", title).split() if re.search(r"\w", w)])


def normalise_title(text: str, *, topic: str = "") -> str:
    """Enforce the format: one trailing emoji, Title Case unless ALL CAPS."""
    text = " ".join(str(text).split())
    emoji = [run for run, is_emoji in split_runs(text) if is_emoji]
    bare = " ".join(EMOJI_RE.sub(" ", text).split()).strip()
    letters = [c for c in bare if c.isalpha()]
    if not (letters and all(c.isupper() for c in letters)):
        bare = title_case(bare)
    chosen = emoji[-1] if emoji else TOPIC_EMOJI.get(topic.strip().lower(), DEFAULT_EMOJI)
    return f"{bare} {chosen}".strip()


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


@dataclass
class TitleOption:
    text: str
    score: int = 0
    pattern: int | None = None
    #: "model" or "fallback" (cut verbatim from the clip).
    source: str = "model"


@dataclass
class HookCopy:
    title: str
    alternatives: list[str]
    caption: str
    rejected: list[dict] = field(default_factory=list)


def validate_option(
    raw: str, *, clip_text: str, topic: str = "", facts: str = ""
) -> tuple[str | None, str]:
    """Return ``(normalised title, "")`` or ``(None, reason it was rejected)``."""
    title = normalise_title(raw, topic=topic)
    if not EMOJI_RE.sub("", title).strip():
        return None, "empty"
    if word_count(title) > MAX_TITLE_WORDS:
        return None, f"more than {MAX_TITLE_WORDS} words"
    grounding = check_grounding(title, clip_text, facts)
    if not grounding.ok:
        parts = []
        if grounding.unsupported_numbers:
            parts.append(f"numbers not in the clip: {grounding.unsupported_numbers}")
        if grounding.unsupported_words:
            parts.append(f"words not in the clip: {grounding.unsupported_words}")
        return None, "; ".join(parts)
    return title, ""


def fallback_titles(transcript: Transcript, start: int, end: int, *, topic: str = "") -> list[str]:
    """Titles cut verbatim from the clip's opening sentences — grounded by construction."""
    sentences: list[str] = []
    current: list[str] = []
    for word in transcript.words[start : end + 1]:
        current.append(word.text)
        if word.ends_sentence:
            sentences.append(" ".join(current))
            current = []
        if len(sentences) >= 6:
            break
    if current and len(sentences) < 6:
        sentences.append(" ".join(current))

    titles: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) < 3:
            continue
        if len(words) > MAX_TITLE_WORDS:
            words = words[:MAX_TITLE_WORDS]
            words[-1] = words[-1].rstrip(",.;:!?") + "…"
        text = " ".join(words).rstrip(",;:")
        titles.append(normalise_title(text, topic=topic))
    return titles


def choose_titles(
    options: list[TitleOption],
    *,
    clip_text: str,
    fallbacks: list[str],
    topic: str = "",
    facts: str = "",
) -> tuple[list[TitleOption], list[dict]]:
    """Validate, rank, and top up to :data:`OPTIONS` titles. Best first."""
    valid: list[TitleOption] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for option in options:
        title, reason = validate_option(option.text, clip_text=clip_text, topic=topic, facts=facts)
        if title is None:
            rejected.append({"text": option.text, "reason": reason})
            continue
        if title.lower() in seen:
            continue
        seen.add(title.lower())
        valid.append(TitleOption(title, option.score, option.pattern, "model"))

    valid.sort(key=lambda o: o.score, reverse=True)
    for text in fallbacks:
        if len(valid) >= OPTIONS:
            break
        if text.lower() not in seen:
            seen.add(text.lower())
            valid.append(TitleOption(text, 0, None, "fallback"))
    return valid[:OPTIONS], rejected


# --------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------


def _hashtag(tag: str) -> str | None:
    cleaned = re.sub(r"[^\w]", "", str(tag).lower())
    return f"#{cleaned}" if cleaned else None


def build_caption(
    line: str,
    hashtags: list[str],
    *,
    title: str,
    clip_text: str,
    topic: str = "",
    handle: str = "@michaelsartain",
    fixed_hashtags: list[str] | None = None,
    facts: str = "",
) -> str:
    """One question line, then the handle, 3-5 topical hashtags, and the fixed ones.

    The line must end in a question and state no number the speaker doesn't
    say; otherwise it is rebuilt from the (already grounded) hook title.
    """
    fixed = [t for t in (_hashtag(tag) for tag in (fixed_hashtags or [])) if t]
    line = " ".join(str(line).split())
    line = re.sub(r"(?:\s*[#@]\w+)+\s*$", "", line).strip()  # stray tags on the end
    numbers_ok = extract_numbers(line) <= extract_numbers(f"{clip_text} {facts}")
    if not line or not line.endswith("?") or not numbers_ok:
        bare = EMOJI_RE.sub("", title).strip().rstrip("?!.… ")
        line = f"{bare}. Do you agree?"

    # The model's own tags first (up to 5), topped up to 3 from the topic's.
    tags: list[str] = []
    for tag in hashtags:
        cleaned = _hashtag(tag)
        if cleaned and cleaned not in tags and cleaned not in fixed and len(tags) < 5:
            tags.append(cleaned)
    for tag in [*TOPIC_HASHTAGS.get(topic.strip().lower(), []), *GENERIC_HASHTAGS]:
        if len(tags) >= 3:
            break
        if tag not in tags and tag not in fixed:
            tags.append(tag)

    footer = " ".join(part for part in [handle.strip(), *tags, *fixed] if part)
    return f"{line}\n\n{footer}"


def caption_filename(rank: int) -> str:
    return f"clip_{rank:02d}_caption.txt"


def write_caption_file(directory: Path, rank: int, caption: str) -> Path | None:
    """Write ``clip_XX_caption.txt``; removes a stale one when the caption is empty."""
    path = directory / caption_filename(rank)
    if not caption.strip():
        path.unlink(missing_ok=True)
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(caption.strip() + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def build_user_prompt(clip_text: str, *, topic: str = "", facts: str = "") -> str:
    facts_block = (
        f"SPEAKER FACTS (true, supplied by the editor — you may use these):\n{facts.strip()}\n\n"
        if facts.strip()
        else "SPEAKER FACTS: none supplied. Do not describe the speaker.\n\n"
    )
    return (
        f"Clip topic: {topic or 'unspecified'}\n\n"
        f"{facts_block}"
        f"CLIP TRANSCRIPT (the complete clip, exactly as spoken):\n---\n{clip_text}\n---\n\n"
        "Respond with ONLY the JSON object."
    )


async def generate(
    provider: LLMProvider,
    clip_text: str,
    *,
    fallbacks: list[str],
    topic: str = "",
    facts: str = "",
    handle: str = "@michaelsartain",
    fixed_hashtags: list[str] | None = None,
    config: DetectionConfig | None = None,
) -> HookCopy:
    """Ask the model for titles and a caption, then validate everything.

    Never raises for a model failure: a clip always ends up with grounded
    titles and a caption, falling back to its own sentences if need be.
    """
    config = config or DetectionConfig(prompt_version=PROMPT)
    system = load_prompt(PROMPT)
    user = build_user_prompt(clip_text, topic=topic, facts=facts)

    payload: dict = {}
    for attempt in range(2):
        try:
            raw = await provider.complete(system, user, config)
            payload = extract_json_object(raw)
            break
        except Exception as exc:
            log.warning("Hook copy attempt %d failed (%s).", attempt + 1, exc)

    options = []
    for item in payload.get("titles") or []:
        if isinstance(item, str):
            options.append(TitleOption(item))
        elif isinstance(item, dict) and item.get("text"):
            try:
                score = int(float(item.get("score") or 0))
            except (TypeError, ValueError):
                score = 0
            pattern = item.get("pattern") if isinstance(item.get("pattern"), int) else None
            options.append(TitleOption(str(item["text"]), score, pattern))

    chosen, rejected = choose_titles(
        options, clip_text=clip_text, fallbacks=fallbacks, topic=topic, facts=facts
    )
    for item in rejected:
        log.info("Rejected hook title %r: %s", item["text"], item["reason"])
    if not chosen:
        chosen = [TitleOption(normalise_title("Watch This", topic=topic), 0, None, "fallback")]

    hashtags = payload.get("hashtags") or []
    if not isinstance(hashtags, list):
        hashtags = str(hashtags).split()
    caption = build_caption(
        str(payload.get("caption") or ""),
        [str(h) for h in hashtags],
        title=chosen[0].text,
        clip_text=clip_text,
        topic=topic,
        handle=handle,
        fixed_hashtags=fixed_hashtags,
        facts=facts,
    )
    return HookCopy(
        title=chosen[0].text,
        alternatives=[o.text for o in chosen[1:]],
        caption=caption,
        rejected=rejected,
    )


async def generate_for_clips(
    clips: list,
    transcript: Transcript,
    provider: LLMProvider,
    *,
    facts: str = "",
    handle: str = "@michaelsartain",
    fixed_hashtags: list[str] | None = None,
    trace_dir: Path | None = None,
    on_progress=None,
    concurrency: int = 3,
) -> None:
    """Fill ``hook_title``, ``hook_title_alts`` and ``post_caption`` on each clip.

    A title the user already set is kept. Rejected model titles are written to
    ``trace_dir`` so it's visible why a clip got the titles it did.
    """
    import asyncio
    import json

    semaphore = asyncio.Semaphore(1 if provider.name == "ollama" else concurrency)
    done = 0

    async def one(clip) -> None:
        nonlocal done
        clip_text = transcript.text_between(clip.start_word, clip.end_word)
        fallbacks = fallback_titles(transcript, clip.start_word, clip.end_word, topic=clip.topic)
        async with semaphore:
            copy = await generate(
                provider,
                clip_text,
                fallbacks=fallbacks,
                topic=clip.topic,
                facts=facts,
                handle=handle,
                fixed_hashtags=fixed_hashtags,
            )
        if not clip.hook_title:
            clip.hook_title = copy.title
            clip.hook_title_alts = copy.alternatives
        clip.post_caption = copy.caption
        if trace_dir is not None:
            try:
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / f"copy_{clip.rank:02d}.json").write_text(
                    json.dumps(
                        {
                            "clip_text": clip_text,
                            "title": copy.title,
                            "alternatives": copy.alternatives,
                            "caption": copy.caption,
                            "rejected": copy.rejected,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.warning("Could not write hook copy trace: %s", exc)
        done += 1
        if on_progress:
            on_progress(done / len(clips))

    await asyncio.gather(*(one(clip) for clip in clips))
