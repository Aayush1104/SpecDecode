"""Domain-specific quality metrics for evaluating output correctness."""

import ast
import re


def compute_quality_metrics(outputs: list[dict], dataset_name: str) -> dict:
    """Dispatch to domain-specific quality evaluator.

    Args:
        outputs: List of dicts with 'prompt', 'generated', 'domain', and optional 'reference'.
        dataset_name: One of 'humaneval', 'gsm8k', 'mt_bench', 'truthfulqa'.

    Returns:
        Dict with quality scores and per-sample details.
    """
    dispatch = {
        "humaneval": _code_quality,
        "gsm8k": _math_quality,
        "mt_bench": _conversation_quality,
        "truthfulqa": _factuality_quality,
    }

    # Fall back to domain-based dispatch
    if dataset_name not in dispatch:
        domains = {o.get("domain", "") for o in outputs}
        if "code" in domains:
            return _code_quality(outputs)
        elif "math" in domains:
            return _math_quality(outputs)
        elif "factuality" in domains:
            return _factuality_quality(outputs)
        else:
            return _conversation_quality(outputs)

    return dispatch[dataset_name](outputs)


def _code_quality(outputs: list[dict]) -> dict:
    """Evaluate code outputs: syntax validity and reference similarity."""
    results = []
    valid_count = 0

    for out in outputs:
        generated = out.get("generated", "")
        reference = out.get("reference", "")

        # Check syntax validity
        syntax_valid = _check_python_syntax(out.get("prompt", "") + generated)

        if syntax_valid:
            valid_count += 1

        # Simple similarity to reference (if available)
        similarity = _text_similarity(generated, reference) if reference else None

        results.append({
            "syntax_valid": syntax_valid,
            "reference_similarity": similarity,
        })

    n = len(outputs) or 1
    return {
        "metric": "syntax_validity",
        "score": valid_count / n,
        "valid_count": valid_count,
        "total": len(outputs),
        "per_sample": results,
    }


def _math_quality(outputs: list[dict]) -> dict:
    """Evaluate math outputs: answer extraction + exact match."""
    results = []
    correct_count = 0

    for out in outputs:
        generated = out.get("generated", "")
        reference = out.get("reference", "")

        extracted = _extract_number(generated)
        ref_number = _extract_number(reference) if reference else None

        match = False
        if extracted is not None and ref_number is not None:
            match = abs(extracted - ref_number) < 1e-6

        if match:
            correct_count += 1

        results.append({
            "extracted_answer": extracted,
            "reference_answer": ref_number,
            "correct": match,
        })

    n = len(outputs) or 1
    return {
        "metric": "exact_match",
        "score": correct_count / n,
        "correct_count": correct_count,
        "total": len(outputs),
        "per_sample": results,
    }


def _conversation_quality(outputs: list[dict]) -> dict:
    """Evaluate conversation outputs: length and diversity."""
    results = []
    lengths = []
    unique_tokens = set()

    for out in outputs:
        generated = out.get("generated", "")
        words = generated.split()
        length = len(words)
        lengths.append(length)
        unique_tokens.update(words)

        results.append({
            "length_words": length,
        })

    n = len(outputs) or 1
    total_words = sum(lengths)
    return {
        "metric": "response_quality",
        "score": min(1.0, len(unique_tokens) / max(total_words, 1)),
        "mean_length_words": total_words / n,
        "vocabulary_diversity": len(unique_tokens) / max(total_words, 1) if total_words > 0 else 0.0,
        "total": len(outputs),
        "per_sample": results,
    }


def _factuality_quality(outputs: list[dict]) -> dict:
    """Evaluate factuality: case-insensitive substring match against reference."""
    results = []
    match_count = 0

    for out in outputs:
        generated = out.get("generated", "").lower().strip()
        reference = out.get("reference", "").lower().strip()

        match = False
        if reference and reference in generated:
            match = True

        if match:
            match_count += 1

        results.append({
            "reference_found": match,
        })

    n = len(outputs) or 1
    return {
        "metric": "reference_match",
        "score": match_count / n,
        "match_count": match_count,
        "total": len(outputs),
        "per_sample": results,
    }


def _check_python_syntax(code: str) -> bool:
    """Check if a string is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_number(text: str) -> float | None:
    """Extract the last number from text (handles integers and floats)."""
    # Look for numbers, preferring the last one (usually the final answer)
    matches = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def _text_similarity(a: str, b: str) -> float:
    """Simple token overlap similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
