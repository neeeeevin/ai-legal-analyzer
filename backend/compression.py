from backend.ml_model import classify_batch


def compress_clauses(clauses, threshold=0.6, min_keep=10):

    if not clauses:
        return []

    LABEL_WEIGHTS = {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.2
    }

    results = classify_batch(clauses)

    processed = []

    for r in results:
        label = r.get("label", "low")
        confidence = float(r.get("confidence", 0))

        weight = LABEL_WEIGHTS.get(label, 0.2)
        risk_score = confidence * weight

        processed.append({
            "text": r.get("text", "").replace(" ,", ",").strip(),
            "label": label,
            "confidence": confidence,
            "risk_score": risk_score
        })

    high_impact = [r for r in processed if r["risk_score"] >= threshold]

    if len(high_impact) < min_keep:
        processed.sort(key=lambda x: x["risk_score"], reverse=True)
        high_impact = processed[:min_keep]
    else:
        high_impact.sort(key=lambda x: x["risk_score"], reverse=True)

    return high_impact