"""Evidence gating uses cosine similarity, never RRF or cross-encoder logits."""
import math

REFUSAL = 'Aranan soruyu yanıtlamak için indekslenmiş belgelerde yeterli kanıt bulunamadı.'


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError('Embedding dimensions differ')
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0


def select_evidence(store, question, points, threshold):
    candidates = [p for p in points if (p.payload or {}).get('text', '').strip()
                  and (p.payload or {}).get('source') and (p.payload or {}).get('page_number', 0) > 0]
    if not candidates:
        return []
    vectors = store.encode([question] + [p.payload['text'] for p in candidates])
    return [p for p, vector in zip(candidates, vectors[1:]) if cosine(vectors[0], vector) >= threshold]


def citations_supported(answer, points):
    """Require at least one page citation and reject references outside the context."""
    import re
    citations = re.findall(r'\(([^()]+\.pdf),\s*(?:Page|Sayfa):\s*(\d+)\)', answer, flags=re.I)
    allowed = {(p.payload['source'], int(p.payload['page_number'])) for p in points}
    return bool(citations) and all((source.strip(), int(page)) in allowed for source, page in citations)
