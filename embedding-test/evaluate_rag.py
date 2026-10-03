"""Reproducible HTTP benchmark; writes raw results and aggregate JSON metrics."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import requests

MODES = ('dense', 'hybrid', 'hybrid_rerank')


def retrieval_metrics(rows):
    # Failures stay in the denominator so outages cannot improve accuracy.
    eligible = [r for r in rows if r['expected_sources']]
    hits, reciprocal, recall = [], [], []
    for row in eligible:
        expected = set(row['expected_sources'])
        found = row.get('sources', [])
        ranks = [i + 1 for i, source in enumerate(found) if source in expected]
        hits.append(bool(ranks))
        reciprocal.append(1 / min(ranks) if ranks else 0)
        recall.append(len(expected.intersection(found)) / len(expected))
    latencies = sorted(r['latency_seconds'] for r in rows if r['ok'])
    return {
        'requests': len(rows), 'errors': sum(not r['ok'] for r in rows),
        'answerable_questions': len(eligible),
        'source_hit_at_k': sum(hits) / len(hits) if hits else None,
        'source_recall_at_k': sum(recall) / len(recall) if recall else None,
        'source_mrr_at_k': sum(reciprocal) / len(reciprocal) if reciprocal else None,
        'mean_latency_seconds': sum(latencies) / len(latencies) if latencies else None,
        'p95_latency_seconds': latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
    }


def refusal_metrics(rows):
    negatives = [r for r in rows if r['should_refuse']]
    positives = [r for r in rows if not r['should_refuse']]
    return {
        'requests': len(rows), 'errors': sum(not r['ok'] for r in rows),
        'refusal_recall': sum(r['ok'] and r.get('refused') is True for r in negatives) / len(negatives) if negatives else None,
        'false_refusal_rate': sum(r['ok'] and r.get('refused') is True for r in positives) / len(positives) if positives else None,
        'answerable_response_rate': sum(r['ok'] and r.get('refused') is False and r.get('answer_mode') == 'generated' for r in positives) / len(positives) if positives else None,
        'extractive_responses': sum(r.get('answer_mode') == 'extractive' for r in rows),
    }


def load_dataset(path):
    items = json.loads(Path(path).read_text())
    if not isinstance(items, list) or not items:
        raise ValueError('Dataset must be a non-empty list')
    seen = set()
    for item in items:
        if not isinstance(item.get('id'), str) or item['id'] in seen:
            raise ValueError('Dataset IDs must be unique strings')
        seen.add(item['id'])
        if not isinstance(item.get('question'), str) or not item['question'].strip():
            raise ValueError('Each item needs a question')
        if not isinstance(item.get('expected_sources'), list) or not all(isinstance(s, str) for s in item['expected_sources']):
            raise ValueError('expected_sources must be a list of filenames')
        if type(item.get('should_refuse')) is not bool:
            raise ValueError('should_refuse must be a boolean')
        if item['should_refuse'] == bool(item['expected_sources']):
            raise ValueError('Answerable items need sources; refusal items must have none')
    return items


def run_evaluation(args):
    items = load_dataset(args.dataset)
    session = requests.Session()
    session.headers.update({'X-API-Key': os.getenv('API_ACCESS_KEY', '')})
    sources_response = session.get(f'{args.base_url.rstrip("/")}/api/v1/sources', timeout=30)
    sources_response.raise_for_status()
    available = set(sources_response.json())
    missing = {source for item in items for source in item['expected_sources']} - available
    if missing:
        raise ValueError(f'Benchmark source documents are missing: {sorted(missing)}')
    rows = []
    for mode in (*MODES, *(() if args.skip_rag else ('rag',))):
        for item in items:
            payload = {'limit': args.k, 'score_threshold': args.threshold}
            if mode == 'rag':
                endpoint = 'ask'
                payload['question'] = item['question']
            else:
                endpoint = 'search'
                payload.update(query=item['question'], retrieval_mode=mode)
            row = {**item, 'mode': mode, 'ok': False}
            start = time.perf_counter()
            try:
                response = session.post(f'{args.base_url.rstrip("/")}/api/v1/{endpoint}', json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                row.update(ok=True, sources=[r['source'] for r in data.get('results', data.get('citations', []))],
                           refused=data.get('refused'), answer_mode=data.get('answer_mode'),
                           answer=data.get('answer'), faithfulness=data.get('faithfulness'),
                           answer_relevance=data.get('answer_relevance'), evidence_threshold=data.get('evidence_threshold'))
            except (requests.RequestException, ValueError, KeyError) as exc:
                # Avoid persisting request headers or provider secrets.
                row['error'] = type(exc).__name__
            row['latency_seconds'] = time.perf_counter() - start
            rows.append(row)
            print(f'{mode}: {item["id"]}: {"ok" if row["ok"] else "error"}', flush=True)
            time.sleep(args.delay)
    report = {
        'dataset': str(args.dataset),
        'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'k': args.k, 'score_threshold': args.threshold,
        'metrics': {mode: retrieval_metrics([r for r in rows if r['mode'] == mode]) for mode in MODES},
        'rag': refusal_metrics([r for r in rows if r['mode'] == 'rag']), 'results': rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report['metrics'], indent=2))
    print(f'Report: {output}')
    return 1 if any(not r['ok'] for r in rows) else 0


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--dataset', default=str(Path(__file__).resolve().parents[1] / 'evaluation/questions.json'))
    parser.add_argument('--output', default='evaluation/results/latest.json')
    parser.add_argument('--k', type=int, choices=range(1, 11), default=3)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--delay', type=float, default=3.1, help='Seconds between requests; respects default rate limit')
    parser.add_argument('--skip-rag', action='store_true', help='Retrieval only; avoids paid LLM calls')
    arguments = parser.parse_args()
    if not 0 <= arguments.threshold <= 1 or arguments.delay < 0:
        parser.error('threshold must be in [0,1]; delay must be non-negative')
    raise SystemExit(run_evaluation(arguments))
