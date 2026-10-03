"""Offline regression tests: no model downloads or production database writes."""
import ast
import logging
import tempfile
import types
import unittest
import uuid
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
import requests

ROOT = Path(__file__).resolve().parents[1] / 'embedding-test'


def load_definition(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


class IngestionTests(unittest.TestCase):
    def test_chunk_identity_preserves_provenance_and_is_repeatable(self):
        store = Mock()
        store.encode.side_effect = lambda texts: [[0.1] for _ in texts]
        store.encode_sparse.side_effect = lambda texts: [{'indices': [1], 'values': [1.0]} for _ in texts]
        upsert = load_definition('ingest_to_qdrant.py', 'bulk_upsert_chunks', {
            'SpaceScienceVectorStore': Mock, 'logger': logging.getLogger(),
            'time': __import__('time'), 'uuid': uuid, 'json': json,
            'PointStruct': types.SimpleNamespace, 'SparseVector': types.SimpleNamespace,
        })
        chunks = [dict(text='same text', source=source, page_number=page, chunk_index=1, char_count=9)
                  for source, page in [('a.pdf', 1), ('b.pdf', 1), ('a.pdf', 2)]]
        upsert(store, 'test', chunks)
        first = store.client.upsert.call_args.kwargs['points']
        self.assertEqual(len({p.id for p in first}), 3)
        self.assertTrue(store.client.upsert.call_args.kwargs['wait'])
        upsert(store, 'test', chunks)
        self.assertEqual([p.id for p in first], [p.id for p in store.client.upsert.call_args.kwargs['points']])
        with self.assertRaises(ValueError):
            upsert(store, 'test', chunks, batch_size=0)

    def test_interrupted_download_is_removed_and_retry_succeeds(self):
        pipeline_class = load_definition('ingest_pdfs.py', 'DocumetPipeline', {
            'os': os, 'tempfile': tempfile, 'requests': requests, 'logger': logging.getLogger(),
        })
        with tempfile.TemporaryDirectory() as directory:
            pipeline = pipeline_class(directory)
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            def interrupted(*args, **kwargs):
                yield b'%PDF-partial'
                raise requests.exceptions.ConnectionError('interrupted')
            response.iter_content.side_effect = interrupted
            with patch.object(requests, 'get', return_value=response):
                with self.assertRaises(requests.exceptions.ConnectionError):
                    pipeline.download_pdf('https://example.test/paper', 'paper.pdf')
                self.assertEqual(list(Path(directory).iterdir()), [])
                response.iter_content.side_effect = lambda **kwargs: iter([b'%PDF-complete'])
                path = pipeline.download_pdf('https://example.test/paper', 'paper.pdf')
                self.assertEqual(Path(path).read_bytes(), b'%PDF-complete')
            with self.assertRaises(ValueError):
                pipeline.download_pdf('https://example.test/paper', '../paper.pdf')

    def test_source_filter_applies_to_both_prefetches(self):
        from qdrant_client import models
        store_class = load_definition('save_to_qdrant.py', 'SpaceScienceVectorStore', {
            'logger': logging.getLogger(), 'time': __import__('time'),
            **{name: getattr(models, name) for name in ['SparseVector', 'Prefetch', 'FusionQuery', 'Fusion']},
        })
        store = object.__new__(store_class)
        store.encode = Mock(return_value=[[0.1]])
        store.encode_sparse = Mock(return_value=[{'indices': [1], 'values': [1.0]}])
        store.client = Mock()
        store.search_documents('test', 'galaxy', source_filter='a.pdf')
        query = store.client.query_points.call_args.kwargs
        for prefetch in query['prefetch']:
            self.assertEqual(prefetch.filter, query['query_filter'])
            self.assertEqual(prefetch.filter.must[0].match.value, 'a.pdf')

    def test_real_qdrant_dense_and_hybrid_thresholds(self):
        import sys
        sys.path.insert(0, str(ROOT))
        from qdrant_client import QdrantClient, models
        store_class = load_definition('save_to_qdrant.py', 'SpaceScienceVectorStore', {
            'logger': logging.getLogger(), 'time': __import__('time'),
            **{name: getattr(models, name) for name in ['SparseVector', 'Prefetch', 'FusionQuery', 'Fusion']},
        })
        store = object.__new__(store_class)
        store.client = QdrantClient(':memory:')
        store.client.create_collection('test', vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
                                       sparse_vectors_config={'sparse-text': models.SparseVectorParams()})
        points = [models.PointStruct(id=i, vector={'': vector, 'sparse-text': models.SparseVector(indices=[1], values=[1.0])},
                                    payload={'text': text, 'source': source})
                  for i, vector, text, source in [(1, [1., 0.], 'relevant', 'a.pdf'), (2, [0., 1.], 'unrelated', 'a.pdf'), (3, [1., 0.], 'relevant', 'b.pdf')]]
        store.client.upsert('test', points=points)
        store.encode = lambda texts: [[0., 1.] if text == 'unrelated' else [1., 0.] for text in texts]
        store.encode_sparse = Mock(return_value=[{'indices': [1], 'values': [1.0]}])
        for mode in ('dense', 'hybrid'):
            result = store.search_documents('test', 'question', limit=3, score_threshold=.35, source_filter='a.pdf', retrieval_mode=mode)
            self.assertEqual([p.id for p in result], [1])
        store.client.close()


if __name__ == '__main__':
    unittest.main()
