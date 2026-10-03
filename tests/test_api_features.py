import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1] / 'embedding-test'
sys.path.insert(0, str(ROOT))
from api_protection import APIProtectionMiddleware
from evidence import select_evidence, citations_supported
from evaluate_rag import retrieval_metrics, refusal_metrics, load_dataset, run_evaluation


class ProtectionTests(unittest.TestCase):
    def client(self, **env):
        app = FastAPI()
        @app.api_route('/api/v1/search', methods=['POST'])
        @app.api_route('/api/v1/ingest/daily', methods=['POST'])
        @app.get('/api/v1/health')
        def endpoint():
            return {'ok': True}
        app.add_middleware(APIProtectionMiddleware)
        with patch.dict(os.environ, {'API_ACCESS_KEY': '', 'INGEST_API_KEY': '',
                                     'API_RATE_LIMIT': '20', 'API_RATE_WINDOW_SECONDS': '60', **env}):
            client = TestClient(app)
            client.get('/api/v1/health')  # Build middleware with controlled environment.
        return client

    def test_authentication_and_admin_separation(self):
        client = self.client(API_ACCESS_KEY='reader', INGEST_API_KEY='admin')
        self.assertEqual(client.post('/api/v1/search').status_code, 401)
        self.assertEqual(client.post('/api/v1/search', headers={'X-API-Key': 'reader'}).status_code, 200)
        self.assertEqual(client.post('/api/v1/ingest/daily', headers={'X-API-Key': 'reader'}).status_code, 401)
        self.assertEqual(client.post('/api/v1/ingest/daily', headers={'X-API-Key': 'admin'}).status_code, 200)
        self.assertEqual(client.get('/api/v1/health').status_code, 200)
        self.assertEqual(self.client().post('/api/v1/ingest/daily').status_code, 503)

    def test_limit_retry_and_spoofed_forwarding_headers(self):
        client = self.client(API_RATE_LIMIT='1')
        self.assertEqual(client.post('/api/v1/search').status_code, 200)
        response = client.post('/api/v1/search', headers={'X-Forwarded-For': 'another-user'})
        self.assertEqual(response.status_code, 429)
        self.assertGreaterEqual(int(response.headers['Retry-After']), 1)
        with patch('api_protection.time.monotonic', return_value=10**12):
            self.assertEqual(client.post('/api/v1/search').status_code, 200)

    def test_body_size_limit(self):
        client = self.client()
        self.assertEqual(client.post('/api/v1/search', content=b'x' * 65537).status_code, 413)


class EvidenceTests(unittest.TestCase):
    def test_gate_uses_cosine_not_retrieval_score(self):
        store = Mock()
        points = [types.SimpleNamespace(score=100, payload={'source': 'a.pdf', 'page_number': 1, 'text': 'irrelevant'}),
                  types.SimpleNamespace(score=.01, payload={'source': 'b.pdf', 'page_number': 2, 'text': 'relevant'})]
        store.encode.return_value = [[1, 0], [0, 1], [1, 0]]
        self.assertEqual(select_evidence(store, 'question', points, .35), [points[1]])
        self.assertFalse(citations_supported('Unsupported (a.pdf, Page: 9)', points))
        self.assertFalse(citations_supported('No citation', points))
        self.assertTrue(citations_supported('Fact (b.pdf, Page: 2)', points))

    def test_missing_provenance_is_not_evidence(self):
        store = Mock()
        points = [types.SimpleNamespace(payload={'text': 'facts', 'source': 'a.pdf', 'page_number': 0})]
        self.assertEqual(select_evidence(store, 'question', points, .35), [])
        store.encode.assert_not_called()


class MetricsTests(unittest.TestCase):
    def test_known_ranks_and_errors(self):
        rows = [{'ok': True, 'expected_sources': ['a.pdf'], 'sources': ['b.pdf', 'a.pdf'], 'latency_seconds': 2},
                {'ok': False, 'expected_sources': ['a.pdf'], 'latency_seconds': 3}]
        result = retrieval_metrics(rows)
        self.assertEqual(result['source_hit_at_k'], .5)
        self.assertEqual(result['source_mrr_at_k'], .25)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['p95_latency_seconds'], 2)

    def test_benchmark_writes_raw_results_and_summary(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'questions.json'
            dataset.write_text(json.dumps([
                {'id': 'answerable', 'question': 'telescope', 'expected_sources': ['a.pdf'], 'should_refuse': False},
                {'id': 'negative', 'question': 'pizza', 'expected_sources': [], 'should_refuse': True},
            ]))
            session = Mock()
            session.get.return_value.json.return_value = ['a.pdf']
            def post(url, json, timeout):
                response = Mock()
                if url.endswith('/ask'):
                    refused = json['question'] == 'pizza'
                    response.json.return_value = {'refused': refused, 'answer_mode': 'refusal' if refused else 'generated',
                                                 'citations': [] if refused else [{'source': 'a.pdf'}]}
                else:
                    response.json.return_value = {'results': [{'source': 'a.pdf'}]}
                return response
            session.post.side_effect = post
            output = Path(directory) / 'report.json'
            args = types.SimpleNamespace(dataset=dataset, output=output, base_url='http://example.test',
                                         k=3, threshold=0, delay=0, skip_rag=False)
            with patch('evaluate_rag.requests.Session', return_value=session), patch('builtins.print'):
                self.assertEqual(run_evaluation(args), 0)
            report = json.loads(output.read_text())
            self.assertEqual(len(report['results']), 8)
            self.assertEqual(report['metrics']['dense']['source_hit_at_k'], 1)
            self.assertEqual(report['rag']['refusal_recall'], 1)
            self.assertEqual(report['rag']['answerable_response_rate'], 1)
            self.assertEqual(len(report['dataset_sha256']), 64)

    def test_refusal_and_generation_are_separate(self):
        rows = [{'ok': True, 'should_refuse': True, 'refused': True, 'answer_mode': 'refusal'},
                {'ok': True, 'should_refuse': False, 'refused': False, 'answer_mode': 'extractive'}]
        result = refusal_metrics(rows)
        self.assertEqual(result['refusal_recall'], 1)
        self.assertEqual(result['false_refusal_rate'], 0)
        self.assertEqual(result['answerable_response_rate'], 0)
        self.assertEqual(len(load_dataset(ROOT.parent / 'evaluation/questions.json')), 30)


class AskIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_store_module = types.ModuleType('save_to_qdrant')
        fake_store_module.SpaceScienceVectorStore = Mock(return_value=Mock())
        fake_langfuse = types.ModuleType('langfuse')
        fake_langfuse.Langfuse = Mock()
        spec = importlib.util.spec_from_file_location('api_under_test', ROOT / 'api.py')
        cls.api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'save_to_qdrant': fake_store_module, 'langfuse': fake_langfuse}), patch.dict(os.environ, {
            'GEMINI_API_KEY': '', 'OPENROUTER_API_KEY': '', 'LANGFUSE_PUBLIC_KEY': '', 'LANGFUSE_SECRET_KEY': '',
        }):
            spec.loader.exec_module(cls.api)

    def test_low_evidence_refuses_without_provider_calls(self):
        point = types.SimpleNamespace(score=99, payload={'source': 'a.pdf', 'page_number': 1, 'text': 'unrelated'})
        self.api.store.search_documents.return_value = [point]
        self.api.store.encode.return_value = [[1, 0], [0, 1]]
        with patch.object(self.api.requests, 'post') as provider:
            result = self.api.ask_question(self.api.AskRequest(question='pizza', score_threshold=0))
        self.assertTrue(result.refused)
        self.assertEqual(result.citations, [])
        provider.assert_not_called()

    def test_relevant_evidence_proceeds_in_extractive_mode(self):
        point = types.SimpleNamespace(score=.2, payload={'source': 'a.pdf', 'page_number': 1, 'text': 'telescope'})
        self.api.store.search_documents.return_value = [point]
        self.api.store.encode.return_value = [[1, 0], [1, 0]]
        with patch.object(self.api, 'rerank_documents', return_value=[point]), patch.dict(os.environ, {'GEMINI_API_KEY': '', 'OPENROUTER_API_KEY': ''}), patch.object(self.api, 'evaluate_rag_response') as judge:
            result = self.api.ask_question(self.api.AskRequest(question='telescope'))
        self.assertFalse(result.refused)
        self.assertEqual(result.answer_mode, 'extractive')
        judge.assert_not_called()

    def test_generated_answer_requires_supported_citations(self):
        point = types.SimpleNamespace(score=.8, payload={'source': 'a.pdf', 'page_number': 1, 'text': 'telescope'})
        self.api.store.search_documents.return_value = [point]
        self.api.store.encode.return_value = [[1, 0], [1, 0]]
        response = Mock(status_code=200)
        for answer, refused in [('Fact (a.pdf, Page: 1)', False), ('Fact (invented.pdf, Page: 99)', True), ('No cited facts', True)]:
            response.json.return_value = {'candidates': [{'content': {'parts': [{'text': answer}]}}]}
            with patch.object(self.api, 'rerank_documents', return_value=[point]), patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key', 'OPENROUTER_API_KEY': ''}), patch.object(self.api.requests, 'post', return_value=response), patch.object(self.api, 'evaluate_rag_response', return_value={}):
                result = self.api.ask_question(self.api.AskRequest(question='telescope'))
            self.assertEqual(result.refused, refused)
            if refused:
                self.assertEqual(result.citations, [])
                self.assertEqual(result.refusal_reason, 'unsupported_citations')

    def test_blank_question_validation(self):
        with self.assertRaises(ValueError):
            self.api.AskRequest(question='   ')


if __name__ == '__main__':
    unittest.main()
