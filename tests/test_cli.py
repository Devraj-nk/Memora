import httpx
import pytest

import memora.cli as cli_module
from memora.cli import build_parser


def _client_returning(handler) -> httpx.Client:
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def _run(monkeypatch: pytest.MonkeyPatch, handler, argv: list[str]) -> None:
    monkeypatch.setattr(cli_module, "_client", lambda: _client_returning(handler))
    args = build_parser().parse_args(argv)
    args.func(args)


def test_ingest_prints_chunk_count(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ingest/"
        assert request.method == "POST"
        return httpx.Response(200, json={"source": "note.md", "chunk_count": 3})

    _run(monkeypatch, handler, ["ingest", "note.md"])

    assert "Ingested 3 chunk(s) from note.md" in capsys.readouterr().out


def test_query_prints_answer_sources_and_trace_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/query/"
        return httpx.Response(
            200,
            json={
                "answer": "LanceDB.",
                "sources": [{"text": "...", "source": "note.md", "chunk_index": 0, "score": 0.9}],
                "trace_id": "abc123",
            },
        )

    _run(monkeypatch, handler, ["query", "what vector store?"])

    out = capsys.readouterr().out
    assert "LanceDB." in out
    assert "note.md (chunk 0)" in out
    assert "trace: abc123" in out


def test_query_with_no_sources_omits_sources_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "I don't know.", "sources": [], "trace_id": "abc123"})

    _run(monkeypatch, handler, ["query", "anything"])

    assert "Sources:" not in capsys.readouterr().out


def test_trace_prints_conflicts(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/observability/traces/t1"
        return httpx.Response(
            200,
            json={
                "trace_id": "t1",
                "query": "where does alice live?",
                "retrieved": [{}],
                "reranked": [{}],
                "conflicts": [{"subject": "Alice", "relation": "lives in", "objects": ["Paris", "Tokyo"]}],
                "context": "ctx",
                "response": "unclear",
                "created_at": "2026-09-16T00:00:00+00:00",
            },
        )

    _run(monkeypatch, handler, ["trace", "t1"])

    out = capsys.readouterr().out
    assert "conflicts: 1" in out
    assert "Alice lives in -> Paris, Tokyo" in out


def test_metrics_prints_each_field(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/observability/metrics"
        return httpx.Response(200, json={"trace_count": 5, "avg_sources_per_trace": 1.5})

    _run(monkeypatch, handler, ["metrics"])

    out = capsys.readouterr().out
    assert "trace_count: 5" in out
    assert "avg_sources_per_trace: 1.5" in out


def test_evaluate_prints_recall(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/observability/evaluate"
        return httpx.Response(200, json={"recall_at_k": 0.8, "sample_size": 10, "top_k": 5})

    _run(monkeypatch, handler, ["evaluate"])

    assert "recall@5: 0.80 (sample size 10)" in capsys.readouterr().out


def test_http_error_prints_detail_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Trace not found: missing"})

    with pytest.raises(SystemExit) as exc_info:
        _run(monkeypatch, handler, ["trace", "missing"])

    assert exc_info.value.code == 1
    assert "Trace not found: missing" in capsys.readouterr().err


def test_connection_error_gives_a_helpful_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(SystemExit) as exc_info:
        _run(monkeypatch, handler, ["metrics"])

    assert exc_info.value.code == 1
    assert "Is it running?" in capsys.readouterr().err
