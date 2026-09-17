"""Command-line client for the Memora API - talks HTTP to a running server
the same way any other client would, rather than importing the pipeline
directly. That makes it a real check of what's actually deployed (request
validation, dependency wiring, error responses) instead of just the
underlying Python functions, which is what this project's earlier one-off
verification scripts exercised.

Start the server first:
    uvicorn memora.api.main:app --reload
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

import httpx

from memora.config import settings
from memora.observability.evaluation import DEFAULT_SAMPLE_SIZE, DEFAULT_TOP_K


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.api_base_url, timeout=120.0)


def _fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _request(client: httpx.Client, method: str, path: str, **kwargs: object) -> dict:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.RequestError:
        _fail(
            f"Couldn't reach the Memora server at {client.base_url}. "
            "Is it running? Start it with: uvicorn memora.api.main:app --reload"
        )

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        _fail(f"{response.status_code} {detail}")

    return response.json()


def cmd_ingest(args: argparse.Namespace) -> None:
    with _client() as client:
        body = _request(client, "POST", "/ingest/", json={"path": args.path})
    print(f"Ingested {body['chunk_count']} chunk(s) from {body['source']}")


def cmd_query(args: argparse.Namespace) -> None:
    with _client() as client:
        body = _request(client, "POST", "/query/", json={"query": args.text})
    print(body["answer"])
    if body["sources"]:
        print("\nSources:")
        for s in body["sources"]:
            print(f"  [{s['score']:.3f}] {s['source']} (chunk {s['chunk_index']})")
    print(f"\ntrace: {body['trace_id']}")


def cmd_trace(args: argparse.Namespace) -> None:
    with _client() as client:
        body = _request(client, "GET", f"/observability/traces/{args.trace_id}")
    print(f"query:     {body['query']}")
    print(f"response:  {body['response']}")
    print(f"created:   {body['created_at']}")
    print(f"retrieved: {len(body['retrieved'])} candidate(s)")
    print(f"reranked:  {len(body['reranked'])} source(s)")
    if body["conflicts"]:
        print(f"conflicts: {len(body['conflicts'])}")
        for c in body["conflicts"]:
            print(f"  {c['subject']} {c['relation']} -> {', '.join(c['objects'])}")
    else:
        print("conflicts: none")


def cmd_metrics(args: argparse.Namespace) -> None:
    with _client() as client:
        body = _request(client, "GET", "/observability/metrics")
    for key, value in body.items():
        print(f"{key}: {value}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    with _client() as client:
        body = _request(
            client,
            "POST",
            "/observability/evaluate",
            json={"top_k": args.top_k, "sample_size": args.sample_size},
        )
    print(f"recall@{body['top_k']}: {body['recall_at_k']:.2f} (sample size {body['sample_size']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memora", description="Memora CLI - a client for the Memora API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a local text/markdown/code file")
    ingest_parser.add_argument("path", help="Path to the file")
    ingest_parser.set_defaults(func=cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Ask a question against ingested memory")
    query_parser.add_argument("text", help="The question to ask")
    query_parser.set_defaults(func=cmd_query)

    trace_parser = subparsers.add_parser("trace", help="Show a recorded retrieval/response trace")
    trace_parser.add_argument("trace_id")
    trace_parser.set_defaults(func=cmd_trace)

    metrics_parser = subparsers.add_parser("metrics", help="Show aggregate memory-quality metrics")
    metrics_parser.set_defaults(func=cmd_metrics)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run a synthetic retrieval-recall check")
    evaluate_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    evaluate_parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, dest="sample_size")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
