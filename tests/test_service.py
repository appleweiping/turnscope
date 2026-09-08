import json
import threading
import urllib.request

from turnscope import TurnScopeService, create_server


def _source(tmp_path):
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "id": "demo",
                "utterances": [
                    {
                        "id": "u1",
                        "role": "user",
                        "text": "where is it",
                        "timestamp": "2026-01-01T00:00:00Z",
                    },
                    {
                        "id": "a1",
                        "role": "assistant",
                        "text": "it is here",
                        "timestamp": "2026-01-01T00:00:01Z",
                        "reply_to": "u1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_service_build_tabular_audit_and_search(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = _source(tmp_path)
    service = TurnScopeService()
    built = service.dispatch({"operation": "build", "input": str(source), "value": 1})
    assert built["windows"][-1]["target"]["id"] == "a1"
    rows = service.dispatch({"operation": "tabular", "input": str(source), "value": 1})
    assert rows["rows"][-1]["context_ids"] == '["u1"]'
    assert service.dispatch({"operation": "audit", "input": str(source)})["passed"]
    assert service.dispatch({"operation": "search", "input": str(source), "query": "where"})["hits"]


def test_service_http_dispatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    server = create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/dispatch",
            data=json.dumps({"operation": "audit", "input": str(_source(tmp_path))}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["passed"] is True
    finally:
        server.shutdown()
        server.server_close()
