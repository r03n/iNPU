import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from intelflow.cli import app
from intelflow.server import app as fastapi_app
from intelflow.config import MODEL_REGISTRY, DEFAULT_PORT
from intelflow.hub import ModelHub

runner = CliRunner()
client = TestClient(fastapi_app)


def test_cli_port():
    res = runner.invoke(app, ["port"])
    assert res.exit_code == 0
    assert str(DEFAULT_PORT) in res.stdout


def test_cli_validate_json():
    res = runner.invoke(app, ["validate", "--json"])
    assert res.exit_code == 0
    assert "drm_node" in res.stdout
    assert "architecture" in res.stdout


def test_cli_list_json():
    res = runner.invoke(app, ["list", "--json"])
    assert res.exit_code == 0
    assert "qwen2.5" in res.stdout


def test_cli_check():
    # qwen2.5 was cached earlier
    res = runner.invoke(app, ["check", "qwen2.5", "--json"])
    assert res.exit_code == 0
    assert "installed" in res.stdout


def test_catalog_expansion():
    catalog = ModelHub.list_catalog()
    assert "qwen2.5" in catalog
    assert "llama3.2:1b" in catalog
    assert "gemma2:2b" in catalog
    assert "phi3.5:mini" in catalog
    assert "bge-small" in catalog


def test_server_version():
    resp_v1 = client.get("/v1/version")
    assert resp_v1.status_code == 200
    assert resp_v1.json()["backend"] == "openvino-npu"

    resp_api = client.get("/api/version")
    assert resp_api.status_code == 200
    assert resp_api.json()["backend"] == "openvino-npu"


def test_server_ollama_tags():
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_server_ollama_ps():
    resp = client.get("/api/ps")
    assert resp.status_code == 200
    assert "models" in resp.json()


def test_server_npu_status():
    resp = client.get("/api/npu/status")
    assert resp.status_code == 200
    assert "npu_available" in resp.json()


if __name__ == "__main__":
    print("Running test_cli_port...")
    test_cli_port()
    print("Running test_cli_validate_json...")
    test_cli_validate_json()
    print("Running test_cli_list_json...")
    test_cli_list_json()
    print("Running test_cli_check...")
    test_cli_check()
    print("Running test_catalog_expansion...")
    test_catalog_expansion()
    print("Running test_server_version...")
    test_server_version()
    print("Running test_server_ollama_tags...")
    test_server_ollama_tags()
    print("Running test_server_ollama_ps...")
    test_server_ollama_ps()
    print("Running test_server_npu_status...")
    test_server_npu_status()
    print("ALL 9 PARITY TESTS PASSED!")
