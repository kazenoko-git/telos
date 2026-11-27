"""Managed-hosting credentials and tunnel lifecycle without loading real weights."""
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from telos.inference.hosting import HostingClient, HostingError, TunnelRuntime, _account_url


def test_session_storage_is_persistent_and_private(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    monkeypatch.setenv("TELOS_HOSTING_CREDENTIALS", str(path))
    client = HostingClient("https://telos.wingit.tech", "fixture-session")
    client.save()
    assert HostingClient.saved().token == "fixture-session"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(HostingError, match="selected account"):
        HostingClient.saved("https://other.example.com")
    client.token = "rotated-session"
    client.save()
    assert HostingClient.saved().token == "rotated-session"


@pytest.mark.parametrize("url", ["http://example.com", "https://u:p@example.com", "https://example.com/api", "https://example.com?key=x"])
def test_credentials_require_a_secure_account_origin(url):
    with pytest.raises(ValueError):
        _account_url(url)


def test_tunnel_stops_process_and_removes_credential_file(monkeypatch):
    client = Mock()
    client.request.return_value = {
        "token": "fixture-tunnel-token", "connection_id": "fixture-connection",
        "origin_key": "fixture-origin-key", "public_url": "https://tiny.tls.wingit.tech/v1",
    }
    monkeypatch.setattr("telos.inference.hosting.threading.Thread", Mock())
    process = Mock()
    process.poll.return_value = None
    with TunnelRuntime(client, "tiny", "fixture", 8000, Mock(), executable="/fixture/cloudflared") as tunnel:
        directory = Path(tunnel.directory.name)
        assert (directory / "token").read_text() == "fixture-tunnel-token"
        assert tunnel.origin_key == "fixture-origin-key"
        tunnel.process = process
    process.terminate.assert_called_once()
    assert not directory.exists()
    client.request.assert_any_call("POST", "/api/hosts/tiny/disconnect", {"connection_id": "fixture-connection"})


def test_missing_connector_fails_before_provisioning(monkeypatch):
    monkeypatch.setattr("telos.inference.hosting.shutil.which", lambda _: None)
    client = Mock()
    with pytest.raises(HostingError, match="cloudflared"):
        TunnelRuntime(client, "tiny", "fixture", 8000, Mock())
    client.request.assert_not_called()
