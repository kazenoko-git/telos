"""Account sessions and managed Cloudflare Tunnel lifecycle."""
from contextlib import AbstractContextManager
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_ACCOUNT_URL = "https://telos.wingit.tech"


class HostingError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _account_url(url):
    parsed = urlparse(url)
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
        raise ValueError("The account service requires HTTPS.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Use the account service origin without a path or credentials.")
    return url.rstrip("/")


def _credentials_path():
    return Path(os.environ.get("TELOS_HOSTING_CREDENTIALS", Path.home() / ".config/telos/hosting.json"))


class HostingClient:
    def __init__(self, url=DEFAULT_ACCOUNT_URL, token=None):
        self.url = _account_url(url)
        self.token = token

    @classmethod
    def saved(cls, url=DEFAULT_ACCOUNT_URL):
        path = _credentials_path()
        if not path.exists():
            raise HostingError("Sign in with 'telos account login' first.")
        value = json.loads(path.read_text())
        if value.get("url") != _account_url(url):
            raise HostingError("Sign in to the selected account service first.")
        return cls(url, value["token"])

    def request(self, method, path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        try:
            with urlopen(Request(self.url + path, data=payload, headers=headers, method=method), timeout=25) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                message = json.load(exc)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = f"Account service returned HTTP {exc.code}."
            raise HostingError(message, exc.code) from None
        except (URLError, TimeoutError) as exc:
            raise HostingError("Could not reach the account service.") from exc

    def save(self):
        path = _credentials_path()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".hosting-")
        try:
            with os.fdopen(fd, "w") as target:
                json.dump({"url": self.url, "token": self.token}, target)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def login(self, email, password, signup=False):
        value = self.request("POST", "/api/signup" if signup else "/api/login", {"email": email, "password": password})
        self.token = value["token"]
        self.save()
        return {"id": value["id"], "email": value["email"]}


class TunnelRuntime(AbstractContextManager):
    """Keep a reserved endpoint online for the lifetime of the local server."""
    def __init__(self, client, slug, model, port, on_failure, executable=None):
        if not slug or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in slug):
            raise ValueError("Use the reserved lowercase tunnel subdomain.")
        self.executable = executable or shutil.which("cloudflared")
        if not self.executable:
            raise HostingError("Install cloudflared to publish a Telos endpoint.")
        self.client, self.slug, self.model, self.port = client, slug, model, port
        self.on_failure = on_failure
        self.stop = threading.Event()
        self.thread = None
        self.process = None
        self.directory = None
        self.connection = None

    @property
    def origin_key(self):
        return self.connection["origin_key"]

    def __enter__(self):
        self.connection = self.client.request("POST", f"/api/hosts/{self.slug}/connect", {
            "port": self.port, "model": self.model,
        })
        self.directory = tempfile.TemporaryDirectory(prefix="telos-tunnel-")
        token_file = Path(self.directory.name) / "token"
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as target:
            target.write(self.connection["token"])
        self.thread = threading.Thread(target=self._run, name="telos-tunnel", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            # Start publishing after the local model server is ready.
            while not self.stop.is_set():
                try:
                    with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                self.stop.wait(0.5)
            if self.stop.is_set():
                return
            token_file = Path(self.directory.name) / "token"
            self.process = subprocess.Popen([
                self.executable, "tunnel", "--no-autoupdate", "run", "--token-file", str(token_file),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            announced = False
            while not self.stop.is_set():
                if self.process.poll() is not None:
                    raise HostingError("cloudflared exited. The public endpoint has gone offline.")
                try:
                    self.client.request("POST", f"/api/hosts/{self.slug}/heartbeat", {"connection_id": self.connection["connection_id"]})
                    if not announced:
                        print(f"Public API: {self.connection['public_url']}", flush=True)
                        announced = True
                except HostingError as exc:
                    if exc.status in {401, 403, 404, 409}:
                        raise
                self.stop.wait(15)
        except Exception as exc:
            print(f"telos tunnel: {exc}", flush=True)
            self.on_failure()

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=30)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        try:
            self.client.request("POST", f"/api/hosts/{self.slug}/disconnect", {"connection_id": self.connection["connection_id"]})
        except HostingError:
            pass
        if self.directory:
            self.directory.cleanup()


def main(command):
    import argparse
    parser = argparse.ArgumentParser(prog="telos " + command)
    parser.add_argument("--account-url", default=DEFAULT_ACCOUNT_URL)
    actions = parser.add_subparsers(dest="action", required=True)
    if command == "account":
        for action in ("signup", "login"):
            sub = actions.add_parser(action)
            sub.add_argument("--email", required=True)
        actions.add_parser("whoami")
        actions.add_parser("logout")
    elif command == "keys":
        sub = actions.add_parser("create")
        sub.add_argument("name")
        actions.add_parser("list")
        for action in ("rotate", "revoke"):
            sub = actions.add_parser(action)
            sub.add_argument("id")
    else:
        sub = actions.add_parser("reserve")
        sub.add_argument("slug")
        sub.add_argument("--model", required=True)
        actions.add_parser("list")
        for action in ("share", "unshare"):
            sub = actions.add_parser(action)
            sub.add_argument("slug")
    args = parser.parse_args()
    try:
        if command == "account" and args.action in {"signup", "login"}:
            client = HostingClient(args.account_url)
            value = client.login(args.email, getpass.getpass("Password: "), signup=args.action == "signup")
        else:
            client = HostingClient.saved(args.account_url)
            if command == "account":
                if args.action == "logout":
                    value = client.request("POST", "/api/logout", {})
                    _credentials_path().unlink(missing_ok=True)
                else:
                    value = client.request("GET", "/api/me")
            elif command == "keys":
                if args.action == "create":
                    value = client.request("POST", "/api/keys", {"name": args.name})
                elif args.action == "list":
                    value = client.request("GET", "/api/keys")
                elif args.action == "rotate":
                    value = client.request("POST", f"/api/keys/{args.id}/rotate", {})
                else:
                    value = client.request("DELETE", f"/api/keys/{args.id}")
            elif args.action == "reserve":
                value = client.request("POST", "/api/hosts", {"slug": args.slug, "model": args.model})
            elif args.action == "list":
                value = client.request("GET", "/api/hosts")
            else:
                value = client.request("PATCH", f"/api/hosts/{args.slug}", {"shared": args.action == "share"})
        print(json.dumps(value, indent=2))
    except (HostingError, ValueError, OSError) as exc:
        parser.exit(2, f"telos {command}: {exc}\n")
