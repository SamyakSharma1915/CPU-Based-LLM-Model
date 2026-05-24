import argparse
import json
import os
import secrets
import time
from pathlib import Path
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse, quote
from urllib.request import Request, urlopen

from .generate import load_model, sample

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
INDEX_HTML = WEB_DIR / "index.html"


class TinyGPTWebApp:
    def __init__(
        self,
        ckpt,
        temp,
        top_k,
        top_p,
        json_mode,
        max_new_tokens,
        google_client_id=None,
        google_client_secret=None,
        public_base_url=None,
    ):
        self.model, self.tokenizer = load_model(ckpt, device="cpu")
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.json_mode = json_mode
        self.max_new_tokens = max_new_tokens
        self.google_client_id = google_client_id or os.getenv("GOOGLE_CLIENT_ID", "")
        self.google_client_secret = google_client_secret or os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.public_base_url = (public_base_url or os.getenv("PUBLIC_BASE_URL", "")).rstrip("/")
        self.oauth_states = {}

    def reply(self, user_message):
        prompt = f"<|user|> {user_message} <|assistant|>"
        return sample(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temp,
            top_k=self.top_k,
            top_p=self.top_p,
            json_mode=self.json_mode,
        )

    def google_auth_enabled(self):
        return bool(self.google_client_id and self.google_client_secret)

    def get_base_url(self, handler):
        if self.public_base_url:
            return self.public_base_url
        host = handler.headers.get("Host", f"127.0.0.1:{handler.server.server_port}")
        return f"http://{host}"

    def get_google_redirect_uri(self, handler):
        return f"{self.get_base_url(handler)}/auth/google/callback"

    def create_oauth_state(self):
        state = secrets.token_urlsafe(24)
        self.oauth_states[state] = time.time()
        return state

    def validate_oauth_state(self, state):
        created_at = self.oauth_states.pop(state, None)
        if created_at is None:
            return False
        return (time.time() - created_at) <= 600

    def exchange_google_code(self, handler, code):
        token_request = urlencode(
            {
                "code": code,
                "client_id": self.google_client_id,
                "client_secret": self.google_client_secret,
                "redirect_uri": self.get_google_redirect_uri(handler),
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")

        request = Request(
            "https://oauth2.googleapis.com/token",
            data=token_request,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(request, timeout=20) as response:
            token_payload = json.loads(response.read().decode("utf-8"))

        access_token = token_payload["access_token"]
        user_request = Request(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urlopen(user_request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_html(self, html, status=HTTPStatus.OK):
            data = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.end_headers()

        def _serve_file(self, path):
            self._write_html(path.read_text(encoding="utf-8"))

        def _google_error_page(self, message):
            safe = (
                message.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            self._write_html(
                f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Google Auth Error</title></head>
<body style="font-family: monospace; background:#111; color:#eee; padding:24px">
<h2>Google auth failed</h2>
<p>{safe}</p>
<p><a href="/klyrex-ai.html" style="color:#00ff87">Return to Klyrex AI</a></p>
</body>
</html>""",
                status=HTTPStatus.BAD_REQUEST,
            )

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path in ("/", "/index.html"):
                self._serve_file(INDEX_HTML)
                return
            if parsed.path == "/auth/google/start":
                if not app.google_auth_enabled():
                    self._google_error_page(
                        "Google OAuth is not configured on the server. "
                        "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET before starting tinygpt.web."
                    )
                    return

                state = app.create_oauth_state()
                query = urlencode(
                    {
                        "client_id": app.google_client_id,
                        "redirect_uri": app.get_google_redirect_uri(self),
                        "response_type": "code",
                        "scope": "openid email profile",
                        "access_type": "offline",
                        "prompt": "consent",
                        "state": state,
                    }
                )
                self._redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{query}")
                return

            if parsed.path == "/auth/google/callback":
                query = parse_qs(parsed.query)
                error = query.get("error", [""])[0]
                code = query.get("code", [""])[0]
                state = query.get("state", [""])[0]

                if error:
                    self._google_error_page(f"Google returned an error: {error}")
                    return
                if not code or not state:
                    self._google_error_page("Missing code or state in Google callback.")
                    return
                if not app.validate_oauth_state(state):
                    self._google_error_page("Invalid or expired Google OAuth state.")
                    return

                try:
                    userinfo = app.exchange_google_code(self, code)
                except HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    self._google_error_page(f"Google token exchange failed: {detail}")
                    return
                except Exception as exc:
                    self._google_error_page(str(exc))
                    return

                name = quote(userinfo.get("name", "Google User"), safe="")
                email = quote(userinfo.get("email", ""), safe="")
                redirect = f"/klyrex-ai.html?auth=success&name={name}&email={email}"
                self._redirect(redirect)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
                message = str(payload.get("message", "")).strip()
                if not message:
                    self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Message is required."})
                    return

                reply = app.reply(message)
                self._write_json(HTTPStatus.OK, {"reply": reply})
            except json.JSONDecodeError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON."})
            except Exception as exc:
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def log_message(self, fmt, *args):
            return

    return Handler


def serve(
    ckpt,
    host,
    port,
    temp,
    top_k,
    top_p,
    json_mode,
    max_new_tokens,
    google_client_id=None,
    google_client_secret=None,
    public_base_url=None,
):
    app = TinyGPTWebApp(
        ckpt,
        temp,
        top_k,
        top_p,
        json_mode,
        max_new_tokens,
        google_client_id=google_client_id,
        google_client_secret=google_client_secret,
        public_base_url=public_base_url,
    )
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"TinyGPT web chat running at http://{host}:{port}")
    if app.google_auth_enabled():
        print(f"Google OAuth enabled at {app.get_google_redirect_uri(type('H', (), {'headers': {}, 'server': type('S', (), {'server_port': port})()})())}")
    else:
        print("Google OAuth disabled. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to enable it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web chat.")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="fine_tuned.pt", help="Path to model checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max", type=int, default=256)
    parser.add_argument("--google-client-id", default=None)
    parser.add_argument("--google-client-secret", default=None)
    parser.add_argument("--public-base-url", default=None)
    args = parser.parse_args()
    serve(
        args.ckpt,
        args.host,
        args.port,
        args.temp,
        args.top_k,
        args.top_p,
        args.json,
        args.max,
        google_client_id=args.google_client_id,
        google_client_secret=args.google_client_secret,
        public_base_url=args.public_base_url,
    )
