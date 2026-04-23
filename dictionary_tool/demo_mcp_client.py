from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class StdioMCPClient:
    def __init__(self, cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        self._next_id = 1

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _write_message(self, payload: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header)
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _read_exact(self, n: int) -> bytes:
        assert self.proc.stdout is not None
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                raise RuntimeError("MCP server closed stdout unexpectedly.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_message(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = b""
                if self.proc.stderr is not None:
                    try:
                        stderr = self.proc.stderr.read() or b""
                    except Exception:
                        stderr = b""
                raise RuntimeError(f"MCP server closed stream unexpectedly. stderr={stderr.decode('utf-8', errors='ignore')}")
            if line == b"\r\n":
                break
            text = line.decode("ascii", errors="ignore").strip()
            if ":" in text:
                k, v = text.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        if "content-length" not in headers:
            raise RuntimeError(f"Missing Content-Length header: {headers}")
        body = self._read_exact(int(headers["content-length"]))
        return json.loads(body.decode("utf-8"))

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
        )
        while True:
            msg = self._read_message()
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"MCP error for {method}: {msg['error']}")
                return msg.get("result", {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal direct MCP client demo for dictionary_tool/server.py")
    ap.add_argument("--server-cmd", nargs="+", default=[sys.executable, "server.py"], help="Command used to start the MCP server.")
    ap.add_argument("--src-lang", default="eng_Latn")
    ap.add_argument("--tgt-lang", default="zho_Hans")
    ap.add_argument("--term", default="dictionary")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--lexicon-dir",
        type=Path,
        default=repo_root() / "training" / "data" / "dictionaries" / "moe_lexicon",
        help="Lexicon dir passed to the server via DICTIONARY_TOOL_LEXICON_DIR.",
    )
    args = ap.parse_args()

    tool_dir = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["DICTIONARY_TOOL_LEXICON_DIR"] = str(args.lexicon_dir.resolve())

    client = StdioMCPClient(args.server_cmd, cwd=tool_dir, env=env)
    try:
        init_result = client.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "dictionary-tool-demo", "version": "0.1.0"},
            },
        )
        client.notify("notifications/initialized", {})

        tools_result = client.request("tools/list")
        lookup_result = client.request(
            "tools/call",
            {
                "name": "lookup_dictionary",
                "arguments": {
                    "src_lang": args.src_lang,
                    "tgt_lang": args.tgt_lang,
                    "term": args.term,
                    "top_k": args.top_k,
                },
            },
        )

        print("== initialize ==")
        print(json.dumps(init_result, ensure_ascii=False, indent=2))
        print("\n== tools/list ==")
        print(json.dumps(tools_result, ensure_ascii=False, indent=2))
        print("\n== tools/call: lookup_dictionary ==")
        print(json.dumps(lookup_result, ensure_ascii=False, indent=2))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
