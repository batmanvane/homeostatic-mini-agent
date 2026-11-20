import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from homeo.ollama import OllamaLM


class FakeOllama(BaseHTTPRequestHandler):
    requests: list[dict] = []
    top: list[tuple[str, float]] = []
    content = "1"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        top = [{"token": t, "logprob": lp} for t, lp in FakeOllama.top]
        reply = {
            "message": {"role": "assistant", "content": FakeOllama.content},
            "logprobs": [{"token": FakeOllama.content, "logprob": -0.1, "top_logprobs": top}],
            "done": True,
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeOllama.requests = []
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_choose_reads_digit_probabilities(server):
    FakeOllama.top = [("2", math.log(0.6)), ("1", math.log(0.3)), (" 3", math.log(0.05)), ("Hello", math.log(0.04))]
    lm = OllamaLM("some-model", host=server)
    choice = lm.choose("prompt", 4)
    assert choice.probs == pytest.approx([0.3 / 0.95, 0.6 / 0.95, 0.05 / 0.95, 0.0])
    assert choice.mass == pytest.approx(0.95)
    assert choice.hidden is None


def test_choose_requests_one_deterministic_token_without_thinking(server):
    FakeOllama.top = [("1", 0.0)]
    OllamaLM("m", host=server).choose("hello", 3)
    req = FakeOllama.requests[-1]
    assert req["model"] == "m"
    assert req["messages"] == [{"role": "user", "content": "hello"}]
    assert req["logprobs"] is True and req["top_logprobs"] >= 10
    assert req["think"] is False and req["stream"] is False
    assert req["options"]["num_predict"] == 1 and req["options"]["temperature"] == 0


def test_choose_fails_loudly_when_no_digit_in_top_tokens(server):
    FakeOllama.top = [("Hello", 0.0)]
    with pytest.raises(RuntimeError):
        OllamaLM("m", host=server).choose("prompt", 4)


def test_sample_and_answer_return_text(server):
    FakeOllama.content = "I will ask the operator."
    lm = OllamaLM("m", host=server)
    assert lm.sample("p", seed=5) == "I will ask the operator."
    assert FakeOllama.requests[-1]["options"]["seed"] == 5
    assert lm.answer("p") == "I will ask the operator."
    assert FakeOllama.requests[-1]["options"]["temperature"] == 0
