"""Personal client sessions must not share an Authorization header."""

import httpx

from railguard.ui.client import RailGuardApiClient


def test_personal_clients_keep_separate_credentials():
    captured = []
    def respond(request):
        captured.append((request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json={"subject": "test", "role": "submitter"})
    transport = httpx.MockTransport(respond)
    first = RailGuardApiClient(base_url="http://local", token="test-user-A", transport=transport)
    second = RailGuardApiClient(base_url="http://local", token="test-user-B", transport=transport)
    anonymous = RailGuardApiClient(base_url="http://local", transport=transport)
    try:
        first.identity()
        second.identity()
        anonymous.health()
    finally:
        first.close()
        second.close()
        anonymous.close()
    assert captured == [("/security/me", "Bearer test-user-A"),
                        ("/security/me", "Bearer test-user-B"), ("/health", None)]
