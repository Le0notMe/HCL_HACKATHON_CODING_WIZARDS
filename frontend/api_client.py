"""HTTP client for the ticket API, used by the Streamlit frontend."""

from typing import Any

import httpx

CHAT_TIMEOUT_SECONDS = 240.0


class ApiError(Exception):
    """A failed API call, with a message that is safe to show the user."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _error_message(response: httpx.Response) -> str:
    """Return the API's ``detail``, flattening FastAPI's list of validation errors."""
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return f"The API returned status {response.status_code}"
    if isinstance(detail, list):
        return "; ".join(f"{item['loc'][-1]}: {item['msg']}" for item in detail)
    return str(detail)


class ApiClient:
    """Calls the API with one caller's key. Every failure becomes an ``ApiError``."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 15.0):
        self.base_url = base_url
        self.http = httpx.Client(
            base_url=base_url, headers={"X-API-Key": api_key}, timeout=timeout
        )

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(f"The API at {self.base_url} timed out") from exc
        except httpx.TransportError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Is uvicorn running?") from exc
        if response.is_error:
            raise ApiError(_error_message(response), response.status_code)
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def analyze(self, ticket: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/v1/tickets/analyze", json=ticket)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/runs/{httpx.URL(path=run_id)}")

    def submit_feedback(self, run_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/runs/{httpx.URL(path=run_id)}/feedback", json=feedback)

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send the conversation. This waits for the model, so it allows a longer timeout."""
        return self._request("POST", "/api/v1/chat", json={"messages": messages}, timeout=CHAT_TIMEOUT_SECONDS)

    def search(self, **filters: Any) -> dict[str, Any]:
        """Search tickets. Filters left as ``None`` or blank are not sent."""
        params = {key: value for key, value in filters.items() if value not in (None, "")}
        return self._request("GET", "/api/v1/tickets/search", params=params)
