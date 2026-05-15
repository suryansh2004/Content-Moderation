from fastapi.testclient import TestClient

from moderation_api.main import app


class FakeModel:
    backend = "fake"
    labels = ["toxic", "threat"]

    def predict(self, texts: list[str], threshold: float):
        results = []
        for text in texts:
            scores = [0.9 if "hate" in text.lower() else 0.1, 0.2]
            labels = [
                {"label": label, "score": score, "flagged": score >= threshold}
                for label, score in zip(self.labels, scores, strict=True)
            ]
            results.append(
                {
                    "text": text,
                    "flagged": any(item["flagged"] for item in labels),
                    "max_score": max(scores),
                    "labels": labels,
                }
            )
        return results, 1.23


def test_moderate_endpoint_with_fake_model():
    app.state.model = FakeModel()
    client = TestClient(app)

    response = client.post("/moderate", json={"texts": ["hello", "I hate this"], "threshold": 0.5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "fake"
    assert payload["results"][0]["flagged"] is False
    assert payload["results"][1]["flagged"] is True
