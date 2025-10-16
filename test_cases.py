import pytest
import unittest
from api import app

# =========================
# Pytest Fixtures
# =========================
@pytest.fixture
def client():
    """Provides a Flask test client."""
    return app.test_client()

@pytest.fixture
def valid_keypoints():
    """Dummy keypoints for testing."""
    return {"keypoints": [1, 2, 3, 4]}

@pytest.fixture
def stub_correct(monkeypatch):
    import fsl_model
    monkeypatch.setattr(fsl_model, "predict_label",
        lambda s, m=None: {"label": "Auntie", "confidence": 0.93})

@pytest.fixture
def stub_wrong(monkeypatch):
    import fsl_model
    monkeypatch.setattr(fsl_model, "predict_label",
        lambda s, m=None: {"label": "Uncle", "confidence": 0.81})

# ========================================
# TEST CASE 1: Correct Sign Classification
# ========================================
def test_correct_sign(client, valid_keypoints, stub_correct):
    p = dict(valid_keypoints, expected_label="Auntie")
    r = client.post("/classify", json=p)
    d = r.get_json()
    assert r.status_code == 200 and d["is_correct"] and "guide" not in d

def test_wrong_sign_triggers_guide(client, valid_keypoints, stub_wrong):
    p = dict(valid_keypoints, expected_label="Auntie")
    r = client.post("/classify", json=p)
    g = r.get_json().get("guide")
    assert r.status_code == 200 and g and g["show_demo"] and g["sign"] == "Auntie"


# ========================================
# TEST CASE 2: Confidence Threshold
# ========================================
def classify_sign(confidence):
    if confidence >= 0.8:
        return "Correct"
    else:
        return "Show 3D Guide"

class TestClassifySign(unittest.TestCase):
    """
    Verifies confidence-based classification logic.
    """

    def test_correct_sign(self):
        result = classify_sign(0.9)
        print("\n[TEST CONFIDENCE EVALUATION - CORRECT]")
        print("Confidence: 0.9 → Result:", result)
        self.assertEqual(result, "Correct")

    def test_incorrect_sign(self):
        result = classify_sign(0.5)
        print("\n[TEST CONFIDENCE EVALUATION - INCORRECT]")
        print("Confidence: 0.5 → Result:", result)
        self.assertEqual(result, "Show 3D Guide")


# ========================================
# TEST CASE 3: Login Success
# ========================================
class TestLoginSuccess(unittest.TestCase):
    """
    Ensures login endpoint works and returns success for valid credentials.
    """

    def setUp(self):
        self.client = app.test_client()

    def test_login_success(self):
        payload = {"username": "z-user123", "password": "z-pass123"}
        response = self.client.post("/api/login", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Login successful", response.get_data(as_text=True))

        print("\nResponse JSON:", response.get_json())
        print("Status Code:", response.status_code)


# ========================================
# TEST CASE 4: Retrieve Supported Phrases
# ========================================
class TestGetPhrases(unittest.TestCase):
    """
    Verifies that /api/phrases returns a JSON list of supported phrases.
    """

    def setUp(self):
        self.client = app.test_client()

    def test_get_phrases(self):
        response = self.client.get("/api/phrases")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("phrases", data)
        self.assertIsInstance(data["phrases"], list)

        print("\nResponse JSON:", data)
        print("Number of phrases:", len(data["phrases"]))


# ========================================
# TEST CASE 5: Teach Endpoint
# ========================================
class TestTeachEndpoint(unittest.TestCase):
    """
    Checks that /api/teach returns the correct video and step information.
    """

    def setUp(self):
        self.client = app.test_client()

    def test_teach_endpoint(self):
        response = self.client.get("/api/teach?phrase=hello")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("video", data)
        self.assertIn("steps", data)
        self.assertTrue(data["video"].endswith(".mp4"))

        print("\nResponse JSON:", data)
        print("Video file:", data["video"])
        print("Number of steps:", len(data["steps"]))


# ========================================
# TEST CASE 6: Signup Endpoint
# ========================================
class TestUserSignup(unittest.TestCase):
    """
    Tests whether a new user can successfully register in the system.
    """

    def setUp(self):
        self.client = app.test_client()

    def test_user_signup(self):
        payload = {"username": "z-user123", "password": "z-pass123", "confirm_password": "z-pass123"}
        response = self.client.post("/api/signup", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("User registered successfully", response.get_data(as_text=True))

        print("\nResponse JSON:", response.get_json())
        print("Status Code:", response.status_code)


# ========================================
# TEST CASE 7: Assessment Stub
# ========================================
class TestAssessEndpoint(unittest.TestCase):
    """
    Verifies that the assess endpoint correctly compares attempted phrases.
    """

    def setUp(self):
        self.client = app.test_client()

    def test_assess_endpoint(self):
        payload = {"target": "good morning", "attempt_text": "good morning"}
        response = self.client.post("/api/assess", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("correct", data)
        self.assertIn("score", data)
        self.assertTrue(data["correct"])

        print("\nResponse JSON:", data)
        print("Status Code:", response.status_code)
        print("Correct:", data.get("correct"))
        print("Score:", data.get("score"))


if __name__ == "__main__":
    unittest.main()
