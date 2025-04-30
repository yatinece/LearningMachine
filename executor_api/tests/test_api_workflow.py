import pytest
import requests
import time
import pandas as pd
from sklearn.datasets import make_classification

# Base URL for the API under test
BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="session")
def session():
    """Provides a requests Session for the test suite."""
    return requests.Session()

@pytest.fixture(scope="session")
def test_data_csv(tmp_path):
    """Generates and saves a classification dataset for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    path = tmp_path / "test_classification.csv"
    df.to_csv(path, index=False)
    return str(path)

@pytest.fixture(scope="session")
def user(session):
    """Creates a test user and returns its details."""
    user_payload = {
        "user_id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "meta_data": {
            "role": "data_scientist",
            "department": "research"
        }
    }
    resp = session.post(f"{BASE_URL}/users/", json=user_payload)
    resp.raise_for_status()
    return resp.json()

@pytest.fixture(scope="session")
def project(session, user):
    """Creates a test project under the given user."""
    project_payload = {
        "name": "Test ML Project",
        "description": "A test project for ML experiments",
        "config": {
            "description": "Project for testing XGBoost classification",
            "tags": ["testing", "classification", "xgboost"]
        }
    }
    resp = session.post(
        f"{BASE_URL}/users/{user['user_id']}/projects/",
        json=project_payload
    )
    resp.raise_for_status()
    return resp.json()

def test_user_and_project_creation(user, project):
    """Validates that user and project were created successfully."""
    assert user.get('user_id') == 1, "Unexpected user_id"
    assert user.get('username') == 'testuser', "Username mismatch"
    assert project.get('name') == 'Test ML Project', "Project name mismatch"

def test_ml_experiment_workflow(session, project, test_data_csv):
    """Tests the full ML experiment flow: upload, status polling, results, and prediction."""
    # Upload dataset and create experiment
    with open(test_data_csv, 'rb') as f:
        files = {"file": (test_data_csv.split('/')[-1], f, "text/csv")}
        data = {
            "name": "Test Classification Experiment",
            "description": "Testing XGBoost classification with auto feature detection",
            "target_feature": "target",
            "model_type": "classification"
        }
        resp = session.post(
            f"{BASE_URL}/experiments/{project['id']}/upload",
            files=files,
            data=data
        )
    resp.raise_for_status()
    exp = resp.json()
    exp_id = exp.get("experiment_id")
    assert exp_id is not None, "No experiment_id returned"

    # Poll for completion
    status = None
    for _ in range(24):  # up to 2 minutes polling every 5 seconds
        time.sleep(5)
        r = session.get(
            f"{BASE_URL}/experiments/{project['id']}/experiment/{exp_id}"
        )
        r.raise_for_status()
        status = r.json().get("status")
        if status in ["completed", "failed"]:
            break
    assert status == "completed", f"Experiment did not complete, last status: {status}"

    details = r.json()
    assert details['status'] == 'completed', "Experiment status is not completed"
    assert 'results' in details, "Results missing from experiment details"

    # Test prediction endpoint
    df = pd.read_csv(test_data_csv)
    prediction_payload = df.drop(columns=["target"]).head(5).to_dict(orient="records")
    pred_resp = session.post(
        f"{BASE_URL}/experiments/{project['id']}/experiment/{exp_id}/predict",
        json={"data": prediction_payload}
    )
    pred_resp.raise_for_status()
    pred_data = pred_resp.json()
    assert 'results' in pred_data, "Prediction results missing"
