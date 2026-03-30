from pramana_engine.web import create_app


def test_home_page_loads():
    app = create_app()
    app.testing = True

    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Unified Pramana Workspace" in response.data


def test_main_app_page_loads():
    app = create_app()
    app.testing = True

    client = app.test_client()
    response = client.get("/app")

    assert response.status_code == 200
    assert b"Pramana Engine v2" in response.data
    assert b"INVOKE PRAMANA ENGINE" in response.data


def test_scenarios_endpoint_loads():
    app = create_app()
    app.testing = True

    client = app.test_client()
    response = client.get("/api/scenarios")

    assert response.status_code == 200
    data = response.get_json()
    assert "scenarios" in data


def test_compare_page_loads():
    app = create_app()
    app.testing = True

    client = app.test_client()
    response = client.get("/compare")

    assert response.status_code == 200
    assert b"Side-by-Side Comparator" in response.data


def test_workspace_page_loads():
    app = create_app()
    app.testing = True

    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Unified Pramana Workspace" in response.data
