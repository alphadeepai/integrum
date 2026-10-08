"""Unit tests for IntegrumServer and tool generation."""

import pytest
import json
from integrum.server import IntegrumServer


def sample_multiply(a: int, b: int = 2) -> int:
    """Multiply two integers."""
    return a * b


class Counter:
    """Simple stateful counter."""

    def __init__(self, initial: int = 0):
        self.count = initial

    def increment(self, step: int = 1) -> int:
        """Increment counter by step."""
        self.count += step
        return self.count

    def get_value(self) -> int:
        """Return current count."""
        return self.count


@pytest.mark.asyncio
async def test_server_function_exposure():
    server = IntegrumServer(name="test-server")
    tool_name = server.expose_function(sample_multiply)
    assert tool_name == "sample_multiply"

    # Call tool via FastMCP
    res = await server.mcp.call_tool("sample_multiply", {"a": 5, "b": 3})
    assert not res.is_error
    assert res.structured_content == 15 or "15" in str(res.content)


@pytest.mark.asyncio
async def test_server_class_stateful_lifecycle():
    server = IntegrumServer(name="stateful-server")
    tools = server.expose_class(Counter)
    assert "Counter_create" in tools
    assert "Counter_increment" in tools

    # 1. Create counter instance
    create_res = await server.mcp.call_tool("Counter_create", {"initial": 10})
    assert not create_res.is_error
    
    # Extract returned object_id from structured content or text
    data = create_res.structured_content
    if isinstance(data, dict):
        obj_id = data.get("object_id")
    else:
        # parsed text content
        text = create_res.content[0].text
        data = json.loads(text) if "{" in text else {}
        obj_id = data.get("object_id")
    
    assert obj_id is not None
    assert server.store.has(obj_id)

    # 2. Call increment on created instance
    inc_res = await server.mcp.call_tool("Counter_increment", {"obj_id": obj_id, "step": 5})
    assert not inc_res.is_error
    assert inc_res.structured_content == 15 or "15" in str(inc_res.content)

    # 3. Verify underlying instance in store
    instance = server.store.get(obj_id)
    assert instance.count == 15


@pytest.mark.asyncio
async def test_server_sklearn_tools_direct():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    server = IntegrumServer(name="ml-server")
    server.expose_function(load_iris)
    server.expose_class(RandomForestClassifier)

    # 1. Call load_iris
    iris_res = await server.mcp.call_tool("load_iris", {})
    assert not iris_res.is_error
    text = iris_res.content[0].text
    iris_data = json.loads(text) if "{" in text else iris_res.structured_content
    iris_id = iris_data["object_id"]
    assert server.store.has(iris_id)
    assert server.store.has(f"{iris_id}.data")
    assert server.store.has(f"{iris_id}.target")

    # 2. Create RandomForestClassifier
    rf_res = await server.mcp.call_tool("RandomForestClassifier_create", {"n_estimators": 10, "random_state": 42})
    assert not rf_res.is_error
    rf_text = rf_res.content[0].text
    rf_data = json.loads(rf_text) if "{" in rf_text else rf_res.structured_content
    rf_id = rf_data["object_id"]
    assert server.store.has(rf_id)

    # 3. Fit RandomForestClassifier using dot notation object IDs
    fit_res = await server.mcp.call_tool(
        "RandomForestClassifier_fit",
        {
            "obj_id": rf_id,
            "X": f"{iris_id}.data",
            "y": f"{iris_id}.target",
        },
    )
    assert not fit_res.is_error

    # Verify classifier is fitted in the store!
    fitted_clf = server.store.get(rf_id)
    assert hasattr(fitted_clf, "estimators_")
    assert len(fitted_clf.estimators_) == 10


@pytest.mark.asyncio
async def test_server_train_test_split_and_score():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    server = IntegrumServer(name="ml-split-server")
    server.expose_function(load_iris)
    server.expose_function(train_test_split)
    server.expose_class(RandomForestClassifier)

    # 1. Load Iris
    iris_res = await server.mcp.call_tool("load_iris", {})
    text = iris_res.content[0].text
    iris_data = json.loads(text) if "{" in text else iris_res.structured_content
    iris_id = iris_data["object_id"]

    # 2. train_test_split
    split_res = await server.mcp.call_tool(
        "train_test_split",
        {
            "arrays": [f"{iris_id}.data", f"{iris_id}.target"],
            "test_size": 0.25,
            "random_state": 42,
        },
    )
    assert not split_res.is_error
    split_text = split_res.content[0].text
    split_data = json.loads(split_text) if "{" in split_text else split_res.structured_content
    split_id = split_data["object_id"]

    assert server.store.has(f"{split_id}.X_train")
    assert server.store.has(f"{split_id}.X_test")
    assert server.store.has(f"{split_id}.y_train")
    assert server.store.has(f"{split_id}.y_test")

    # 3. Create RF
    rf_res = await server.mcp.call_tool("RandomForestClassifier_create", {"n_estimators": 20, "random_state": 42})
    rf_text = rf_res.content[0].text
    rf_data = json.loads(rf_text) if "{" in rf_text else rf_res.structured_content
    rf_id = rf_data["object_id"]

    # 4. Fit on train set
    fit_res = await server.mcp.call_tool(
        "RandomForestClassifier_fit",
        {
            "obj_id": rf_id,
            "X": f"{split_id}.X_train",
            "y": f"{split_id}.y_train",
        },
    )
    assert not fit_res.is_error

    # 5. Score on test set
    score_res = await server.mcp.call_tool(
        "RandomForestClassifier_score",
        {
            "obj_id": rf_id,
            "X": f"{split_id}.X_test",
            "y": f"{split_id}.y_test",
        },
    )
    assert not score_res.is_error
    score = float(score_res.content[0].text)
    assert 0.8 <= score <= 1.0

