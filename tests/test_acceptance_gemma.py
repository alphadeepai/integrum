"""End-to-end acceptance test: Gemma 4 26B + Integrum + scikit-learn (train_test_split + accuracy)."""

import asyncio
import json
import logging
import os
import subprocess
import time
import pytest
from dotenv import load_dotenv
import openai

from integrum.server import IntegrumServer

logger = logging.getLogger("integrum.acceptance")


def get_active_api_key() -> str:
    """Retrieve API key from .env, falling back to fresh gcloud token if expired."""
    load_dotenv(override=False)
    token = os.getenv("AI_API_KEY", "")

    try:
        cmd = ["gcloud", "auth", "print-access-token"]
        gcloud_token = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        if gcloud_token:
            return gcloud_token
    except Exception:
        pass

    return token


@pytest.mark.asyncio
async def test_gemma_trains_random_forest_on_iris_with_split_and_accuracy():
    """Acceptance test: Expose load_iris, train_test_split, RandomForestClassifier,

    and verify that Gemma 4 26B uses the tools to:
    1. Load iris dataset
    2. Split into train and test sets
    3. Instantiate a RandomForestClassifier
    4. Fit the classifier on the training set
    5. Evaluate and report accuracy on the test set
    """
    load_dotenv()
    base_url = os.getenv("AI_BASE_URL")
    model = os.getenv("AI_MODEL")
    api_key = get_active_api_key()

    if not base_url or not model or not api_key:
        pytest.skip("Gemma endpoint credentials not found in .env; skipping acceptance test.")

    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    # 1. Initialize IntegrumServer and expose scikit-learn modules
    server = IntegrumServer(name="sklearn-mcp")
    server.expose(
        [
            "sklearn.datasets.load_iris",
            "sklearn.model_selection.train_test_split",
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.metrics.accuracy_score",
        ],
        include="^(load_iris|train_test_split|fit|predict|score|accuracy_score)$",
    )

    # 2. Extract tools in OpenAI format
    mcp_tools = await server.mcp.list_tools()
    openai_tools = []
    for t in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        })

    tool_names = [t["function"]["name"] for t in openai_tools]
    logger.info(f"Exposed OpenAI tools: {tool_names}")
    assert "load_iris" in tool_names
    assert "train_test_split" in tool_names
    assert "RandomForestClassifier_create" in tool_names
    assert "RandomForestClassifier_fit" in tool_names
    assert "RandomForestClassifier_score" in tool_names

    # 3. Setup multi-turn conversation
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI data scientist using tools from an Integrum MCP server.\n"
                "Key Rules:\n"
                "- When a tool returns an object, it assigns an object ID (e.g. 'lucid_yak').\n"
                "- Use 'obj_id' to call methods on that object (e.g. RandomForestClassifier_fit, RandomForestClassifier_score).\n"
                "- To split data with train_test_split, pass arrays as a list, e.g. arrays=['<id>.data', '<id>.target'] "
                "(or arrays=['<id>_0', '<id>_1']), along with test_size (e.g. 0.2) and random_state.\n"
                "- train_test_split returns 4 arrays: X_train, X_test, y_train, y_test (accessible via '<split_id>.X_train', '<split_id>.X_test', etc.).\n"
                "- Fit the classifier on the training set (X_train, y_train).\n"
                "- Evaluate test accuracy using RandomForestClassifier_score (passing obj_id, X='<split_id>.X_test', y='<split_id>.y_test') "
                "or accuracy_score.\n"
                "- Always report the final accuracy score in your final reply."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please perform the following end-to-end task using the tools:\n"
                "1. Load the iris dataset using load_iris.\n"
                "2. Split the dataset into train and test sets using train_test_split (test_size=0.2, random_state=42).\n"
                "3. Create a RandomForestClassifier instance.\n"
                "4. Train / fit the classifier on the training set (X_train and y_train).\n"
                "5. Evaluate accuracy on the test set (X_test and y_test) and report the final test accuracy."
            ),
        },
    ]

    # Helper for API call with retry on rate limit (429)
    def call_llm_with_retry(msgs):
        max_retries = 10
        delay = 5.0
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    tools=openai_tools,
                    tool_choice="auto",
                )
            except (openai.RateLimitError, openai.APIStatusError) as e:
                status = getattr(e, "status_code", 429)
                if status in (429, 503):
                    if attempt == max_retries - 1:
                        raise
                    print(f"Endpoint throttled (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 1.5, 20.0)
                else:
                    raise

    # 4. Multi-turn agent loop
    max_turns = 12
    score_evaluated = False
    assistant_replies = []

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")
        time.sleep(3)  # Gentle pacing between turns to avoid burst concurrency limit
        response = call_llm_with_retry(messages)
        choice = response.choices[0]
        message = choice.message

        msg_dict = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(msg_dict)

        if not message.tool_calls:
            print(f"Assistant replied: {message.content}")
            if message.content:
                assistant_replies.append(message.content)

            # Check if training and scoring are done
            has_fitted = any(
                hasattr(obj, "estimators_") and len(obj.estimators_) > 0
                for obj in server.store._sessions.get("default", {}).values()
            )
            if has_fitted and (score_evaluated or "accuracy" in (message.content or "").lower()):
                break

            messages.append({
                "role": "user",
                "content": "Please continue and report the test set accuracy score using the tools.",
            })
            continue

        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                fn_args = {}

            print(f"Tool Call: {fn_name}({fn_args})")
            tool_result = await server.mcp.call_tool(fn_name, fn_args)

            if tool_result.content:
                res_content = tool_result.content[0].text
            else:
                res_content = json.dumps(tool_result.structured_content or {})

            print(f"Tool Result: {res_content}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": res_content,
            })

            if ("score" in fn_name.lower()) and not tool_result.is_error:
                score_evaluated = True

    # 5. Assertions & Verification
    # A. Check that a fitted RandomForestClassifier exists in the store
    fitted_models = [
        obj for obj in server.store._sessions.get("default", {}).values()
        if hasattr(obj, "estimators_") and len(obj.estimators_) > 0
    ]
    assert len(fitted_models) >= 1, "No fitted RandomForestClassifier found in ObjectStore!"
    fitted_clf = fitted_models[0]
    assert hasattr(fitted_clf, "n_classes_")
    assert fitted_clf.n_classes_ == 3

    # B. Check that train_test_split was executed and test sets exist
    session_store = server.store._sessions.get("default", {})
    x_test_keys = [k for k in session_store if k.endswith(".X_test")]
    y_test_keys = [k for k in session_store if k.endswith(".y_test")]
    assert len(x_test_keys) >= 1, "X_test not found in ObjectStore!"
    assert len(y_test_keys) >= 1, "y_test not found in ObjectStore!"

    X_test = session_store[x_test_keys[0]]
    y_test = session_store[y_test_keys[0]]
    calculated_accuracy = fitted_clf.score(X_test, y_test)
    print(f"\nCalculated test accuracy on test set: {calculated_accuracy:.4f}")
    assert calculated_accuracy >= 0.85, f"Expected accuracy >= 0.85, got {calculated_accuracy}"

    # C. Verify assistant output
    combined_replies = " ".join(assistant_replies).lower()
    print(f"Final Assistant Replies: {combined_replies}")
    assert score_evaluated or any(term in combined_replies for term in ["accuracy", "1.0", "0.9", "100%", "score"]), (
        "Assistant did not report test accuracy!"
    )

    print(f"\nSUCCESS: Verified end-to-end train_test_split, training, and test accuracy evaluation ({calculated_accuracy:.4f}) with Gemma 4 26B!")
