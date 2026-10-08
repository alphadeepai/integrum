# Example: End-to-End ML Pipeline with Gemma 4 26B & Integrum

This example documents the exact multi-turn chat and tool calling session between **Gemma 4 26B** and the **Integrum FastMCP Server** exposing `scikit-learn`.

The model loads the Iris dataset, performs a train-test split, instantiates a `RandomForestClassifier`, fits the classifier on `X_train` and `y_train`, evaluates accuracy on `X_test` and `y_test`, and reports the final result.

---

## 1. System Prompt & Setup

### Exposed Tools
- `load_iris`: Loads Iris dataset.
- `train_test_split`: Splits arrays into `X_train`, `X_test`, `y_train`, `y_test`.
- `RandomForestClassifier_create`: Instantiates the classifier and returns an object ID.
- `RandomForestClassifier_fit`: Trains the classifier instance using virtual parameter `obj_id`.
- `RandomForestClassifier_score`: Evaluates test set accuracy using virtual parameter `obj_id`.

### System Message
```text
You are an expert AI data scientist using tools from an Integrum MCP server.
Key Rules:
- When a tool returns an object, it assigns an object ID (e.g. 'lucid_yak').
- Use 'obj_id' to call methods on that object (e.g. RandomForestClassifier_fit, RandomForestClassifier_score).
- To split data with train_test_split, pass arrays as a list, e.g. arrays=['<id>.data', '<id>.target'] (or arrays=['<id>_0', '<id>_1']), along with test_size (e.g. 0.2) and random_state.
- train_test_split returns 4 arrays: X_train, X_test, y_train, y_test (accessible via '<split_id>.X_train', '<split_id>.X_test', etc.).
- Fit the classifier on the training set (X_train, y_train).
- Evaluate test accuracy using RandomForestClassifier_score (passing obj_id, X='<split_id>.X_test', y='<split_id>.y_test') or accuracy_score.
- Always report the final accuracy score in your final reply.
```

### User Request
```text
Please perform the following end-to-end task using the tools:
1. Load the iris dataset using load_iris.
2. Split the dataset into train and test sets using train_test_split (test_size=0.2, random_state=42).
3. Create a RandomForestClassifier instance.
4. Train / fit the classifier on the training set (X_train and y_train).
5. Evaluate accuracy on the test set (X_test and y_test) and report the final test accuracy.
```

---

## 2. Multi-Turn Execution Transcript

### Turn 1: Load Dataset

**Gemma Tool Call:**
```json
{
  "name": "load_iris",
  "arguments": {
    "return_X_y": true
  }
}
```

**Integrum FastMCP Output:**
```json
{
  "status": "success",
  "object_id": "warm_hawk",
  "type": "builtins.tuple",
  "shape": "len=2",
  "items": [
    "warm_hawk_0",
    "warm_hawk_1"
  ],
  "message": "Stored tuple with object_id 'warm_hawk'. Access items using: ['warm_hawk_0', 'warm_hawk_1']."
}
```

---

### Turn 2: Train/Test Split

**Gemma Tool Call:**
```json
{
  "name": "train_test_split",
  "arguments": {
    "arrays": [
      "warm_hawk_0",
      "warm_hawk_1"
    ],
    "random_state": 42,
    "test_size": 0.2
  }
}
```

**Integrum FastMCP Output:**
```json
{
  "status": "success",
  "object_id": "serene_panda",
  "type": "builtins.list",
  "shape": "len=4",
  "items": [
    "serene_panda_0",
    "serene_panda_1",
    "serene_panda_2",
    "serene_panda_3"
  ],
  "sub_keys": [
    "serene_panda.X_train",
    "serene_panda.X_test",
    "serene_panda.y_train",
    "serene_panda.y_test"
  ],
  "message": "train_test_split returned 4 arrays stored under 'serene_panda':\n  - X_train: 'serene_panda.X_train' (or 'serene_panda_0')\n  - X_test:  'serene_panda.X_test' (or 'serene_panda_1')\n  - y_train: 'serene_panda.y_train' (or 'serene_panda_2')\n  - y_test:  'serene_panda.y_test' (or 'serene_panda_3')"
}
```

---

### Turn 3: Instantiate RandomForestClassifier

**Gemma Tool Call:**
```json
{
  "name": "RandomForestClassifier_create",
  "arguments": {
    "random_state": 42
  }
}
```

**Integrum FastMCP Output:**
```json
{
  "status": "success",
  "object_id": "radiant_newt",
  "class": "RandomForestClassifier",
  "message": "Created RandomForestClassifier instance with object_id 'radiant_newt'. Use obj_id='radiant_newt' in subsequent method calls like RandomForestClassifier_fit."
}
```

---

### Turn 4: Fit Model on Training Data

**Gemma Tool Call:**
```json
{
  "name": "RandomForestClassifier_fit",
  "arguments": {
    "obj_id": "radiant_newt",
    "X": "serene_panda.X_train",
    "y": "serene_panda.y_train"
  }
}
```

**Integrum FastMCP Output:**
```json
{
  "status": "success",
  "object_id": "radiant_newt",
  "message": "Successfully executed on object 'radiant_newt' (returned self)."
}
```

---

### Turn 5: Evaluate Model on Test Data

**Gemma Tool Call:**
```json
{
  "name": "RandomForestClassifier_score",
  "arguments": {
    "obj_id": "radiant_newt",
    "X": "serene_panda.X_test",
    "y": "serene_panda.y_test"
  }
}
```

**Integrum FastMCP Output:**
```text
1.0
```

---

### Turn 6: Assistant Final Response

**Gemma Response:**
```markdown
The end-to-end task has been completed successfully. Here is the summary of the steps performed:

1. **Dataset Loaded**: The Iris dataset was loaded.
2. **Data Split**: The dataset was split into training and testing sets (80% train, 20% test) using `random_state=42`.
3. **Model Created**: A `RandomForestClassifier` instance was initialized.
4. **Model Trained**: The classifier was fitted on the training data (`X_train`, `y_train`).
5. **Model Evaluated**: The model's accuracy was evaluated on the test set (`X_test`, `y_test`).

**Final Test Accuracy: 1.0** (100%)
```

---

## 3. Verification Assertion

The test asserted:
- **Object Store Verification**: Model `radiant_newt` in `ObjectStore` has 100 decision tree estimators and 3 target classes.
- **Independent Test Set Score**: Re-running `score(X_test, y_test)` on the store instance yielded `1.0000` (100% test accuracy).
