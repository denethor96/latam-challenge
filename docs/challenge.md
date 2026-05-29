# Challenge Documentation: Flight Delay Prediction

## Part I: Model Operationalization

### Runtime and Dependency Modernization

The original challenge dependencies were pinned to older versions of the Python data and API stack. To make the project compatible with a modern and stable runtime, the environment was updated to Python 3.12-compatible dependencies.

This modernization improves:

- runtime support;
- installation reliability;
- deployment compatibility;
- maintainability;
- compatibility with modern FastAPI and Pydantic versions.

The dependency files were kept separated by purpose:

- `requirements.txt` for runtime dependencies;
- `requirements-test.txt` for test and stress-test dependencies;
- `requirements-dev.txt` for local exploratory analysis dependencies.

CI installs only runtime and test dependencies. Development dependencies remain available for local notebook exploration.

### Model Selection

The notebook evaluates multiple model alternatives, including XGBoost and Logistic Regression. For the deployed solution, I selected the Logistic Regression model trained with the top 10 features and manual class weighting.

Reasons:

- The notebook concludes that there is no significant performance difference between XGBoost and Logistic Regression.
- The top 10 feature subset keeps the serving pipeline simple while preserving the relevant predictive signal.
- Manual class weighting improves recall for the delayed class, which is important because delay is the minority class.
- Logistic Regression is lighter to deploy and operate, keeping the runtime smaller and avoiding an additional production dependency for XGBoost.

The class weighting strategy follows the notebook logic:

```python
class_weight = {
    0: n_y1 / n_total,
    1: n_y0 / n_total,
}
```

This gives a higher weight to the minority delayed class.

### Feature Engineering

The notebook explores the following engineered variables:

- `high_season`
- `period_day`
- `min_diff`
- `delay`

The final selected model consumes only the top 10 one-hot encoded features derived from:

- `OPERA`
- `MES`
- `TIPOVUELO`

Because of this, the production serving payload only needs those three raw input fields. The model preprocessing layer internally converts them into the top 10 features expected by the trained model.

For example, a request with:

```json
{
  "OPERA": "Grupo LATAM",
  "TIPOVUELO": "I",
  "MES": 12
}
```

can activate one-hot encoded features such as:

- `OPERA_Grupo LATAM`
- `TIPOVUELO_I`
- `MES_12`

### Training-Serving Skew Prevention

The production pipeline intentionally excludes `high_season` and `period_day` because these variables are not part of the selected feature set used by the deployed model.

The `min_diff` transformation is only used during training to reconstruct the target variable `delay`, defined as:

```text
delay = 1 if operation_time - scheduled_time > 15 minutes, else 0
```

Keeping only the transformations required by the selected model avoids unnecessary API inputs and reduces the risk of training-serving skew.

### Model Persistence

The model supports persistence with `joblib`.

During serving startup:

1. If a persisted model artifact exists, it is loaded.
2. If no artifact exists, the model is trained once from the bundled dataset.
3. The trained model is then saved for reuse.

Prediction requests do not retrain the model.

This keeps request latency stable and avoids repeated training work.

## Part II: API Development

The API is implemented with FastAPI.

### API Contract

#### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "OK"
}
```

#### Prediction

```http
POST /predict
```

Request:

```json
{
  "flights": [
    {
      "OPERA": "Aerolineas Argentinas",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}
```

Response:

```json
{
  "predict": [0]
}
```

The endpoint supports non-empty batch prediction through the `flights` list.

### API Validation

The API uses Pydantic schemas to validate request payloads.

Validation rules:

- `flights` must contain at least one flight.
- `TIPOVUELO` must be either `I` or `N`.
- `MES` must be a valid month from 1 to 12.
- `OPERA` must exist in the airline categories from the training dataset.

`TIPOVUELO` and `MES` are represented as enums. Airline validation is data-driven instead of hardcoded, keeping the API contract aligned with the training data.

FastAPI normally returns HTTP 422 for schema validation errors. The challenge tests expect HTTP 400, so request validation errors are normalized to HTTP 400.

### Model Startup

The model is initialized through FastAPI's lifespan handler.

This avoids deprecated startup hooks and ensures the model is prepared once when the application starts, not during every request.

## Part III: Deployment in GCP

### Docker

The API is containerized with a Python 3.12 slim image.

The Dockerfile:

- installs runtime dependencies;
- copies only the application package and data needed for serving;
- exposes port `8080`;
- uses Cloud Run's `PORT` environment variable when available.

The container command is:

```bash
uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}
```

A `.gcloudignore` file avoids sending local virtual environments, caches, tests, reports, and other unnecessary files to Cloud Build.

### Cloud Run Deployment

The API is deployed to Google Cloud Run.

Cloud Run was selected because it provides:

- managed HTTPS serving;
- container-native deployment;
- autoscaling;
- low operational overhead;
- simple integration with Cloud Build and Artifact Registry.

The deployed API URL is:

```text
https://flight-delay-api-834425492744.us-central1.run.app/
```

Cloud Run is configured with:

- `512Mi` memory;
- `1` CPU;
- `1` minimum instance;
- `3` maximum instances.

One minimum instance is used to reduce cold-start latency during review and stress testing. This can be scaled back to zero after the challenge review to minimize costs.

## Part IV: CI/CD Pipeline

GitHub Actions is used for CI/CD.

### Continuous Integration

CI runs on pull requests and pushes to:

- `develop`
- `main`

The CI workflow:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs runtime and test dependencies.
4. Runs `make model-test`.
5. Runs `make api-test`.

CI runs on both `develop` and `main` to validate integration and release branches.

### Continuous Delivery

CD runs only on pushes to `main`.

The CD workflow:

1. Authenticates to Google Cloud using GitHub repository secrets.
2. Sets up the Google Cloud SDK.
3. Deploys the service to Cloud Run from source.

The CD workflow applies the same Cloud Run resource and scaling configuration used during manual deployment.

CD is restricted to `main` so only official releases trigger production deployment.

## Makefile Adjustments

The provided model tests load the dataset using a relative path:

```python
../data/data.csv
```

To keep the tests unchanged and still support running the official `make model-test` target from the repository root, the Makefile runs model tests from the `challenge/` directory.

The provided Makefile referenced a `.coveragerc` file that was not included in the scaffold. That reference was removed while preserving coverage reports in `reports/`.

The `reports` directory is created with `mkdir -p` to make the targets idempotent in Linux and CI environments.

## Validation Results

### Model Tests

Command:

```bash
make model-test
```

Result:

```text
4 passed
```

### API Tests

Command:

```bash
make api-test
```

Result:

```text
4 passed
```

### Stress Test

Command:

```bash
make stress-test
```

Result:

```text
6029 requests
0 failures
~90.88 requests/second
average response time ~274 ms
median response time ~210 ms
```

The stress test was executed against the deployed Cloud Run API.


## Assumptions and Trade-offs

- **Logistic Regression instead of XGBoost:** Logistic Regression was selected because the notebook showed comparable performance, while the model is simpler to deploy and keeps the runtime lighter.

- **Reduced serving inputs:** The API only requires `OPERA`, `TIPOVUELO`, and `MES` because the deployed model uses one-hot encoded features derived from those fields. Additional notebook variables such as `high_season` and `period_day` are not requested because they are not part of the selected feature set.

- **Target reconstruction during training:** `min_diff` is computed only when the `delay` target needs to be reconstructed from `Fecha-I` and `Fecha-O`. It is not required for serving predictions.

- **Startup fallback training:** The API loads a persisted model artifact when available. If the artifact is missing, it trains once from the bundled dataset during startup. This keeps the challenge reproducible without training on every request.

- **Cloud Run minimum instance:** Cloud Run is configured with `min-instances=1` to reduce cold-start latency during review and stress testing. This is a temporary review-oriented setting and can be changed to zero after evaluation to reduce cost.


## Development Workflow

The implementation followed a GitFlow-style workflow:

- `main` contains the official release.
- `develop` is the integration branch.
- Feature branches were used for each challenge part:
  - project setup;
  - dependency modernization;
  - model implementation;
  - API implementation;
  - deployment;
  - CI/CD;
  - documentation.

Development branches were kept as requested by the challenge instructions.

Commit messages followed a Conventional Commits style where applicable, using prefixes such as `feat:`, `fix:`, `build:`, `ci:`, and `docs:`.

Project hygiene was also improved with `.gitignore` and `.gcloudignore` files to keep local environments, caches, generated reports, and cloud build context artifacts out of version control and deployment context.