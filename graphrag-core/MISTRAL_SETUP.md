Mistral testing and secure API key handling
=========================================

Local testing
-------------

- Do NOT commit API keys to the repository.
- For local testing, export the key as an environment variable before running tests:

```bash
export MISTRAL_EMBEDDINGS_API_KEY="your_key_here"
cargo test -p graphrag-core
```

CI (GitHub Actions)
--------------------

- Add a repository secret named `MISTRAL_API_KEY` in your GitHub repo settings.
- The CI workflow will map `MISTRAL_API_KEY` to `MISTRAL_EMBEDDINGS_API_KEY` during the test job.

How the code keeps keys out of commits
-------------------------------------

- `EmbeddingConfig.api_key` and `LlmConfig.api_key` are annotated with `#[serde(skip_serializing)]`. Keys are not written when serializing configs to disk.
- The registry will prefer an explicit config API key but falls back to environment variables `MISTRAL_EMBEDDINGS_API_KEY` or `MISTRAL_API_KEY` if none is set in config.

If you want CI to run extra feature-gated tests (e.g., `ureq`), enable the features in the CI job or in `Cargo.toml`.
