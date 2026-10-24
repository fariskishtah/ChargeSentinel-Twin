# End-to-End Tests

Docker tests are opt-in and require the Compose stack to be running:

```bash
RUN_DOCKER_TESTS=1 pytest -m docker
```

The tests check service health, internal UI-to-API networking, deterministic Digital Twin behavior, and the expected 503 response for unavailable model endpoints. They do not fabricate predictions.
