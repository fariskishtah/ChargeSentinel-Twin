# Model Handoff Contract

Each model owner must give Faris all ten items:

1. Model URI or artifact path.
2. Preprocessing artifact or code.
3. Input schema.
4. Output schema.
5. Inference function.
6. Model version.
7. Evaluation metrics.
8. Known limitations.
9. At least one valid test fixture.
10. No notebook dependency.

The handoff must load in a clean Python process and must not depend on undocumented global state.
