# Running architecture

```mermaid
flowchart LR
  J[Jiaxing transactions] --> DH[Hourly demand history]
  J --> SR[EV session risk pipeline]
  DH --> DM[Demand pipeline]
  DH --> GL[Grid-load interval screen]
  P[PVGIS reference PV] --> SO[Solar pipeline]
  W[Open-Meteo prior runs] --> SO
  DM --> DB[Portable demand bundle]
  SR --> SB[Portable anomaly-v1 bundle]
  SO --> PB[Portable solar bundles]
  GL --> GB[Rule config + history checksum]
  DB --> A[Integration adapters]
  SB --> A
  PB --> A
  GB --> A
  A --> API[FastAPI + Pydantic]
  API --> UI[Streamlit]
  API --> DT[Digital Twin]
  API --> CS[Charging Station Simulator]
  DM -. tracked runs .-> ML[MLflow]
  SR -. tracked runs .-> ML
  SO -. tracked runs .-> ML
  AV[Availability: not integrated] -. safe 503 .-> API
  RE[Recommendation: not integrated] -. safe 503 .-> API
```

Editable Mermaid source: `docs/architecture.mmd`. Demand, 24h/48h solar, and EV session risk use portable bundles. Grid-load anomaly uses a checksum-checked rule configuration and the committed historical demand proxy. All four are integrated with FastAPI/Streamlit; the Grid-load endpoint does not alter the session-risk endpoint. Digital Twin and Charging Station Simulator are deterministic simulation tools, not live grid controls. MLflow records training experiments, while runtime serving has no MLflow dependency. Availability and recommendation adapters safely return 503 until model handoff.
