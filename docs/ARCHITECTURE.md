# Running architecture

```mermaid
flowchart LR
  J[Jiaxing transactions] --> DH[Hourly demand history]
  J --> SR[EV session risk pipeline]
  J --> AV[Availability duration pipeline]
  DH --> DM[Demand pipeline]
  DH --> GL[Grid-load interval screen]
  P[PVGIS reference PV] --> SO[Solar pipeline]
  W[Open-Meteo prior runs] --> SO
  DM --> DB[Portable demand bundle]
  SR --> SB[Portable anomaly-v1 bundle]
  SO --> PB[Portable solar bundles]
  GL --> GB[Rule config + history checksum]
  AV --> AB[Portable duration bundle + training risk table]
  DB --> A[Integration adapters]
  SB --> A
  PB --> A
  GB --> A
  AB --> A
  A --> API[FastAPI + Pydantic]
  API --> UI[Streamlit]
  API --> DT[Digital Twin]
  API --> CS[Charging Station Simulator]
  DM -. tracked runs .-> ML[MLflow]
  SR -. tracked runs .-> ML
  SO -. tracked runs .-> ML
  RK[Deterministic charger ranker] --> A
```

Editable Mermaid source: `docs/architecture.mmd`. Demand, 24h/48h solar, EV session risk and availability use portable bundles. Grid-load anomaly uses a checksum-checked rule configuration and the committed historical demand proxy. The ranker uses caller-supplied historical charger state and a training-period abnormal-end rate; missing demand/solar terms are disclosed neutral constants. All components use FastAPI/Streamlit historical contracts. Digital Twin and Charging Station Simulator are deterministic simulation tools, not live grid controls. MLflow records training experiments, while runtime serving has no MLflow dependency.
