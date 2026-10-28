# Anomaly v1 limitations

The model estimates the risk of a future abnormal end-cause at session start. It does not detect an already observed electrical fault. The Jiaxing `is_abnormal` label is derived from the recorded `end_cause`; it is not a separately verified fault label. Only charger identity, location, district, and start-calendar features are used. The source timestamp's timezone is unverified.

On the chronological test set, precision is 0.215 and recall is 0.888 at threshold 0.40. False alarms are frequent (36,711 of 54,858 normal sessions). An alert is an investigation flag requiring human review. Performance has not been validated on Egyptian stations, real-time telemetry, or production fault labels. Distribution shift and miscalibration remain possible.

SMOTE is intentionally not used; class imbalance is handled through training-time class weighting.
