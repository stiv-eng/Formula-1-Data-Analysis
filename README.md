# Formula-1-Data-Analysis
Formula 1 telemetry and lap-time analysis using FastF1 and machine learning models for lap performance evaluation and driver comparison.

The code is organised into a source code (F1_Data_Analysis), which calls different functions to load each portion of the Dashboard. 

The source code also plots the first dashboard, with: 
- Lap time analysis over a race session
- Tyre stint performance and degradation trends
- Telemetry analysis (speed, RPM, throttle) over distance for a lap
- Track map visualization with speed-based color mapping
<img width="480" height="260" alt="image" src="https://github.com/user-attachments/assets/f70d7e10-87c8-4e6f-aefd-273913cf4101" />

The code Page_2 features a telemetry comparison display, allowing two laps to be compared directly, either from the same driver or from different drivers.
Using telemetry channels such as speed, throttle, brake signal, and gear selection, the dashboard highlights where lap time is gained or lost along the lap.

<img width="480" height="252" alt="image" src="https://github.com/user-attachments/assets/e7f6dfbe-2562-4b2b-bb1c-96b32771a1c5" />

Page_3 instead introduces a Support Vector Machine (SVM) regression model trained to predict lap time based on lap features such as session type (FP / Qualifying / Race), tyre compound and selected telemetry metrics.
For now the model follows a single-driver, single-track approach, allowing it to learn how a driver performs over previous laps and attempt to predict future lap times under similar conditions.
This is an early step toward building data-driven performance models from motorsport telemetry.

<img width="480" height="249" alt="image" src="https://github.com/user-attachments/assets/e5f4e273-ab17-4823-9ec6-73ea6d2f36f7" />

