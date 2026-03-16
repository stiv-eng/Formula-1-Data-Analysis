def load_page_3(
        event_name,
        year,
        driver,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import fastf1

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    session_codes = ["Q", "R"]

    all_rows = []

    # ---------------------------------------
    # 2) Loop through the whole weekend
    # ---------------------------------------
    for sess_code in session_codes:
        try:
            sess = fastf1.get_session(year, event_name, sess_code)
            sess.load()

            laps = sess.laps.pick_drivers(driver).copy()

            # Keep only laps with valid lap times
            laps = laps[laps["LapTime"].notna()].copy()

            # Remove in-laps and out-laps if possible
            if "PitInTime" in laps.columns:
                laps = laps[laps["PitInTime"].isna()]
            if "PitOutTime" in laps.columns:
                laps = laps[laps["PitOutTime"].isna()]

            # Keep only accurate laps if available
            if "IsAccurate" in laps.columns:
                laps = laps[laps["IsAccurate"] == True]

            for _, lap in laps.iterrows():
                try:
                    tel = lap.get_car_data()

                    if tel is None or len(tel) == 0:
                        continue

                    lap_time_s = lap["LapTime"].total_seconds()

                    avg_speed = tel["Speed"].mean() if "Speed" in tel.columns else np.nan
                    throttle_mean = tel["Throttle"].mean() if "Throttle" in tel.columns else np.nan

                    brake_time = np.nan
                    if "Brake" in tel.columns and "Time" in tel.columns:
                        brake_mask = tel["Brake"].astype(bool)
                        time_s = tel["Time"].dt.total_seconds().to_numpy()

                        if len(time_s) > 1:
                            dt = np.diff(time_s)
                            mean_dt = np.mean(dt)
                            brake_time = brake_mask.sum() * mean_dt
                        else:
                            brake_time = 0.0

                    compound = lap["Compound"] if "Compound" in lap.index else "UNKNOWN"
                    lap_number = lap["LapNumber"] if "LapNumber" in lap.index else np.nan

                    all_rows.append({
                        "SessionType": sess_code,
                        "Compound": compound,
                        "LapNumber": lap_number,
                        "AvgSpeed": avg_speed,
                        "ThrottleMean": throttle_mean,
                        "BrakeTime": brake_time,
                        "LapTime": lap_time_s
                    })

                except Exception:
                    continue

        except Exception as e:
            print(f"Skipping {sess_code}: {e}")
            continue

    # ---------------------------------------
    # 3) Build dataframe
    # ---------------------------------------
    df = pd.DataFrame(all_rows)

    if df.empty:
        print(f"No usable laps found for driver {driver} in {event_name}.")
        return None

    df = df[df["LapTime"].notna()].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    print("\nDataset preview:")
    print(df)
    print(f"\nNumber of laps used: {len(df)}")
    print("\nLaps per session:")
    print(df["SessionType"].value_counts())

    # ---------------------------------------
    # 4) Define X and y
    # ---------------------------------------
    X = df[["Compound", "SessionType", "LapNumber", "AvgSpeed", "ThrottleMean", "BrakeTime"]]
    y = df["LapTime"]

    categorical_features = ["Compound", "SessionType"]
    numerical_features = ["LapNumber", "AvgSpeed", "ThrottleMean", "BrakeTime"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numerical_features
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # ---------------------------------------
    # 5) Train/test split
    # ---------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    # ---------------------------------------
    # 6) Train SVR
    # ---------------------------------------
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------------------------------
    # 7) Metrics
    # ---------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nSVR results")
    print(f"MAE  = {mae:.4f} s")
    print(f"MSE  = {mse:.4f} s²")
    print(f"RMSE = {rmse:.4f} s")
    print(f"R²   = {r2:.4f}")

    # ---------------------------------------
    # 8) Plot
    # ---------------------------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Lap Time [s]")
    plt.ylabel("Predicted Lap Time [s]")
    plt.title(f"{driver} - {event_name} (Full weekend)\nLap Time Prediction with SVR")
    plt.grid(True)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    metrics_text = (
        f"R²   = {r2:.3f}\n"
        f"MSE  = {mse:.3f} s²\n"
        f"RMSE = {rmse:.3f} s\n"
        f"MAE  = {mae:.3f} s"
    )

    plt.text(
        0.05, 0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="grey", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()

    return df, model
