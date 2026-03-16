# Page_2.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _get_lap_telemetry(session, driver: str, lap_number: int):
    """
    Returns telemetry dataframe with Distance column.
    Raises ValueError if lap not found.
    """
    laps = session.laps.pick_drivers(driver)

    # Validate lap exists
    laps_available = laps["LapNumber"].dropna().astype(int).unique()
    if int(lap_number) not in laps_available:
        laps_available = np.sort(laps_available)
        raise ValueError(
            f"Lap {lap_number} not found for {driver}. "
            f"Available laps: {laps_available.tolist()}"
        )

    lap = laps.loc[laps["LapNumber"] == int(lap_number)].iloc[0]
    tel = lap.get_car_data().add_distance()
    return tel


def load_page_2(
    session,
    driver_A="LEC",
    lap_A=20,
    driver_B="VER",
    lap_B=20,
    channels=None,
):
    """
    Page 2: Driver-vs-driver telemetry comparison.
    Call after session.load().

    channels: list of telemetry channels to plot in order.
              Supported by default: Speed, Throttle, Brake, RPM, Gear, DRS
    """

    # Choose default channels if not provided
    if channels is None:
        channels = ["Speed", "Throttle", "Brake", "RPM"]

    # Load telemetry for the two laps
    telA = _get_lap_telemetry(session, driver_A, lap_A)
    telB = _get_lap_telemetry(session, driver_B, lap_B)

    # X axis (distance in km)
    xA = telA["Distance"].to_numpy() / 1000.0
    xB = telB["Distance"].to_numpy() / 1000.0

    # Figure layout
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Telemetry Comparison", fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.93,
        f"A: {driver_A} Lap {lap_A}     vs     B: {driver_B} Lap {lap_B}",
        ha="center",
        fontsize=11
    )

    # Grid: one row per channel
    n = len(channels)
    gs = gridspec.GridSpec(
        n, 1, figure=fig,
        left=0.07, right=0.98, top=0.90, bottom=0.08,
        hspace=0.25
    )

    axes = []
    for i, ch in enumerate(channels):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        ax.grid(True)
        axes.append(ax)

        # Pick y-data robustly
        if ch in telA:
            yA = telA[ch].to_numpy()
        else:
            yA = np.full_like(xA, np.nan, dtype=float)

        if ch in telB:
            yB = telB[ch].to_numpy()
        else:
            yB = np.full_like(xB, np.nan, dtype=float)

        # Special handling for Brake if it's boolean (common in FastF1)
        if ch.lower() == "brake":
            # If dtype is bool, show as 0/100%
            if "Brake" in telA and telA["Brake"].dtype == bool:
                yA = telA["Brake"].astype(float).to_numpy() * 100.0
            if "Brake" in telB and telB["Brake"].dtype == bool:
                yB = telB["Brake"].astype(float).to_numpy() * 100.0

        if ch.lower() == "time":

            time_A = telA["Time"].dt.total_seconds()
            time_B = telB["Time"].dt.total_seconds()

            # Mask to remove infinte values
            mask = np.isfinite(xA) & np.isfinite(time_A)
            xA_mask, time_A = xA[mask], time_A[mask]

            time_B = np.interp(xA_mask, xB, time_B)

            yA = time_A - time_B

            ax.plot(xA, yA, label = f"Delta Time {driver_A} - {driver_B}")
            ax.set_ylabel(f"Delta Time [s] \n{driver_A} - {driver_B}")
            continue

        # Plot
        ax.plot(xA, yA, label=f"A: {driver_A}", linewidth=1.6)
        ax.plot(xB, yB, label=f"B: {driver_B}", linewidth=1.6)

        # Labels
        if ch.lower() == "speed":
            ax.set_ylabel("Speed [km/h]")
        elif ch.lower() == "throttle":
            ax.set_ylabel("Throttle [%]")
        elif ch.lower() == "brake":
            ax.set_ylabel("Brake [%]")
        elif ch.lower() == "rpm":
            ax.set_ylabel("RPM")
        elif ch.lower() == "gear":
            ax.set_ylabel("Gear")
        elif ch.lower() == "drs":
            ax.set_ylabel("DRS")
        else:
            ax.set_ylabel(ch)

        # Legend only on top axis (clean)
        if i == 0:
            ax.legend(loc="upper right")

        # Autoscale (ignore NaNs)
        ax.relim()
        ax.autoscale_view()

    axes[-1].set_xlabel("Distance [km]")


    axes[-1].set_xlabel("Distance [km]")

    # --- Shared vertical cursor across all subplots ---
    vlines = []
    for ax in axes:
        vl = ax.axvline(np.nan, linewidth=1.2, alpha=0.8, color = "grey")  # hidden initially
        vlines.append(vl)

    def _on_move(event):
        # Only react if we're inside one of the telemetry axes
        if event.inaxes not in axes:
            for vl in vlines:
                vl.set_xdata([np.nan, np.nan])
            fig.canvas.draw_idle()
            return

        x = event.xdata
        if x is None or not np.isfinite(x):
            return

        # Update all vertical lines to the same x position
        for vl in vlines:
            vl.set_xdata([x, x])

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    plt.show()
    return fig

    plt.show()
    return fig

