import fastf1 as ff1
from fastf1 import plotting
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from utils_func import format_laptime
from matplotlib.collections import LineCollection
from DarkF1_Plotting_Theme import apply_f1_dark_theme
from Page_2 import load_page_2
from Page_3 import load_page_3
from Page_4 import load_page_4

apply_f1_dark_theme()

# Load Session from FF1
GP = "Monza"
Session = "R"
Driver = "LEC"
Year = 2025

session = ff1.get_session(Year, GP, Session)
session.load(weather=False)

df_laps = session.laps
df_laps = df_laps.pick_drivers(Driver)
df_laps["LapTimeS"] = df_laps["LapTime"].dt.total_seconds()

# Create on object gridpsec() to insert graphs. 
fig = plt.figure()

fig.suptitle(f"{Year} {GP} GP – Session: {Session}", fontsize=16, fontweight="bold")
fig.text(0.5, 0.93, f"Driver: {Driver}", ha="center", fontsize=11)

gs = gridspec.GridSpec(10,14, figure = fig)

ax1 = fig.add_subplot(gs[0:4,0:5])
ax1.grid(True)
ax1.set_xlabel("Lap Number")
ax1.set_ylabel("Lap Time [s]")
ax1.set_title(f"Lap Times {Driver}")
ax1.plot(df_laps["LapNumber"], df_laps["LapTimeS"])
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_laptime))
ax1.yaxis.set_major_locator(MaxNLocator(nbins = 15))

ax2 = fig.add_subplot(gs[6:,0:5])
ax2.grid(True)
ax2.set_xlabel("Tyre Life [Laps]")
ax2.set_ylabel("Lap Time [s]")
ax2.set_title(f"Tyres Stint Times {Driver}")

Compounds_Colors = {
    "['SOFT']": "red",
    "['MEDIUM']": "orange",
    "['HARD']": "grey"
}

Stints = df_laps["Stint"].unique()
counter = 0

for stint in Stints:
    counter += 1

    x = df_laps.loc[df_laps["Stint"] == stint]["TyreLife"]
    y = df_laps.loc[df_laps["Stint"] == stint]["LapTimeS"]
    compound = str(df_laps.loc[df_laps["Stint"] == stint]["Compound"].unique())

    ax2.scatter(x,y,s = 20, alpha = 0.5, c = Compounds_Colors[compound],
                label =  f"Stint:{stint} {compound}")

    # Remove from y the outliers
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    x = x[(y> Q1 - 1.5*IQR) & (y < Q3 + 1.5*IQR)]
    y = y[(y> Q1 - 1.5*IQR) & (y < Q3 + 1.5*IQR)]

    # Perform Linear Interpolation of x and y
    slope, intercept = np.polyfit(x,y,1)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope *x_fit + intercept

    ax2.plot(x_fit,y_fit,c = Compounds_Colors[compound])
    ax2.text(
        int(0.75*x_fit[-1]), int(y_fit[-1]) + counter*7,
        f"{slope:.3f} sec/lap",
        bbox = dict(facecolor = "black", alpha = 0.8, edgecolor = Compounds_Colors[compound]),
        fontsize = 9,
        clip_on = True,
        zorder = 20
    )
    

ax2.legend()
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_laptime))
ax2.yaxis.set_major_locator(MaxNLocator(nbins = 15))

# For now we select a lap manually, later one could allow to select the lap from a dropdown menu or a slider
n_lap = 10
laps = df_laps["LapNumber"].unique()
lap = laps[n_lap]

Lap = df_laps.loc[df_laps["LapNumber"] == lap]
telemetry = Lap.get_car_data().add_distance()

Tel_Data = ["Speed", "RPM", "Throttle"]

ax3 = fig.add_subplot(gs[0:2,7:])
ax3.grid(True)
ax3.set_ylabel("Velocity [km/h]")
ax3.plot(telemetry["Distance"] /1000, telemetry[Tel_Data[0]], label = Driver)
ax3.set_title(f"Telemetry: {Tel_Data[0]} - {Tel_Data[1]} - {Tel_Data[2]}")

ax3 = fig.add_subplot(gs[2:4,7:])
ax3.grid(True)
ax3.set_ylabel("RPM")
ax3.plot(telemetry["Distance"] /1000, telemetry[Tel_Data[1]], label = Driver, color = "red")

ax3 = fig.add_subplot(gs[4:6,7:])
ax3.grid(True)
ax3.set_xlabel(" Distance [km]")
ax3.set_ylabel("Throttle %")
ax3.plot(telemetry["Distance"] /1000, telemetry[Tel_Data[2]], label = Driver, color = "orange")

# Don't change the variable names x_map and y_map to maintain the code working. 
# If interested in rotating the map 90 degrees, simply switch the "X" and "Y" when selecting from telemetry[""].
x_map = Lap.telemetry["Y"]
y_map = Lap.telemetry["X"]
speed = Lap.telemetry["Speed"]

points = np.array([x_map,y_map]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1], points[1:]], axis = 1)
ax4 = fig.add_subplot(gs[7:,8:12])
ax4.axis("off")
ax4.plot(x_map, y_map, color = "black", linestyle = '-', linewidth = 7, zorder = 0)
norm = mcolors.Normalize(vmin = np.min(speed), vmax = np.max(speed))
lc = LineCollection(segments, cmap = "jet", norm = norm)
lc.set_array(speed)
lc.set_linewidth(5)
line = ax4.add_collection(lc)
ax4.set_aspect('equal', 'box')

cbar = fig.colorbar(
    lc,
    ax=ax4,
    orientation='vertical',
    fraction=0.046,   # width relative to the width of the figure
    pad=0.02          # distanzce from the map
)

cbar.set_label("Speed [km/h]", fontsize=10)
cbar.ax.tick_params(labelsize=9)
plt.show()

load_page_2(session, channels = ["Speed", "Throttle", "Brake","nGear","DRS","Time"])
load_page_3(GP, Year, Driver)
load_page_4(session, drivers = ["VER","HAM", "ALO", "LEC", "RUS", "NOR", "PIA"], n_clusters = 3)
plt.show()



