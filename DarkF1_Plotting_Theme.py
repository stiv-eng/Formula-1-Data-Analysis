import matplotlib.pyplot as plt

F1_DARK_THEME = {
    "figure.facecolor": "#000000",
    "axes.facecolor": "#0d0d0d",

    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",

    "axes.edgecolor": "#888888",

    "axes.grid": True,
    "grid.color": "#444444",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,

    "axes.prop_cycle": plt.cycler(color=[
        "#00FFFF",
        "#FF4444",
        "#FFD700",
        "#1E90FF",
        "#32CD32",
        "#FF00FF"
    ]),

    "font.size": 11,

    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_f1_dark_theme():
    """Apply the dark F1 theme to Matplotlib."""
    plt.rcParams.update(F1_DARK_THEME)
