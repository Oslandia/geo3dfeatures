
import seaborn as sns


PALETTE = sns.color_palette("colorblind")
GLOSSARY = {
    "vegetation": {"id": 0, "color": PALETTE[2]},  # Vegetation: green
    "falaise": {"id": 1, "color": PALETTE[7]},  # Vegetation: grey
    "eboulis": {"id": 2, "color": PALETTE[5]},  # Vegetation: marron
    "route": {"id": 3, "color": PALETTE[0]},  # Vegetation: blue
    "structure": {"id": 4, "color": PALETTE[1]},  # Vegetation: orange
    }
