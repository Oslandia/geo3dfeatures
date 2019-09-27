
import seaborn as sns


PALETTE = sns.color_palette("colorblind")
GLOSSARY = {
    "vegetation": {"id": 0, "color": PALETTE[2]},  # Vegetation: green
    "falaise": {"id": 1, "color": PALETTE[7]},  # Cliff: grey
    "eboulis": {"id": 2, "color": PALETTE[5]},  # Scree: marron
    "route": {"id": 3, "color": PALETTE[0]},  # Road: blue
    "beton": {"id": 4, "color": PALETTE[1]},  # Concrete: orange
    "sol": {"id": 5, "color": PALETTE[8]},  # Ground: yellow
    }
