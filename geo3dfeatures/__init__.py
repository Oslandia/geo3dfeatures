"""geo3dfeatures package
"""

import logging

import daiquiri

__version__ = "0.4.0"


daiquiri.setup(
    level=logging.INFO,
    outputs=(
        daiquiri.output.Stream(
            formatter=daiquiri.formatter.ColorFormatter(
                fmt=(
                    "%(color)s[%(asctime)s] %(module)s.%(funcName)s "
                    "(%(levelname)s) -%(color_stop)s %(message)s"
                ),
                datefmt="%H:%M:%S"
            )
        ),
    ),
)
logger = daiquiri.getLogger("root")
