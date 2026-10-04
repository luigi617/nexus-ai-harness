from __future__ import annotations

import logging

# Libraries never configure logging; stay silent until the application adds handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())
