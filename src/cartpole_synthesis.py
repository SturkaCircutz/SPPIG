from __future__ import annotations

import sys

from cartpole.psm import synthesis as _synthesis

sys.modules[__name__] = _synthesis
