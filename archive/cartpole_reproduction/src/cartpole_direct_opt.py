from __future__ import annotations

import sys

from cartpole.direct_opt import core as _core

sys.modules[__name__] = _core
