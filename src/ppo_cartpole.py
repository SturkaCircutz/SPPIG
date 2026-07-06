from __future__ import annotations

import sys

from cartpole.ppo import core as _core

sys.modules[__name__] = _core
