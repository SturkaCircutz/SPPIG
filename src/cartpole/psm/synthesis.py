from __future__ import annotations

import sys
import types

from . import synthesis_config as _synthesis_config
from . import synthesis_programs as _synthesis_programs
from . import synthesis_student as _synthesis_student
from . import synthesis_teacher as _synthesis_teacher
from . import synthesis_switch_fit as _synthesis_switch_fit
from . import synthesis_switch_search as _synthesis_switch_search
from .synthesis_config import *  # noqa: F401,F403
from .synthesis_programs import *  # noqa: F401,F403
from .synthesis_student import *  # noqa: F401,F403
from .synthesis_teacher import *  # noqa: F401,F403
from .synthesis_switch_fit import *  # noqa: F401,F403
from .synthesis_switch_search import *  # noqa: F401,F403

_SECTION_MODULES = (
    _synthesis_config,
    _synthesis_programs,
    _synthesis_student,
    _synthesis_teacher,
    _synthesis_switch_fit,
    _synthesis_switch_search,
)

def _is_shared_name(name: str) -> bool:
    return not (
        name in {"annotations", "sys", "types"}
        or (name.startswith("__") and name.endswith("__"))
        or name.startswith("_synthesis_")
        or name in {"_SECTION_MODULES", "_SynthesisFacade", "_is_shared_name", "_refresh_section_globals"}
    )


for _module in _SECTION_MODULES:
    for _name, _value in _module.__dict__.items():
        if _is_shared_name(_name):
            globals().setdefault(_name, _value)


def _refresh_section_globals() -> None:
    shared = {name: value for name, value in globals().items() if _is_shared_name(name)}
    for module in _SECTION_MODULES:
        module.__dict__.update(shared)


_refresh_section_globals()


class _SynthesisFacade(types.ModuleType):
    """Compatibility facade for the split SPPIG implementation.

    A number of tests and legacy scripts patch private names on ``cartpole_synthesis``.
    The actual implementation now lives in focused section modules, so setting an
    attribute here is propagated to every section module to preserve those patch points.
    """

    def __setattr__(self, name: str, value) -> None:  # type: ignore[override]
        super().__setattr__(name, value)
        if _is_shared_name(name):
            for module in _SECTION_MODULES:
                module.__dict__[name] = value


sys.modules[__name__].__class__ = _SynthesisFacade
