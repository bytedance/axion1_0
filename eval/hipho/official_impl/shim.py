"""Import VLMEvalKit's official HiPhO grader outside VLMEvalKit.

The three files next to this one are copied unmodified from
open-compass/VLMEvalKit; see README.md. They import `vlmeval.smp`,
`vlmeval.utils` and several relative modules, which this shim provides so they
load as-is.

Only the per-problem grading path is used, `_evaluate_single_problem` and what
it calls. Symbols reached only by the full `evaluate()` path are stubs that
raise.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))

PKG = "hipho_official_pkg"


def _boom(name: str):
    def f(*a, **k):
        raise RuntimeError(
            f"the official hipho.py called {name}(), which this shim does not "
            f"implement -- that means execution reached the evaluate()/xlsx path "
            f"rather than the per-problem grading path"
        )
    return f


def _load(mod_name: str, filename: str):
    """Load `filename` from this directory under the given module name."""
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def install():
    """Install the faked vlmeval modules and return the official hipho module."""
    if "vlmeval" not in sys.modules:
        vlmeval = types.ModuleType("vlmeval")
        smp = types.ModuleType("vlmeval.smp")
        for n in ("dump", "get_intermediate_file_path", "load", "toliststr"):
            setattr(smp, n, _boom(n))
        utils = types.ModuleType("vlmeval.utils")
        utils.track_progress_rich = _boom("track_progress_rich")
        vlmeval.smp = smp
        vlmeval.utils = utils
        sys.modules.update({"vlmeval": vlmeval, "vlmeval.smp": smp, "vlmeval.utils": utils})

    # hipho.py uses relative imports, so it loads as a submodule of a package.
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [HERE]
        sys.modules[PKG] = pkg

        image_base = types.ModuleType(f"{PKG}.image_base")

        class ImageBaseDataset:          # base class placeholder
            MODALITY = "IMAGE"

            def __init__(self, *a, **k):
                raise RuntimeError("do not call HiPhODataset.__init__; use new_scorer()")

        image_base.ImageBaseDataset = ImageBaseDataset
        sys.modules[f"{PKG}.image_base"] = image_base

        sub_utils = types.ModuleType(f"{PKG}.utils")
        sub_utils.__path__ = [HERE]
        sub_utils.build_judge = _boom("build_judge")
        sys.modules[f"{PKG}.utils"] = sub_utils

        _load(f"{PKG}.utils.hipho_prompt_inference", "hipho_prompt_inference.py")
        _load(f"{PKG}.utils.hipho_verifier", "hipho_verifier.py")

    return _load(f"{PKG}.hipho", "hipho.py")


def new_scorer():
    """A HiPhODataset instance usable for grading only.

    Built with `object.__new__` to skip `__init__`, which downloads the dataset
    and builds a dataframe. The grading path reads only class-level constants
    and templates from `self`.
    """
    mod = install()
    return object.__new__(mod.HiPhODataset), mod
