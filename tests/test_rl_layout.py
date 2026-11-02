"""CPU checks for the RL package's module, config, and launcher boundaries.

Training imports are intentionally not executed: the separate RL environment
owns torch, Ray, Megatron and service clients. Hydra still reads real configs.
"""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "axion/training/rl"
VERL = ROOT / "third_party/verl/verl"


def _config_packages():
    """Give Hydra real resource paths without executing GPU package imports."""
    packages = {}
    for name, directory in (
        ("verl", VERL),
        ("verl.trainer", VERL / "trainer"),
        ("verl.trainer.config", VERL / "trainer/config"),
    ):
        spec = importlib.util.spec_from_file_location(
            name, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        packages[name] = importlib.util.module_from_spec(spec)
    return packages


@pytest.mark.parametrize("directory,name", [
    (RL / "config", "paper"),
    (VERL / "experimental/fully_async_policy/config", "fully_async_ppo_megatron_trainer"),
])
def test_trainer_configs_compose_outside_source_directory(directory, name, tmp_path, monkeypatch):
    # A cwd-relative verl search path loses base defaults after relocation.
    monkeypatch.chdir(tmp_path)
    with mock.patch.dict(sys.modules, _config_packages()):
        with initialize_config_dir(version_base=None, config_dir=str(directory)):
            config = compose(config_name=name, overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])
    assert "actor" in config.actor_rollout_ref
    assert "optim" in config.actor_rollout_ref.actor
    assert "adv_estimator" in config.algorithm


def _assert_class_exists(target):
    module, class_name = target.rsplit(".", 1)
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    assert path.is_file(), target
    definitions = ast.parse(path.read_text()).body
    assert any(isinstance(node, ast.ClassDef) and node.name == class_name for node in definitions), target


def test_configured_agent_tools_managers_and_node_files_are_shipped(monkeypatch):
    # Dynamic loader targets and node defaults must resolve without a recipe alias.
    for variable in ("SEARCH_NODES_CONF", "SCHOLAR_NODES_CONF", "PYTHON_NODES_CONF"):
        monkeypatch.delenv(variable, raising=False)
    for item in yaml.safe_load((RL / "config/agent_loop_config.yaml").read_text()):
        _assert_class_exists(item["_target_"])
    for filename in ("tools.yaml",):
        tools = OmegaConf.load(RL / "config" / filename)
        for item in tools.tools:
            _assert_class_exists(item.class_name)
            for key in ("search_nodes_conf", "scholar_nodes_conf", "python_nodes_conf"):
                if key in item.config:
                    assert (ROOT / item.config[key]).is_file(), item.config[key]
    config = OmegaConf.load(RL / "config/paper.yaml")
    for path, class_name in (
        (config.reward_manager.module.path, config.reward_manager.name),
        (config.reward_model.reward_loop_module_path, config.reward_model.reward_loop_class_name),
    ):
        assert path.startswith("pkg://")
        _assert_class_exists(path.removeprefix("pkg://") + "." + class_name)


def test_local_python_imports_resolve_without_legacy_package_aliases():
    # Check module closure without importing training/service dependencies.
    local_names = {path.stem for path in RL.glob("*.py")}
    local_names.update(path.name for path in RL.iterdir() if (path / "__init__.py").is_file())
    for source in RL.rglob("*.py"):
        package_parts = source.parent.relative_to(ROOT).parts
        package = ".".join(package_parts)
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    names = [importlib.util.resolve_name("." * node.level + (node.module or ""), package)]
                else:
                    names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name.startswith(("axion.training.rl", "recipe.deepresearch")):
                    path = ROOT.joinpath(*name.split("."))
                elif name.split(".")[0] in local_names:
                    path = RL.joinpath(*name.split("."))
                else:
                    continue
                assert path.with_suffix(".py").is_file() or (path / "__init__.py").is_file(), (source, name)


@pytest.mark.parametrize("script,module", [
    ("init_faiss_search.sh", "axion.training.rl.scripts.build_search_faiss"),
    ("init_faiss_scholar.sh", "axion.training.rl.scripts.build_scholar_faiss"),
    ("run_search_service.sh", "axion.training.rl.tools.search_service"),
    ("run_scholar_service.sh", "axion.training.rl.tools.scholar_service"),
])
def test_helper_shell_dispatch_from_unrelated_cwd(script, module, tmp_path):
    # Intercept only the external process boundary; run the actual shell script.
    binary = tmp_path / "python"
    binary.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'argv': sys.argv[1:]}))\n"
    )
    binary.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
               DEEPRESEARCH_SECRETS_ENV=str(tmp_path / "absent-secrets"))
    env.pop("SEARCH_SERVICE_CONFIG", None)
    env.pop("SCHOLAR_SERVICE_CONFIG", None)
    result = subprocess.run(["bash", str(RL / "scripts" / script)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, check=True)
    dispatch = json.loads(result.stdout)
    assert Path(dispatch["cwd"]) == ROOT
    assert dispatch["argv"][:2] == ["-m", module]
    if "--config" in dispatch["argv"]:
        config = dispatch["argv"][dispatch["argv"].index("--config") + 1]
        assert Path(config).resolve() == RL / "config/tools.yaml"
    else:
        assert dispatch["argv"][2:] == ["--device", "cuda"]


def test_launcher_paths_and_pythonpath_resolve_from_unrelated_cwd(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(["bash", str(RL / "train.sh"), "--help"], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--deployment" in result.stdout and "--updates" in result.stdout


@pytest.mark.parametrize("relative,root_name", [
    ("scripts/build_search_faiss.py", "RECIPE_PARENT"),
    ("scripts/build_scholar_faiss.py", "RECIPE_PARENT"),
    ("tools/search_service.py", "_VERL_ROOT"),
    ("tools/scholar_service.py", "_VERL_ROOT"),
])
def test_python_helper_bootstrap_finds_repository(relative, root_name, tmp_path, monkeypatch):
    # Execute the real bootstrap assignments only, before service/GPU imports.
    path = RL / relative
    names = {"SCRIPT_DIR", "_SCRIPT_DIR", root_name}
    assignments = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    monkeypatch.chdir(tmp_path)
    namespace = {"__file__": str(path), "os": os}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    assert Path(namespace[root_name]) == ROOT
    assert (Path(namespace[root_name]) / "axion/__init__.py").is_file()


def test_script_modules_are_discoverable_as_regular_packages():
    # namespaces=false must not silently omit all FAISS/helper Python modules.
    from setuptools import find_packages
    assert "axion.training.rl.scripts" in find_packages(where=str(ROOT))


def test_relocated_shell_files_have_valid_syntax():
    for path in RL.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
