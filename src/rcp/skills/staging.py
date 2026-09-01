from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath

from rcp.skill_registry import SkillRegistry, SkillSelection, official_registry
from rcp.transport import RemoteRunStage


def skill_bundle_label(selection: SkillSelection) -> str:
    """Return the stable content label for one resolved official package bundle.

    The label covers the staged bytes, not only the declared versions, because a
    reused bundle is accepted only when its whole tree still matches. Addressing
    it by version alone let an edited package keep its label, so an upgraded RCP
    met the previous release's immutable bundle and every reusing turn failed at
    staging. Editing a package now yields a new label and a new bundle instead,
    and the superseded one is left untouched for its stage's sweeper.
    """

    registry = official_registry()
    packages = sorted(
        (reference.kind, reference.id, reference.version)
        for reference in selection.resolved_skill_packages
    )
    contents = sorted(
        (path, entry_kind, digest)
        for path, (entry_kind, digest) in _selection_manifest(registry, selection).items()
    )
    payload = json.dumps([packages, contents], ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(payload).hexdigest()
    return f"rcp-skills-v1-{digest}"


def stage_skill_selection(
    selection: SkillSelection,
    *,
    local_stage: Path | None,
    remote_stage: RemoteRunStage | None,
    label: str,
    reuse_existing: bool = False,
) -> list[dict[str, object]]:
    """Stage one resolved official selection and return prompt-safe pointers.

    The source-controlled package directories are copied into the run stage as
    immutable inputs. Existing callers stage one bundle per attempt. Persistent
    chat stages may opt into reuse with a content-addressed label from
    :func:`skill_bundle_label`; an existing bundle is accepted only when its
    full tree is safe, immutable, and identical to the resolved official
    packages.
    """

    if (local_stage is None) == (remote_stage is None):
        raise ValueError("exactly one task stage must be selected")
    if not label or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
        for character in label
    ):
        raise ValueError("skill staging label contains unsupported characters")
    if reuse_existing and label != skill_bundle_label(selection):
        raise ValueError("reusable skill staging requires its content-addressed label")
    if not selection.resolved_skill_packages:
        return []

    registry = official_registry()
    if remote_stage is not None:
        if remote_stage.root is None:
            raise RuntimeError("remote run stage is not open")
        with tempfile.TemporaryDirectory(prefix="rcp-skill-bundle-") as temporary:
            source_bundle = Path(temporary)
            _copy_packages(registry, selection, source_bundle)
            remote_stage.put_directory(source_bundle, label, reuse=reuse_existing)
        return _pointers(registry, selection, remote_stage.root / "inputs" / label)

    assert local_stage is not None
    if reuse_existing and (local_stage.is_symlink() or not local_stage.is_dir()):
        raise ValueError("persistent local skill stage is not a safe directory")
    inputs = local_stage / "inputs"
    if reuse_existing and os.path.lexists(inputs) and (inputs.is_symlink() or not inputs.is_dir()):
        raise ValueError("persistent local skill input root is not a safe directory")
    inputs.mkdir(mode=0o700, parents=True, exist_ok=True)
    bundle = inputs / label
    if os.path.lexists(bundle):
        if reuse_existing:
            _validate_reusable_bundle(registry, selection, bundle)
            return _pointers(registry, selection, bundle)
        raise ValueError("immutable skill staging bundle already exists")
    if reuse_existing:
        _stage_reusable_local_bundle(registry, selection, inputs, bundle)
        return _pointers(registry, selection, bundle)
    bundle.mkdir(mode=0o700)
    _copy_packages(registry, selection, bundle)
    _protect_tree(bundle)
    return _pointers(registry, selection, bundle)


def _stage_reusable_local_bundle(
    registry: SkillRegistry,
    selection: SkillSelection,
    inputs: Path,
    bundle: Path,
) -> None:
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{bundle.name}-", dir=inputs))
    try:
        _copy_packages(registry, selection, temporary)
        _protect_tree(temporary)
        try:
            temporary.rename(bundle)
        except OSError:
            if not os.path.lexists(bundle):
                raise
            _validate_reusable_bundle(registry, selection, bundle)
        else:
            temporary = None
    finally:
        if temporary is not None and os.path.lexists(temporary):
            _make_tree_writable(temporary)
            shutil.rmtree(temporary)


def _validate_reusable_bundle(
    registry: SkillRegistry,
    selection: SkillSelection,
    bundle: Path,
) -> None:
    expected = _selection_manifest(registry, selection)
    actual = _tree_manifest(bundle, require_immutable=True)
    if actual != expected:
        raise ValueError("existing immutable skill staging bundle does not match its selection")


def _selection_manifest(
    registry: SkillRegistry,
    selection: SkillSelection,
) -> dict[str, tuple[str, str]]:
    manifest: dict[str, tuple[str, str]] = {}
    for reference in selection.resolved_skill_packages:
        kind_prefix = Path(reference.kind)
        package_prefix = kind_prefix / reference.id
        manifest.setdefault(kind_prefix.as_posix(), ("directory", ""))
        manifest[package_prefix.as_posix()] = ("directory", "")
        source = registry.package_path(reference)
        for relative, value in _tree_manifest(source, require_immutable=False).items():
            manifest[(package_prefix / relative).as_posix()] = value
    return manifest


def _tree_manifest(root: Path, *, require_immutable: bool) -> dict[str, tuple[str, str]]:
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ValueError("existing skill staging bundle is unavailable") from exc
    if root.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("skill staging bundle is not a safe directory")
    if require_immutable and root_info.st_mode & 0o222:
        raise ValueError("existing skill staging bundle is writable")

    manifest: dict[str, tuple[str, str]] = {}

    def visit(directory: Path, prefix: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError("skill staging bundle cannot be inspected safely") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = prefix / entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError("skill staging bundle cannot be inspected safely") from exc
            if entry.is_symlink():
                raise ValueError("skill staging bundle contains a symlink")
            if require_immutable and info.st_mode & 0o222:
                raise ValueError("existing skill staging bundle contains a writable entry")
            if stat.S_ISDIR(info.st_mode):
                manifest[relative.as_posix()] = ("directory", "")
                visit(path, relative)
            elif stat.S_ISREG(info.st_mode):
                digest = hashlib.sha256()
                try:
                    with path.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                except OSError as exc:
                    raise ValueError("skill staging bundle cannot be read safely") from exc
                manifest[relative.as_posix()] = ("file", digest.hexdigest())
            else:
                raise ValueError("skill staging bundle contains a non-regular entry")

    visit(root, Path())
    return manifest


def _copy_packages(registry: SkillRegistry, selection: SkillSelection, destination: Path) -> None:
    for reference in selection.resolved_skill_packages:
        source = registry.package_path(reference)
        target = destination / reference.kind / reference.id
        target.parent.mkdir(mode=0o700, exist_ok=True)
        shutil.copytree(source, target, symlinks=False)


def _protect_tree(root: Path) -> None:
    for directory, _children, files in os.walk(root):
        Path(directory).chmod(0o500)
        for filename in files:
            path = Path(directory) / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("official skill staging contains a non-regular file")
            path.chmod(0o400)


def _make_tree_writable(root: Path) -> None:
    for directory, _children, files in os.walk(root):
        Path(directory).chmod(0o700)
        for filename in files:
            path = Path(directory) / filename
            if not path.is_symlink():
                path.chmod(0o600)


def _pointers(
    registry: SkillRegistry,
    selection: SkillSelection,
    bundle: Path | PurePosixPath,
) -> list[dict[str, object]]:
    pointers: list[dict[str, object]] = []
    for reference in selection.resolved_skill_packages:
        package = registry.package(reference.kind, reference.id)
        dependencies = ", ".join(f"{item.id}@{item.version}" for item in package.dependencies)
        pointers.append(
            {
                "id": reference.id,
                "kind": reference.kind,
                "label": package.label,
                "description": package.description,
                "version": reference.version,
                "path": str(bundle / reference.kind / reference.id),
                "dependencies": dependencies,
            }
        )
    return pointers
