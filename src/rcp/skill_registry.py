from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

SkillKind = Literal["skill", "workflow"]

_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

_OFFICIAL_PACKAGE_SPECS: tuple[tuple[SkillKind, str, str], ...] = (
    ("skill", "graph-audit", "graph-audit/SKILL.md"),
    ("skill", "evidence-triage", "evidence-triage/SKILL.md"),
    ("skill", "experiment-causality", "experiment-causality/SKILL.md"),
    ("skill", "episode-report", "episode-report/SKILL.md"),
    ("workflow", "research-graph-audit", "workflows/research-graph-audit/WORKFLOW.md"),
)


def _default_skill_ids() -> list[str]:
    return sorted(
        package_id
        for kind, package_id, _relative_file in _OFFICIAL_PACKAGE_SPECS
        if kind == "skill"
    )


class SkillReference(BaseModel):
    """One immutable official package selected for a run."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: SkillKind
    version: str


class SkillDefaults(BaseModel):
    """Project defaults. These are IDs; a run resolves them to exact versions."""

    model_config = ConfigDict(extra="forbid")

    workflow_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=_default_skill_ids)


class SkillSelection(BaseModel):
    """The selection captured on a task before its provider is launched."""

    model_config = ConfigDict(extra="forbid")

    workflow_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    resolved_skill_packages: list[SkillReference] = Field(default_factory=list)


class SkillPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: SkillKind
    label: str
    version: str
    description: str
    dependencies: list[SkillReference] = Field(default_factory=list)
    _source_path: Path = PrivateAttr()

    @model_validator(mode="after")
    def validate_identity(self) -> SkillPackage:
        if not _ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"official skill package id is invalid: {self.id!r}")
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError(f"official skill package version is invalid: {self.version!r}")
        return self

    def reference(self) -> SkillReference:
        return SkillReference(id=self.id, kind=self.kind, version=self.version)

    def catalog_entry(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "version": self.version,
            "description": self.description,
            "dependencies": [item.model_dump(mode="json") for item in self.dependencies],
        }


class SkillRegistry:
    """The built-in, source-controlled RCP package registry.

    v1 deliberately has no user import or mutation path. Git versions the full
    package directories; task records retain the exact package references that
    were resolved at launch time.
    """

    def __init__(self, packages: list[SkillPackage], root: Path) -> None:
        self._root = root.resolve()
        self._packages = {(item.kind, item.id): item for item in packages}
        if len(self._packages) != len(packages):
            raise ValueError("official skill registry contains a duplicate package")
        self._validate_dependencies()

    @property
    def packages(self) -> tuple[SkillPackage, ...]:
        return tuple(self._packages[key] for key in sorted(self._packages))

    def catalog(self) -> list[dict[str, object]]:
        return [item.catalog_entry() for item in self.packages]

    def package(self, kind: SkillKind, package_id: str) -> SkillPackage:
        try:
            return self._packages[(kind, package_id)]
        except KeyError as exc:
            raise ValueError(f"official {kind} {package_id!r} is not available") from exc

    def package_path(self, reference: SkillReference) -> Path:
        return self.package(reference.kind, reference.id)._source_path

    def package_body(self, kind: SkillKind, package_id: str) -> str:
        """The package's own prose, for the read-only Settings inspector.

        The front matter is dropped: identity, version, and dependencies are
        already carried as fields, and rendering them again as a stray Markdown
        paragraph reads as noise.
        """

        package = self.package(kind, package_id)
        filename = "WORKFLOW.md" if kind == "workflow" else "SKILL.md"
        lines = (package._source_path / filename).read_text(encoding="utf-8").splitlines()
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
        return "\n".join(lines[end + 1 :]).strip() + "\n"

    def resolve(
        self,
        defaults: SkillDefaults | None = None,
        *,
        workflow_ids: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> SkillSelection:
        """Resolve selected ids and their workflow dependency closure.

        The registry as it stands right now is authoritative. A caller never
        supplies resolved packages: a stored selection records the ids a task
        chose, and every launch of that task — first attempt, retry, or resume —
        resolves them again here. Upgrading a package upgrades the next attempt,
        which is the intended behavior; a recorded version is a receipt of what
        an attempt ran with, never an input that could pin or fail a later one.
        """

        if workflow_ids is None:
            workflow_ids = list(defaults.workflow_ids if defaults else [])
        if skill_ids is None:
            skill_ids = list(defaults.skill_ids if defaults else [])
        workflow_ids = _unique_ids(workflow_ids, "workflow")
        skill_ids = _unique_ids(skill_ids, "skill")
        return SkillSelection(
            workflow_ids=workflow_ids,
            skill_ids=skill_ids,
            resolved_skill_packages=self._closure(workflow_ids, skill_ids),
        )

    def _closure(self, workflow_ids: list[str], skill_ids: list[str]) -> list[SkillReference]:
        resolved: list[SkillReference] = []
        seen: set[tuple[SkillKind, str]] = set()
        for package_id in workflow_ids:
            workflow = self.package("workflow", package_id)
            _append_reference(workflow.reference(), resolved, seen)
            for dependency in workflow.dependencies:
                # Kind and version are checked once, in _validate_dependencies.
                _append_reference(dependency, resolved, seen)
        for package_id in skill_ids:
            _append_reference(self.package("skill", package_id).reference(), resolved, seen)
        return resolved

    def _validate_dependencies(self) -> None:
        for package in self.packages:
            if package.dependencies and package.kind == "skill":
                raise ValueError("skills cannot declare dependencies in the v1 registry")
            for dependency in package.dependencies:
                if dependency.kind != "skill":
                    raise ValueError("workflows can depend only on skills in the v1 registry")
                dependency_package = self.package(dependency.kind, dependency.id)
                if dependency_package.version != dependency.version:
                    raise ValueError(
                        f"{package.kind} {package.id!r} depends on {dependency.id!r}@"
                        f"{dependency.version}, but the registry has {dependency_package.version}"
                    )


def _unique_ids(values: list[str], kind: SkillKind) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError(f"{kind} id is invalid: {value!r}")
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _append_reference(
    reference: SkillReference,
    resolved: list[SkillReference],
    seen: set[tuple[SkillKind, str]],
) -> None:
    key = (reference.kind, reference.id)
    if key in seen:
        prior = next(item for item in resolved if (item.kind, item.id) == key)
        if prior.version != reference.version:
            raise ValueError(
                f"package {reference.id!r} was selected at incompatible versions "
                f"{prior.version} and {reference.version}"
            )
        return
    seen.add(key)
    resolved.append(reference)


def _front_matter(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} must begin with an RCP package front matter block")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError(f"{path} has an unterminated front matter block") from exc
    values: dict[str, object] = {}
    dependencies: list[str] = []
    reading_dependencies = False
    for raw in lines[1:end]:
        line = raw.strip()
        if not line:
            continue
        if line == "dependencies:":
            reading_dependencies = True
            continue
        if line.startswith("- ") and reading_dependencies:
            dependencies.append(line[2:].strip())
            continue
        reading_dependencies = False
        if ":" not in line:
            raise ValueError(f"{path} has invalid front matter line: {raw!r}")
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    values["dependencies"] = dependencies
    return values


def _package(
    root: Path,
    *,
    kind: SkillKind,
    package_id: str,
    relative_file: str,
) -> SkillPackage:
    source_path = (root / relative_file).resolve()
    if source_path.parent.parent != root.resolve() and root.resolve() not in source_path.parents:
        raise ValueError(f"official package path escapes registry root: {relative_file}")
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError(f"official package file is not a regular file: {source_path}")
    metadata = _front_matter(source_path)
    if metadata.get("id") != package_id or metadata.get("kind") != kind:
        raise ValueError(f"front matter identity does not match registry entry {package_id!r}")
    dependencies: list[SkillReference] = []
    for raw in metadata.get("dependencies", []):
        if not isinstance(raw, str) or "@" not in raw:
            raise ValueError(f"dependency in {source_path} must be written as id@version")
        dependency_id, version = raw.rsplit("@", 1)
        dependencies.append(SkillReference(id=dependency_id, kind="skill", version=version))
    package = SkillPackage(
        id=package_id,
        kind=kind,
        label=str(metadata.get("label") or package_id),
        version=str(metadata.get("version") or ""),
        description=str(metadata.get("description") or ""),
        dependencies=dependencies,
    )
    package._source_path = source_path.parent
    return package


def official_registry() -> SkillRegistry:
    root = Path(__file__).parent / "skills"
    packages = [
        _package(root, kind=kind, package_id=package_id, relative_file=relative_file)
        for kind, package_id, relative_file in _OFFICIAL_PACKAGE_SPECS
    ]
    return SkillRegistry(packages, root)
