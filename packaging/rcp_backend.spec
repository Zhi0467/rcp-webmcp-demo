from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


PROJECT_ROOT = Path(SPECPATH).parent
SOURCE_ROOT = PROJECT_ROOT / "src"
WEB_DIST = PROJECT_ROOT / "web" / "dist"
RECORD_PARSER = SOURCE_ROOT / "rcp" / "sources" / "record_parsing.py"
STAGED_COMMAND_CLIENT = SOURCE_ROOT / "rcp" / "agents" / "staged_command_client.py"
STAGED_COMMAND_BROKER = SOURCE_ROOT / "rcp" / "agents" / "staged_command_broker.py"
TRANSPORT_ROOT = SOURCE_ROOT / "rcp" / "transport"
REMOTE_LOCK_HOLDER = TRANSPORT_ROOT / "remote_lock_holder.py"
REMOTE_ARCHIVE_RESEARCH = TRANSPORT_ROOT / "remote_archive_research.py"
REMOTE_READ_KEPT_VIEW = TRANSPORT_ROOT / "remote_read_kept_view.py"
SKILL_ROOT = SOURCE_ROOT / "rcp" / "skills"
SKILL_GRAPH_AUDIT = SKILL_ROOT / "graph-audit"
SKILL_EVIDENCE_TRIAGE = SKILL_ROOT / "evidence-triage"
SKILL_EXPERIMENT_CAUSALITY = SKILL_ROOT / "experiment-causality"
SKILL_EPISODE_REPORT = SKILL_ROOT / "episode-report"
WORKFLOW_REGISTRY = SKILL_ROOT / "workflows"
RUNTIME_HOOK = PROJECT_ROOT / "packaging" / "hooks" / "validate_frozen_resources.py"

if not (WEB_DIST / "index.html").is_file():
    raise SystemExit("web/dist is missing; run the frontend build before PyInstaller")

analysis = Analysis(
    [str(SOURCE_ROOT / "rcp" / "__main__.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=[],
    datas=[
        (str(WEB_DIST), "rcp/web_dist"),
        (str(RECORD_PARSER), "rcp/sources"),
        (str(STAGED_COMMAND_CLIENT), "rcp/agents"),
        (str(STAGED_COMMAND_BROKER), "rcp/agents"),
        (str(REMOTE_LOCK_HOLDER), "rcp/transport"),
        (str(REMOTE_ARCHIVE_RESEARCH), "rcp/transport"),
        (str(REMOTE_READ_KEPT_VIEW), "rcp/transport"),
        (str(SKILL_GRAPH_AUDIT), "rcp/skills/graph-audit"),
        (str(SKILL_EVIDENCE_TRIAGE), "rcp/skills/evidence-triage"),
        (str(SKILL_EXPERIMENT_CAUSALITY), "rcp/skills/experiment-causality"),
        (str(SKILL_EPISODE_REPORT), "rcp/skills/episode-report"),
        (str(WORKFLOW_REGISTRY), "rcp/skills/workflows"),
    ],
    hiddenimports=collect_submodules("uvicorn"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(RUNTIME_HOOK)],
    excludes=["PyInstaller", "pytest", "ruff"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="rcp-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
