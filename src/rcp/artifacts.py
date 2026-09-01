from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import stat
import xml.etree.ElementTree as ET
from contextlib import suppress
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

ArtifactMediaType = Literal[
    "text/html",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/svg+xml",
]

# Supported file types are an artifact contract, not an operational tuning knob.
ARTIFACT_MEDIA_TYPES: dict[str, ArtifactMediaType] = {
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


class AgentArtifactDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    name: str = Field(min_length=1, max_length=255)
    media_type: ArtifactMediaType
    size_bytes: int | None = Field(default=None, ge=1)
    kept_filename: str | None = Field(default=None, min_length=1, max_length=255)
    kept_at: str | None = None

    def is_kept(self) -> bool:
        return self.kept_filename is not None


class ResultViewDescriptor(BaseModel):
    """Public metadata for one stable, conversation-scoped result view."""

    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    chat_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=255)
    media_type: Literal["text/html"]
    state: Literal["temporary", "kept"]
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)
    kept_filename: str | None = None
    kept_at: str | None = None
    can_revise: bool


def validate_result_view_id(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{24}", value) is None:
        raise ValueError("result view id must be exactly 24 lowercase hexadecimal characters")
    return value


def artifact_id(scope_id: str, name: str) -> str:
    """Return an opaque, task-scope-bound identifier without exposing a path."""
    return hashlib.sha256(f"{scope_id}\0{name}".encode()).hexdigest()[:24]


def descriptor_for(
    scope_id: str,
    name: str,
    *,
    size_bytes: int | None = None,
    kept_filename: str | None = None,
    kept_at: str | None = None,
) -> AgentArtifactDescriptor:
    media_type = ARTIFACT_MEDIA_TYPES[Path(name).suffix.casefold()]
    return AgentArtifactDescriptor(
        artifact_id=artifact_id(scope_id, name),
        name=name,
        media_type=media_type,
        size_bytes=size_bytes,
        kept_filename=kept_filename,
        kept_at=kept_at,
    )


def validate_artifact_bytes(name: str, data: bytes) -> ArtifactMediaType:
    """Validate extension, bounded caller-provided bytes, and the format signature."""
    try:
        media_type = ARTIFACT_MEDIA_TYPES[Path(name).suffix.casefold()]
    except KeyError as exc:
        raise ValueError("unsupported artifact type") from exc
    valid = {
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": data.startswith(b"\xff\xd8\xff"),
        "image/gif": data.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP",
        "image/svg+xml": False,
    }
    if media_type == "text/html":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("HTML artifact is not UTF-8") from exc
        if "\x00" in text:
            raise ValueError("HTML artifact contains NUL bytes")
    elif media_type == "image/svg+xml":
        try:
            root = ET.fromstring(data.decode("utf-8"))
        except (UnicodeDecodeError, ET.ParseError) as exc:
            raise ValueError("SVG artifact is not valid UTF-8 XML") from exc
        if root.tag.rsplit("}", 1)[-1].casefold() != "svg":
            raise ValueError("SVG artifact must have an svg root element")
    elif not valid[media_type]:
        raise ValueError(f"artifact bytes do not match {media_type}")
    return media_type


def read_local_regular_file(directory: Path, name: str, *, max_bytes: int) -> bytes:
    """Read one direct regular child without following a symlink."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("artifact name must be a plain base name")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = _open_local_directory(directory)
    try:
        try:
            file_fd = os.open(name, os.O_RDONLY | no_follow, dir_fd=directory_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ValueError("artifact is not a readable regular file") from exc
        try:
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("artifact is not a regular file")
            if metadata.st_size > max_bytes:
                raise ValueError("artifact exceeds the per-file limit")
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(file_fd, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise ValueError("artifact exceeds the per-file limit")
            return data
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def list_local_regular_files(directory: Path) -> list[tuple[str, int]]:
    """List direct regular children without following any directory symlink."""
    directory_fd = _open_local_directory(directory)
    try:
        values: list[tuple[str, int]] = []
        for name in os.listdir(directory_fd):
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode):
                values.append((name, metadata.st_size))
        return sorted(values)
    finally:
        os.close(directory_fd)


def replace_local_regular_file(directory: Path, name: str, data: bytes) -> None:
    """Atomically replace one direct regular child without a digest precondition."""

    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("artifact name must be a plain base name")
    directory_fd = _open_local_directory(directory)
    temporary_name = f".{name}.rcp-{secrets.token_hex(8)}"
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("artifact is not a regular file")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short artifact replacement write")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        raise
    finally:
        os.close(directory_fd)


def _open_local_directory(directory: Path) -> int:
    if not directory.is_absolute():
        raise ValueError("artifact directory must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open("/", flags)
    try:
        for part in directory.parts[1:]:
            following = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = following
        return current
    except FileNotFoundError:
        os.close(current)
        raise
    except OSError as exc:
        os.close(current)
        raise ValueError("artifact directory is not a regular directory") from exc


class _ArtifactHTMLSanitizer(HTMLParser):
    """Neutralize browser capabilities while preserving inline presentation and scripts."""

    _request_attributes = {
        "src",
        "srcset",
        "poster",
        "action",
        "formaction",
        "ping",
        "data",
        "codebase",
        "background",
        "manifest",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "meta" and any(
            name.casefold() == "http-equiv" and (value or "").casefold() == "refresh"
            for name, value in attrs
        ):
            return
        rendered: list[tuple[str, str | None]] = []
        for name, value in attrs:
            lowered = name.casefold()
            if (
                lowered in self._request_attributes
                or lowered in {"download", "target"}
                or lowered.endswith(":href")
                or lowered.endswith(":src")
            ):
                continue
            if lowered == "href":
                if tag == "a" and value and _is_http_url(value):
                    rendered.append(("data-rcp-href", value))
                continue
            if lowered == "http-equiv" and tag == "meta":
                continue
            rendered.append((name, value))
        self.parts.append(f"<{tag}{_html_attributes(rendered)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        before = len(self.parts)
        self.handle_starttag(tag, attrs)
        if len(self.parts) > before:
            self.parts[-1] = self.parts[-1][:-1] + "/>"

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.parts.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self.parts.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self.parts.append(f"<?{data}>")


def html_preview_document(data: bytes, *, result_view_gestures: bool = False) -> tuple[str, str]:
    """Build an RCP-owned wrapper and its CSP for an opaque sandboxed document."""
    source = data.decode("utf-8")
    sanitizer = _ArtifactHTMLSanitizer()
    sanitizer.feed(source)
    sanitizer.close()
    secret = secrets.token_urlsafe(24)
    secret_json = json.dumps(secret)
    bootstrap = f"""<script>(()=>{{
const secret={secret_json};
const send=window.parent.postMessage.bind(window.parent);
const closest=Element.prototype.closest;
const utf8=new TextEncoder();
const bounded=(value,limit)=>{{
  const text=String(value||'').replace(/\\s+/g,' ').trim();
  if(utf8.encode(text).byteLength<=limit) return text;
  let result='';
  for(const character of text){{
    if(utf8.encode(result+character).byteLength>limit) break;
    result+=character;
  }}
  return result;
}};
window.addEventListener('click',(event)=>{{
  if(!event.isTrusted || !(event.target instanceof Element)) return;
  const anchor=closest.call(event.target,'a[data-rcp-href]');
  if(!anchor) return;
  event.preventDefault(); event.stopImmediatePropagation();
  send({{kind:'rcp-reference',secret,url:anchor.getAttribute('data-rcp-href')}},'*');
}},true);
document.addEventListener('mouseup',()=>{{
  const selection=document.getSelection();
  const text=bounded(selection?.toString(),4096);
  if(!text || !selection?.rangeCount) return;
  const range=selection.getRangeAt(0);
  const container=range.commonAncestorContainer.nodeType===Node.ELEMENT_NODE
    ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
  const surrounding=bounded(container?.textContent,6144);
  send({{kind:'rcp-artifact-selection',secret,selection:{{kind:'text',text,surrounding_text:surrounding}}}},'*');
}});
let boxing=false,startX=0,startY=0,box=null;
window.addEventListener('message',(event)=>{{
  if(event.source!==window.parent || !event.data ||
     event.data.kind!=='rcp-artifact-box-start' || event.data.secret!==secret) return;
  boxing=true;
  document.documentElement.style.cursor='crosshair';
}});
document.addEventListener('pointerdown',(event)=>{{
  if(!boxing || event.button!==0) return;
  event.preventDefault(); event.stopImmediatePropagation();
  startX=event.clientX; startY=event.clientY;
  box=document.createElement('div');
  Object.assign(box.style,{{position:'fixed',zIndex:'2147483647',pointerEvents:'none',
    border:'2px solid #bd5b36',background:'rgba(189,91,54,.12)',left:`${{startX}}px`,top:`${{startY}}px`}});
  document.documentElement.appendChild(box);
}},true);
document.addEventListener('pointermove',(event)=>{{
  if(!box) return;
  const left=Math.min(startX,event.clientX),top=Math.min(startY,event.clientY);
  Object.assign(box.style,{{left:`${{left}}px`,top:`${{top}}px`,
    width:`${{Math.abs(event.clientX-startX)}}px`,height:`${{Math.abs(event.clientY-startY)}}px`}});
}},true);
document.addEventListener('pointerup',(event)=>{{
  if(!box) return;
  event.preventDefault(); event.stopImmediatePropagation();
  const left=Math.max(0,Math.min(startX,event.clientX));
  const top=Math.max(0,Math.min(startY,event.clientY));
  const right=Math.min(innerWidth,Math.max(startX,event.clientX));
  const bottom=Math.min(innerHeight,Math.max(startY,event.clientY));
  box.remove(); box=null; boxing=false; document.documentElement.style.cursor='';
  const boxWidth=right-left,boxHeight=bottom-top;
  if(boxWidth<=0||boxHeight<=0) return;
  const labels=[];
  const seen=new Set();
  const steps=5;
  for(let xi=0;xi<=steps;xi++) for(let yi=0;yi<=steps;yi++){{
    const x=left+(right-left)*xi/steps,y=top+(bottom-top)*yi/steps;
    let element=document.elementFromPoint(x,y);
    for(let depth=0;element&&depth<3;depth++,element=element.parentElement){{
      const tag=element.tagName?.toLowerCase();
      if(tag==='html'||tag==='body'||tag==='head'||tag==='style'||tag==='script') continue;
      const value=bounded(element.getAttribute?.('aria-label') || element.textContent,512);
      if(value){{if(!seen.has(value)){{seen.add(value);labels.push(value);}}break;}}
    }}
  }}
  send({{kind:'rcp-artifact-selection',secret,selection:{{kind:'box',
    rect:{{x:left/innerWidth,y:top/innerHeight,width:boxWidth/innerWidth,height:boxHeight/innerHeight}},
    viewport:{{width:innerWidth,height:innerHeight}},labels:bounded(labels.join(' | '),4096)}}}},'*');
}},true);
document.currentScript?.remove();
}})();</script>"""
    # Chromium does not currently enforce ``navigate-to``. The opaque sandbox is
    # the boundary that prevents this document from navigating the RCP parent;
    # inline scripts may still navigate their own isolated child frame. Keep the
    # directive as defense in depth for engines that do implement it.
    artifact_csp = (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src data: blob:; font-src data:; connect-src 'none'; object-src 'none'; "
        "frame-src 'none'; child-src 'none'; media-src 'none'; worker-src 'none'; "
        "form-action 'none'; base-uri 'none'; navigate-to 'none'"
    )
    artifact = (
        f'<meta http-equiv="Content-Security-Policy" content="{html.escape(artifact_csp)}">'
        + bootstrap
        + "".join(sanitizer.parts)
    )
    wrapper_script = f"""<script>
const artifact=document.getElementById('artifact');
window.addEventListener('message',(event)=>{{
  const value=event.data;
  if(event.source!==artifact.contentWindow || !value || value.secret!=={secret_json}) return;
  if(value.kind==='rcp-artifact-selection' && value.selection && window.parent!==window){{
    window.parent.postMessage({{type:'rcp-artifact-selection',version:1,selection:value.selection}},'*');
    return;
  }}
  if(value.kind!=='rcp-reference' || typeof value.url!=='string') return;
  try {{
    const target=new URL(value.url);
    if(target.protocol==='http:' || target.protocol==='https:')
      window.open(target.href,'_blank','noopener,noreferrer');
  }} catch {{}}
}});
window.addEventListener('message',(event)=>{{
  if(event.source!==window.parent || !event.data ||
     event.data.type!=='rcp-artifact-box-start') return;
  artifact.contentWindow?.postMessage({{kind:'rcp-artifact-box-start',secret:{secret_json}}},'*');
}});
    </script>"""
    if result_view_gestures:
        wrapper_script += """<script>(()=>{
const legacyArtifact=document.getElementById('artifact');
const expectedKeys=['description','gesture','type','version'];
const utf8=new TextEncoder();
window.addEventListener('message',(event)=>{
  const value=event.data;
  if(event.source!==legacyArtifact.contentWindow || !value || typeof value!=='object') return;
  const keys=Object.keys(value).sort();
  if(keys.length!==expectedKeys.length ||
     keys.some((key,index)=>key!==expectedKeys[index])) return;
  if(value.type!=='rcp-result-view-gesture' || value.version!==1 ||
     (value.gesture!=='box' && value.gesture!=='underscore') ||
     typeof value.description!=='string' || !value.description.trim() ||
     utf8.encode(value.description).byteLength>2048) return;
  if(window.parent===window) return;
  window.parent.postMessage({
    type:'rcp-result-view-gesture',
    version:1,
    gesture:value.gesture,
    description:value.description
  },'*');
});
})();</script>"""
    document = (
        '<!doctype html><meta charset="utf-8">'
        "<title>Artifact preview</title>"
        "<style>html,body,iframe{border:0;margin:0;width:100%;height:100%;display:block}</style>"
        f'<iframe id="artifact" sandbox="allow-scripts" srcdoc="{html.escape(artifact, quote=True)}">'
        "</iframe>" + wrapper_script
    )
    wrapper_csp = (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "frame-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'"
    )
    return document, wrapper_csp


def artifact_viewer_document(
    *,
    preview_url: str,
    keep_url: str | None,
    project_id: str,
    chat_id: str | None,
    operation_id: str,
    descriptor: AgentArtifactDescriptor,
    source: Literal["task", "episode_report"] = "task",
    episode_id: str | None = None,
) -> tuple[str, str]:
    """Build the common task-artifact shell around one isolated preview."""

    def js(value: object) -> str:
        return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")

    config = {
        "previewUrl": preview_url,
        "keepUrl": keep_url,
        "projectId": project_id,
        "chatId": chat_id,
        "chatAvailable": chat_id is not None,
        "operationId": operation_id,
        "source": source,
        "episodeId": episode_id,
        "artifactId": descriptor.artifact_id,
        "artifactName": descriptor.name,
        "mediaType": descriptor.media_type,
        "kept": descriptor.kept_filename is not None,
    }
    preview_markup = (
        f'<iframe id="preview" sandbox="allow-scripts" src={js(preview_url)} '
        f"title={js(descriptor.name)}></iframe>"
        if descriptor.media_type == "text/html"
        else f'<img id="previewImage" src={js(preview_url)} alt={js(descriptor.name)}>'
        '<div id="boxLayer" aria-hidden="true"></div>'
    )
    if chat_id is None and source == "task":
        readonly_preview = (
            f'<iframe id="preview" sandbox="allow-scripts" src={js(preview_url)} '
            f"title={js(descriptor.name)}></iframe>"
            if descriptor.media_type == "text/html"
            else f'<img id="previewImage" src={js(preview_url)} alt={js(descriptor.name)}>'
        )
        readonly_keep = (
            '<button id="keep" type="button">Keep</button>'
            if keep_url and descriptor.kept_filename is None
            else ""
        )
        readonly_script = (
            f"""<script>(()=>{{const keep=document.getElementById('keep');keep?.addEventListener('click',async()=>{{keep.disabled=true;try{{const response=await fetch({js(keep_url)},{{method:'POST',credentials:'same-origin'}});if(!response.ok)throw new Error('Keep failed');document.getElementById('state').textContent='kept';keep.remove();}}catch{{keep.disabled=false;}}}});}})();</script>"""
            if readonly_keep
            else ""
        )
        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(descriptor.name)}</title>
<style>
:root{{--paper:#f4f1e8;--ink:#211f1a;--muted:#736f65;--rule:#c9c3b5;--panel:#fbfaf5}}
*{{box-sizing:border-box}}html,body{{margin:0;height:100%;background:var(--paper);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
body{{display:grid;grid-template-rows:48px minmax(0,1fr)}}header{{display:flex;align-items:center;gap:12px;padding:0 16px;border-bottom:1px solid var(--rule);background:var(--panel)}}
header strong{{font-family:Georgia,serif;font-size:16px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}header .state{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}header button{{margin-left:auto;border:1px solid var(--rule);background:transparent;color:var(--ink);padding:6px 10px;font:inherit;cursor:pointer}}
main{{min-height:0;background:white}}iframe{{display:block;border:0;width:100%;height:100%}}main>img{{display:block;width:100%;height:100%;object-fit:contain}}
</style></head><body>
<header><strong>{html.escape(descriptor.name)}</strong><span id="state" class="state">{"kept" if descriptor.kept_filename else "temporary"}</span>{readonly_keep}</header>
<main>{readonly_preview}</main>{readonly_script}</body></html>"""
        csp = (
            "default-src 'none'; "
            + ("script-src 'unsafe-inline'; connect-src 'self'; " if readonly_keep else "")
            + "style-src 'unsafe-inline'; frame-src 'self'; img-src 'self' data: blob:; "
            "base-uri 'none'; form-action 'none'; object-src 'none'"
        )
        return document, csp
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(descriptor.name)}</title>
<style>
:root{{--paper:#f4f1e8;--ink:#211f1a;--muted:#736f65;--rule:#c9c3b5;--accent:#a94f31;--panel:#fbfaf5}}
*{{box-sizing:border-box}}html,body{{margin:0;height:100%;background:var(--paper);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
body{{display:grid;grid-template-rows:48px minmax(0,1fr)}}
header{{display:flex;align-items:center;gap:12px;padding:0 16px;border-bottom:1px solid var(--rule);background:var(--panel)}}
header strong{{font-family:Georgia,serif;font-size:16px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
header .state{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
button{{border:1px solid var(--rule);background:transparent;color:var(--ink);padding:6px 10px;border-radius:2px;font:inherit;cursor:pointer}}
button:hover{{border-color:var(--accent);color:var(--accent)}}button:disabled{{opacity:.45;cursor:default}}
.spacer{{flex:1}}main{{display:grid;grid-template-columns:minmax(0,1fr) 300px;min-height:0}}
.canvas{{position:relative;min-width:0;background:white;border-right:1px solid var(--rule)}}
iframe{{display:block;border:0;width:100%;height:100%}}.canvas>img{{display:block;width:100%;height:100%;object-fit:contain}}#boxLayer{{display:none;position:absolute;inset:0;cursor:crosshair}}#boxLayer.active{{display:block}}#boxLayer div{{position:absolute;border:2px solid var(--accent);background:rgba(169,79,49,.12)}}aside{{padding:14px;overflow:auto;background:var(--panel)}}
aside h2{{margin:0 0 12px;font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}}
.empty{{color:var(--muted);font-family:Georgia,serif;font-style:italic}}.selection{{border-top:1px solid var(--rule);padding:12px 0}}
.selection b{{display:block;margin-bottom:5px;color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
.excerpt{{max-height:90px;overflow:auto;font-family:Georgia,serif;font-size:13px}}
textarea{{width:100%;min-height:62px;margin-top:8px;resize:vertical;border:1px solid var(--rule);background:white;padding:8px;color:var(--ink);font:13px/1.4 Georgia,serif}}
.add{{width:100%;margin-top:12px;background:var(--ink);color:var(--paper);border-color:var(--ink)}}.add:hover{{background:var(--accent);color:white}}
.notice{{margin-top:10px;color:var(--accent);font-size:12px}}@media(max-width:760px){{main{{grid-template-columns:1fr;grid-template-rows:minmax(360px,1fr) auto}}.canvas{{border-right:0;border-bottom:1px solid var(--rule)}}aside{{max-height:42vh}}}}
</style></head><body>
<header><strong>{html.escape(descriptor.name)}</strong><span id="state" class="state">{"kept" if descriptor.kept_filename else "temporary"}</span><span class="spacer"></span><button id="box" type="button">Box</button>{'<button id="keep" type="button">Keep</button>' if keep_url and descriptor.kept_filename is None else ""}</header>
<main><div class="canvas">{preview_markup}</div>
<aside><h2>Selections</h2><div id="empty" class="empty">Select text in the artifact or draw a box.</div><div id="items"></div><button id="add" class="add" type="button" disabled>Add to chat</button><div id="notice" class="notice" role="status"></div></aside></main>
<script>(()=>{{
const config={js(config)};const selections=[];const frame=document.getElementById('preview'),boxLayer=document.getElementById('boxLayer');
const items=document.getElementById('items'),empty=document.getElementById('empty'),add=document.getElementById('add'),notice=document.getElementById('notice');
const bounded=(value,limit)=>String(value||'').replace(/\\s+/g,' ').trim().slice(0,limit);
function render(){{items.replaceChildren();empty.hidden=selections.length>0;add.disabled=selections.length===0||!config.chatAvailable;
  selections.forEach((selection,index)=>{{const card=document.createElement('section');card.className='selection';
    const label=document.createElement('b');label.textContent=`${{index+1}} · ${{selection.kind}}`;
    const excerpt=document.createElement('div');excerpt.className='excerpt';excerpt.textContent=selection.kind==='text'?selection.text:(selection.labels||`Box ${{Math.round(selection.rect.x*100)}}–${{Math.round((selection.rect.x+selection.rect.width)*100)}}%`);
    const comment=document.createElement('textarea');comment.placeholder='Comment or question';comment.value=selection.comment||'';comment.addEventListener('input',()=>selection.comment=bounded(comment.value,2048));
    card.append(label,excerpt,comment);items.append(card);}});
}}
function appendSelection(selection){{if(selections.length>=12){{notice.textContent='A prompt can include at most 12 selections.';return;}}selections.push(selection);render();}}
window.addEventListener('message',(event)=>{{if(!frame||event.source!==frame.contentWindow) return;const value=event.data;
  if(!value||value.type!=='rcp-artifact-selection'||value.version!==1||!value.selection) return;
  const raw=value.selection;if(raw.kind==='text'&&typeof raw.text==='string') appendSelection({{kind:'text',text:bounded(raw.text,4096),surrounding_text:bounded(raw.surrounding_text,6144),comment:''}});
  else if(raw.kind==='box'&&raw.rect&&raw.viewport) appendSelection({{kind:'box',rect:raw.rect,viewport:raw.viewport,labels:bounded(raw.labels,4096),comment:''}});
}});
document.getElementById('box').addEventListener('click',()=>{{if(frame) frame.contentWindow?.postMessage({{type:'rcp-artifact-box-start'}},'*');else boxLayer?.classList.add('active');}});
if(boxLayer){{let start=null,mark=null;boxLayer.addEventListener('pointerdown',(event)=>{{start={{x:event.offsetX,y:event.offsetY}};mark=document.createElement('div');boxLayer.append(mark);}});boxLayer.addEventListener('pointermove',(event)=>{{if(!start||!mark)return;const left=Math.min(start.x,event.offsetX),top=Math.min(start.y,event.offsetY);Object.assign(mark.style,{{left:`${{left}}px`,top:`${{top}}px`,width:`${{Math.abs(event.offsetX-start.x)}}px`,height:`${{Math.abs(event.offsetY-start.y)}}px`}});}});boxLayer.addEventListener('pointerup',(event)=>{{if(!start||!mark)return;const width=boxLayer.clientWidth,height=boxLayer.clientHeight,left=Math.min(start.x,event.offsetX),top=Math.min(start.y,event.offsetY),right=Math.max(start.x,event.offsetX),bottom=Math.max(start.y,event.offsetY),boxWidth=right-left,boxHeight=bottom-top;mark.remove();mark=null;start=null;boxLayer.classList.remove('active');if(boxWidth<=0||boxHeight<=0)return;appendSelection({{kind:'box',rect:{{x:left/width,y:top/height,width:boxWidth/width,height:boxHeight/height}},viewport:{{width,height}},labels:'',comment:''}});}});}}
add.addEventListener('click',()=>{{if(!config.chatAvailable){{notice.textContent='The originating chat is unavailable.';return;}}const payload={{type:'rcp-artifact-context',version:1,project_id:config.projectId,chat_id:config.chatId,operation_id:config.operationId,artifact_id:config.artifactId,artifact_name:config.artifactName,media_type:config.mediaType,selections}};
  payload.source=config.source;payload.episode_id=config.episodeId;const key=`rcp:artifact-context:${{encodeURIComponent(config.projectId)}}:${{encodeURIComponent(config.chatId)}}`;localStorage.setItem(key,JSON.stringify(payload));
  try{{const channel=new BroadcastChannel('rcp-artifact-context');channel.postMessage(payload);channel.close();}}catch{{}}
  notice.textContent='Added to the originating chat draft.';
}});
const keep=document.getElementById('keep');if(keep) keep.addEventListener('click',async()=>{{keep.disabled=true;notice.textContent='';try{{const response=await fetch(config.keepUrl,{{method:'POST',credentials:'same-origin'}});if(!response.ok)throw new Error('Keep failed');document.getElementById('state').textContent='kept';keep.remove();notice.textContent='Kept as a live repository artifact.';}}catch(error){{keep.disabled=false;notice.textContent=error instanceof Error?error.message:String(error);}}}});
}})();</script></body></html>"""
    csp = (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "frame-src 'self'; img-src 'self' data: blob:; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; object-src 'none'"
    )
    return document, csp


def _html_attributes(attrs: list[tuple[str, str | None]]) -> str:
    return "".join(
        f" {html.escape(name, quote=True)}"
        if value is None
        else f' {html.escape(name, quote=True)}="{html.escape(value, quote=True)}"'
        for name, value in attrs
    )


def _is_http_url(value: str) -> bool:
    try:
        return urlsplit(value).scheme.casefold() in {"http", "https"}
    except ValueError:
        return False
