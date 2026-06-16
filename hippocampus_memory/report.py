from __future__ import annotations

from html import escape
from pathlib import Path

from hippocampus_memory.db import Database


def render_memory_browser(db: Database, project: str | None = None) -> str:
    stats = db.stats()
    projects = db.list_projects()
    candidates = db.list_candidates(project=project, status="pending", limit=100)
    conflicts = db.list_conflicts(project=project, limit=100)
    memories = db.list_memories(
        project=project,
        include_archived=False,
        include_private=False,
        include_sensitive=False,
        limit=100,
    )
    title = f"hippocampus-memory - {project}" if project else "hippocampus-memory"
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(memory.id)}</td>"
        f"<td>{escape(memory.memory_type)}</td>"
        f"<td>{escape(memory.project or 'global')}</td>"
        f"<td>{escape(memory.status)}</td>"
        f"<td>{escape(memory.visibility)}</td>"
        f"<td>{escape(memory.content[:240])}</td>"
        "</tr>"
        for memory in memories
    )
    project_items = "\n".join(
        f"<li>{escape(str(item.get('name')))} - {escape(str(item.get('root_path') or ''))}</li>"
        for item in projects
    )
    stat_items = "\n".join(
        f"<li>{escape(key)}: {escape(str(value))}</li>" for key, value in stats.items()
    )
    candidate_rows = "\n".join(
        "<tr>"
        f"<td>{escape(item['id'])}</td>"
        f"<td>{escape(item['memory_type'])}</td>"
        f"<td>{escape(str(item.get('project') or 'global'))}</td>"
        f"<td>{escape(item['content'][:240])}</td>"
        "</tr>"
        for item in candidates
    )
    conflict_rows = "\n".join(
        "<tr>"
        f"<td>{escape(item['id'])}</td>"
        f"<td>{escape(str(item.get('entity') or ''))}</td>"
        f"<td>{escape(item['description'][:240])}</td>"
        f"<td>{escape(str(item.get('resolution') or ''))}</td>"
        "</tr>"
        for item in conflicts
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <h2>Stats</h2>
  <ul>{stat_items}</ul>
  <h2>Projects</h2>
  <ul>{project_items or '<li>No projects yet.</li>'}</ul>
  <h2>Recent Non-Sensitive Memories</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Type</th><th>Project</th><th>Status</th><th>Visibility</th><th>Content</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Pending Candidates</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Type</th><th>Project</th><th>Content</th></tr>
    </thead>
    <tbody>{candidate_rows}</tbody>
  </table>
  <h2>Open Conflicts</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Entity</th><th>Description</th><th>Resolution</th></tr>
    </thead>
    <tbody>{conflict_rows}</tbody>
  </table>
</body>
</html>"""


def write_memory_browser(
    db: Database,
    output: str | Path,
    project: str | None = None,
) -> Path:
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_memory_browser(db, project), encoding="utf-8")
    return path
