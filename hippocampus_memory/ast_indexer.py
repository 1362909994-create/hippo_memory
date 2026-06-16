from __future__ import annotations

import ast


def extract_python_index(content: str) -> tuple[list[str], list[str], list[str]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [], [], []
    symbols: set[str] = set()
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            symbols.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.add(module)
            imports.update(f"{module}.{alias.name}".strip(".") for alias in node.names)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                calls.add(name)
    return sorted(symbols), sorted(imports), sorted(calls)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
