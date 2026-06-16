from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(slots=True)
class CodeSymbol:
    name: str
    qualified_name: str
    kind: str
    container: str | None
    line: int
    end_line: int | None
    signature: str | None = None
    docstring: str | None = None


@dataclass(slots=True)
class CodeReference:
    symbol: str
    kind: str
    line: int
    column: int
    container: str | None
    context: str | None = None


def extract_python_lsp_index(content: str) -> tuple[list[CodeSymbol], list[CodeReference]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [], []
    visitor = _PythonLspVisitor(content.splitlines())
    visitor.visit(tree)
    return visitor.symbols, visitor.references


class _PythonLspVisitor(ast.NodeVisitor):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.stack: list[str] = []
        self.symbols: list[CodeSymbol] = []
        self.references: list[CodeReference] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol(node.name, "class", node, signature=None)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, kind="function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, kind="async_function")

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name:
            self._add_reference(name, "call", node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._add_reference(node.id, "reference", node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add_reference(node.attr, "reference", node)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        self._add_symbol(node.name, kind, node, signature=_signature(node))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _add_symbol(
        self,
        name: str,
        kind: str,
        node: ast.AST,
        *,
        signature: str | None,
    ) -> None:
        container = ".".join(self.stack) if self.stack else None
        qualified_name = f"{container}.{name}" if container else name
        self.symbols.append(
            CodeSymbol(
                name=name,
                qualified_name=qualified_name,
                kind=kind,
                container=container,
                line=getattr(node, "lineno", 1),
                end_line=getattr(node, "end_lineno", None),
                signature=signature,
                docstring=ast.get_docstring(node) if isinstance(node, _DOC_NODES) else None,
            )
        )

    def _add_reference(self, symbol: str, kind: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", 1)
        context = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else None
        self.references.append(
            CodeReference(
                symbol=symbol,
                kind=kind,
                line=line,
                column=getattr(node, "col_offset", 0),
                container=".".join(self.stack) if self.stack else None,
                context=context,
            )
        )


_DOC_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    args.extend(arg.arg for arg in node.args.posonlyargs)
    args.extend(arg.arg for arg in node.args.args)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(args)})"


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
