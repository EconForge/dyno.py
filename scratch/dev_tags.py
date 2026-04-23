from dyno.experimental.model import DynoModel
from lark import Tree, Token

txt = """

a <- 0.1 :: "This is a parameter"
b <- 0.1 :: [label="This is a parameter", in=R+]


x[t] = x[t-1] + e[t]  :: "Productivity"

[transition] {

y[t] = y[t-1] + e[t]  :: Production
z[t] = z[t-1] + e[t]  :: Inventory

}

# e[t] <- N(0,1)

e[0] <- 0.0
e[1] <- 0.1
e[2] <- 0.0


@name: Demo
"""
model = DynoModel(txt=txt)
model.print_equations_with_tags()

print("\n--- Normalized metadata by statement type ---")


def statement_kind(node: Tree) -> str:
	data = str(node.data)
	if data in {"equality", "formula"}:
		return "equation"
	if data == "assignment":
		return "assignment"
	if data == "metadata_assignment":
		return "metadata_assignment"
	if data == "metadata_block":
		return "metadata_block"
	return data


def extract_inline_raw(annotated: Tree) -> str | None:
	if len(annotated.children) < 2:
		return None
	inline_meta = annotated.children[1]
	if not isinstance(inline_meta, Tree) or len(inline_meta.children) == 0:
		return None
	first = inline_meta.children[0]
	if isinstance(first, Token):
		return str(first)
	return None


def extract_block_raw(metadata_block: Tree) -> str | None:
	if len(metadata_block.children) == 0:
		return None
	block_meta = metadata_block.children[0]
	if not isinstance(block_meta, Tree) or len(block_meta.children) == 0:
		return None
	first = block_meta.children[0]
	if isinstance(first, Token):
		return str(first)
	return None


def visit_statements(node: Tree, depth: int = 0):
	data = str(node.data)

	if data in {"free_block", "block"}:
		for child in node.children:
			if isinstance(child, Tree):
				yield from visit_statements(child, depth)
		return

	if data == "annotated_statement":
		if len(node.children) == 0:
			return
		core = node.children[0]
		if isinstance(core, Tree):
			yield (depth, core, extract_inline_raw(node))
		return

	if data == "metadata_block":
		yield (depth, node, None)
		if len(node.children) > 1 and isinstance(node.children[1], Tree):
			inner_block = node.children[1]
			yield from visit_statements(inner_block, depth + 1)
		return

	# Fallback for any direct statement-like node
	yield (depth, node, None)


for i, (depth, stmt, inline_raw) in enumerate(visit_statements(model.symbolic.tree), start=1):
	indent = "  " * depth
	kind = statement_kind(stmt)

	if kind == "metadata_block":
		raw = extract_block_raw(stmt)
		print(f"{i:>2}. {indent}kind={kind:<20} raw={raw}")
		continue

	meta = getattr(stmt.meta, "statement_metadata", {})
	if inline_raw is None:
		print(f"{i:>2}. {indent}kind={kind:<20} meta={meta}")
	else:
		print(
			f"{i:>2}. {indent}kind={kind:<20} meta={meta}  raw_inline={inline_raw}"
		)


print("\n--- Equations and attached metadata ---")
for i, eq in enumerate(model.symbolic.equations, start=1):
	meta = getattr(eq.meta, "statement_metadata", {})
	tags = meta.get("tags", [])
	print(i, tags, meta)