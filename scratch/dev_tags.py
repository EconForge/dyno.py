from dyno import DynoModel
from lark import Tree, Token

txt = """

a <- 0.1 :: "This is a parameter"
b <- 0.1 :: [label="This is a parameter", in=R+]


x[t] = x[t-1] + e[t]  :: "Productivity"

[transition] {

	y[t] = y[t-1] + e[t]  :: Production
	z[t] = z[t-1] + e[t]  :: Inventory

}


[arbitrage, plop, discount="beta"] :: {

x[t] = x[t-1] + x[t]  :: "That is my name"
y[t] = y[t-1] + e[t]  :: Production
z[t] = z[t-1] + e[t]  :: Inventory
z[t] = z[t-1] + e[t]  :: [label="Inventory"]

}

e[t] <- N(0,1) :: "Exogenous Shock"

# e[0] <- 0.0
# e[1] <- 0.1
# e[2] <- 0.0


@name: Demo
"""
model = DynoModel(txt=txt)
model.print_equations_with_tags()

print("\n--- Normalized metadata by statement type ---")


## iterate on all statements

print(model.metadata)

for eq, meta in model.symbolic.iter_equations_with_metadata():
	if 'transition' in meta.get('tags', []) :
		print(eq, meta)

exit()







def statement_kind(node: Tree) -> str:
	data = str(node.data)
	if data in {"equality", "bare_formula", "formula"}:
		return "equation"
	if data == "assignment":
		return "assignment"
	if data == "quantified_assignment":
		return "quantified_assignment"
	return data



def visit_statements(node: Tree, depth: int = 0):
	data = str(node.data)

	if data in {"free_block", "block"}:
		for child in node.children:
			if isinstance(child, Tree):
				yield from visit_statements(child, depth)
		return

	if data == "annotated_statement":
		stmt = node.children[0]  # statement_core — always present
		meta_node = node.children[1] if len(node.children) > 1 else None
		yield (depth, stmt, meta_node)
		return

	if data == "annotated_block":
		block_tag = node.children[0]  # block_tag node
		block = node.children[1]
		yield (depth, block, block_tag)
		for child in block.children:
			if isinstance(child, Tree):
				yield from visit_statements(child, depth + 1)
		return

	# model_metadata (@key: value) — skip silently
	if data == "model_metadata":
		return


for i, (depth, stmt, meta_node) in enumerate(visit_statements(model.symbolic.tree), start=1):
	indent = "  " * depth
	kind = statement_kind(stmt)

	meta = getattr(stmt.meta, "statement_metadata", {})
	raw = str(meta_node.children[0]) if meta_node is not None else None
	if raw is None:
		print(f"{i:>2}. {indent}kind={kind:<25} meta={meta}")
	else:
		print(f"{i:>2}. {indent}kind={kind:<25} meta={meta}  raw={raw}")


print("\n--- Equations and attached metadata ---")
for i, eq in enumerate(model.symbolic.equations, start=1):
	meta = getattr(eq.meta, "statement_metadata", {})
	tags = meta.get("tags", [])
	print(i, tags, meta)