import dyno.dyno_model as Base
from dyno.dyno_model import Self

class DynoModel(Base.DynoModel):
    # does everything the same as DynoModel but adds a method to print equations with tags


    def equations_with_tags(self: Self) -> list[tuple[int, str, list[str]]]:
        """Return equations paired with their metadata tags.

        Each entry is ``(index, equation_text, tags)`` where ``index`` is 1-based.
        """

        equations = list(getattr(getattr(self, "symbolic", None), "equations", []))
        rows: list[tuple[int, str, list[str]]] = []

        for i, eq in enumerate(equations, start=1):
            eq_text = str(eq)
            tags: list[str] = []

            try:
                from dyno.dynsym.grammar import str_expression

                eq_text = str_expression(eq)
            except Exception:
                pass

            meta = getattr(eq, "meta", None)
            statement_metadata = (
                getattr(meta, "statement_metadata", {}) if meta is not None else {}
            )
            if isinstance(statement_metadata, dict):
                raw_tags = statement_metadata.get("tags", [])
                if isinstance(raw_tags, list):
                    tags = [str(tag) for tag in raw_tags]
                for k, v in statement_metadata.items():
                    if k != "tags":
                        tags.append(f"{k}={v}")

            rows.append((i, eq_text, tags))

        return rows

    def print_equations_with_tags(self: Self) -> None:
        """Print all equations with their tags."""

        for i, eq_text, tags in self.equations_with_tags():
            tags_txt = ", ".join(tags) if len(tags) > 0 else "-"
            print(f"{i:>3}. {eq_text}  [tags: {tags_txt}]")


