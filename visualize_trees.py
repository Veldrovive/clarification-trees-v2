from pathlib import Path
from typer import Typer

from clarification_trees.dialog_tree import DialogTree, visualize_tree

app = Typer()

@app.command()
def main(tree_path: Path, output_path: Path | None = None):
    tree = DialogTree.load(tree_path)
    if output_path is None:
        output_path = tree_path.parent / f"{tree_path.stem}"
    visualize_tree(tree, str(output_path), view=False)

if __name__ == "__main__":
    app()
