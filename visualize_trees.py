from pathlib import Path
from typer import Typer
from hydra import compose, initialize
from clarification_trees.dialog_tree import DialogTree, visualize_tree, TreeSidecar

app = Typer()

@app.command()
def main(tree_path: Path, output_path: Path | None = None):
    if tree_path.is_dir():
        tree_path = tree_path / "tree.json"
        
    assert tree_path.exists(), "Tree path does not exist"
    tree = DialogTree.load(tree_path)
    # Look for a sidecar at tree_path.parent / f"{tree_path.stem}_sidecar.json"
    sidecar_path = tree_path.parent / f"{tree_path.stem}_sidecar.json"
    if sidecar_path.exists():
        # Load the default config
        with initialize(version_base=None, config_path="src/clarification_trees/config"):
            cfg = compose(config_name="config")
        tree_sidecar = TreeSidecar.load(sidecar_path, cfg=cfg)
    else:
        tree_sidecar = None

    if output_path is None:
        output_path = tree_path.parent / f"{tree_path.stem}"
    visualize_tree(tree, tree_sidecar, str(output_path), view=False)

if __name__ == "__main__":
    app()
