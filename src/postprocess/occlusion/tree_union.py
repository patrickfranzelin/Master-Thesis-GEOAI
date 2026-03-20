from shapely.ops import unary_union


def build_tree_union(grouped):
    all_trees = [
        t
        for entry in grouped.values()
        for t in entry["trees"]
    ]

    if not all_trees:
        return None

    return unary_union(all_trees)