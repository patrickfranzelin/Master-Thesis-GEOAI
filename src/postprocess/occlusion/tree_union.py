from shapely.ops import unary_union


def build_tree_union(grouped, buffer=0.5):
    all_trees = [
        t
        for entry in grouped.values()
        for t in entry["trees"]
    ]

    if not all_trees:
        return None

    raw_union = unary_union(all_trees)
    return raw_union.buffer(buffer)