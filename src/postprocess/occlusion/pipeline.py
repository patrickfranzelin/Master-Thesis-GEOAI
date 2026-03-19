from .dents import fix_tree_dents


def apply_tree_occlusion_fix(gdf, tree_union):
    if tree_union is None:
        return gdf

    def refine(g):
        g = fix_tree_dents(
            g,
            tree_union,
            max_dent_area=120,
            proximity_dist=4.0,
        )
        return g

    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(refine)
    return gdf