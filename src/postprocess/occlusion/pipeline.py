from .dents import detect_tree_occlusions, OcclusionConfig

def apply_tree_occlusion_fix(gdf, tree_union):
    if tree_union is None:
        return gdf

    cfg = OcclusionConfig(
        tree_outer_buffer=0.8,
        tree_inner_buffer=0.3,
        min_edge_length=0.1,  # was 1.0
        min_overlap_ratio=0.05,  # was 0.3
        min_curvature=0.01,  # was 0.05
    )

    def refine(row):
        geom = row.geometry
        print(f"\n{'='*50}")
        print(f"Building ID: {row.building_id}")

        # STEP 1: Basic proximity check
        tree_zone = tree_union.buffer(cfg.tree_outer_buffer)
        near_tree = geom.intersects(tree_zone)
        print(f"Near tree zone: {near_tree}")

        if not near_tree:
            print("SKIPPED - no tree proximity")
            return geom

        # STEP 2: Run occlusion detection
        new_geom, detections = detect_tree_occlusions(geom, tree_union, cfg)

        if detections:
            print(f" FIXED {len(detections)} occlusion(s):")
            for d in detections:
                print(f"  L={d['edge_length']:.2f} curv={d['curvature']:.3f} overlap={d['overlap_ratio']:.2f}")
            print("GEOMETRY CHANGED")
        else:
            print("No occlusions detected")

        return new_geom

    gdf = gdf.copy()
    gdf["geometry"] = gdf.apply(refine, axis=1)
    return gdf
