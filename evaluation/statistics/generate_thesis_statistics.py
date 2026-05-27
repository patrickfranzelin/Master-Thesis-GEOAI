"""Generate read-only thesis statistics notebooks, tables, and figures.

All database access is read-only and targets the src_google schema.
"""

from __future__ import annotations

import os
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[2]
STAT_DIR = ROOT / "evaluation" / "statistics"
TABLE_DIR = STAT_DIR / "tables"
FIG_DIR = STAT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
PG_CONN = os.environ.get("PG_CONN")
if not PG_CONN:
    raise RuntimeError("PG_CONN is not set. Cannot query src_google.")

engine = create_engine(PG_CONN)

sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
    }
)

PALETTE = {
    "original": "#4C78A8",
    "sam": "#F58518",
    "post": "#54A24B",
    "bad": "#B23A48",
    "ok": "#F2C14E",
    "good": "#5DA271",
    "perfect": "#2E7D32",
    "improved": "#2E7D32",
    "unchanged": "#7A7A7A",
    "degraded": "#B23A48",
}
RATING_ORDER = ["bad", "ok", "good", "perfect"]
ERROR_ORDER = [
    "SHIFTED",
    "SHAPE_MISMATCH",
    "OVERSIMPLIFIED",
    "MISSING_PARTS",
    "EXTRA_PARTS",
]
ERROR_COLORS = {
    "SHIFTED": "#3E6FB6",
    "SHAPE_MISMATCH": "#4E9A61",
    "OVERSIMPLIFIED": "#D89C2B",
    "MISSING_PARTS": "#C7545A",
    "EXTRA_PARTS": "#7A4FA3",
}
INCLUDED_COUNTRIES = ["Liberia", "Mexico", "Mozambique", "Nepal", "Niger"]
INCLUDED_COUNTRY_SQL = "('Liberia', 'Mexico', 'Mozambique', 'Nepal', 'Niger')"

COUNTRY_CASE_B = """
case
  when lower(b.tiff_path) like '%mozambique%' then 'Mozambique'
  when lower(b.tiff_path) like '%mexico%' then 'Mexico'
  when lower(b.tiff_path) like '%nepal2%' then 'Nepal2'
  when lower(b.tiff_path) like '%nepal%' then 'Nepal'
  when lower(b.tiff_path) like '%niger%' then 'Niger'
  when lower(b.tiff_path) like '%bangladesh%' then 'Bangladesh'
  when lower(b.tiff_path) like '%liberia%' then 'Liberia'
  else 'Unknown'
end
"""
SCORE_ORIGINAL = (
    "case e.original when 'bad' then 1 when 'ok' then 2 "
    "when 'good' then 3 when 'perfect' then 4 end"
)
SCORE_POST = (
    "case e.post when 'bad' then 1 when 'ok' then 2 "
    "when 'good' then 3 when 'perfect' then 4 end"
)


def read_sql(sql: str) -> pd.DataFrame:
    """Run a SQL query inside an explicit read-only transaction."""

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            df = pd.read_sql_query(text(sql), conn)
        finally:
            trans.rollback()
    return df


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    headers = [str(c) for c in shown.columns]
    rows = [[str(v) for v in row] for row in shown.to_numpy(dtype=object).tolist()]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |"

    out = [fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out.extend(fmt(row) for row in rows)
    return "\n".join(out) + "\n"


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    csv_path = TABLE_DIR / f"{name}.csv"
    md_path = TABLE_DIR / f"{name}.md"
    tex_path = TABLE_DIR / f"{name}.tex"
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        warnings.warn(f"Could not write locked table file: {csv_path}", stacklevel=2)
    try:
        md_path.write_text(md_table(df), encoding="utf-8")
    except PermissionError:
        warnings.warn(f"Could not write locked table file: {md_path}", stacklevel=2)
    try:
        df.to_latex(tex_path, index=False, escape=True)
    except PermissionError:
        warnings.warn(f"Could not write locked table file: {tex_path}", stacklevel=2)
    except Exception as exc:  # pragma: no cover - optional export
        warnings.warn(f"Could not save LaTeX table {name}: {exc}")
    return df


def save_fig(fig, name: str, tight: bool = True) -> None:
    if tight and not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def build_tables() -> dict[str, pd.DataFrame]:
    table_counts = read_sql(
        f"""
        with selected_buildings as (
          select id from src_google.buildings b where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
        )
        select 'buildings' as table_name, count(*)::int as row_count from selected_buildings
        union all select 'building_mlqa', count(*)::int from src_google.building_mlqa m join selected_buildings b on b.id=m.building_id
        union all select 'detected_house', count(*)::int from src_google.detected_house d join selected_buildings b on b.id=d.building_id
        union all select 'detected_house_regularized', count(*)::int from src_google.detected_house_regularized r join selected_buildings b on b.id=r.building_id
        union all select 'detected_tree', count(*)::int from src_google.detected_tree t join selected_buildings b on b.id=t.building_id
        union all select 'evaluation', count(*)::int from src_google.evaluation e join selected_buildings b on b.id=e.building_id
        order by table_name
        """
    )
    save_table(table_counts, "01_table_counts")

    coverage = read_sql(
        f"""
        with b as (
          select id, {COUNTRY_CASE_B} as country
          from src_google.buildings b
        ), mlqa as (
          select building_id, house_present, full_house_present from src_google.building_mlqa
        ), sam as (select distinct building_id from src_google.detected_house),
        post as (select distinct building_id from src_google.detected_house_regularized),
        ev as (select distinct building_id from src_google.evaluation)
        select b.country,
          count(*)::int as total_buildings,
          count(mlqa.building_id)::int as mlqa_analyzed,
          count(*) filter (where mlqa.house_present is false)::int as no_visible_house_mlqa,
          count(*) filter (where mlqa.house_present is true and mlqa.full_house_present is false)::int as partial_house_mlqa,
          count(*) filter (where mlqa.house_present is true and mlqa.full_house_present is true)::int as full_house_mlqa,
          count(sam.building_id)::int as with_sam,
          count(post.building_id)::int as with_post,
          count(ev.building_id)::int as manually_evaluated
        from b
        left join mlqa on mlqa.building_id=b.id
        left join sam on sam.building_id=b.id
        left join post on post.building_id=b.id
        left join ev on ev.building_id=b.id
        where b.country in {INCLUDED_COUNTRY_SQL}
        group by b.country order by b.country
        """
    )
    coverage["post_rate_of_sam_pct"] = (
        100 * coverage["with_post"] / coverage["with_sam"].replace(0, np.nan)
    ).round(1)
    coverage["manual_eval_rate_of_post_pct"] = (
        100 * coverage["manually_evaluated"] / coverage["with_post"].replace(0, np.nan)
    ).round(1)
    save_table(coverage, "01_country_pipeline_coverage")

    eval_rows = read_sql(
        f"""
        select e.id, e.building_id, e.original,
               e.sam as post_introduced_new_errors,
               e.post, e.tags, e.has_post, e.created_at,
               {COUNTRY_CASE_B} as country,
               {SCORE_ORIGINAL} as original_score,
               {SCORE_POST} as post_score
        from src_google.evaluation e
        join src_google.buildings b on b.id = e.building_id
        where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
        """
    )
    eval_rows["change"] = np.select(
        [
            eval_rows["post_score"] > eval_rows["original_score"],
            eval_rows["post_score"] < eval_rows["original_score"],
        ],
        ["improved", "degraded"],
        default="unchanged",
    )

    dup_summary = pd.DataFrame(
        {
            "metric": [
                "evaluation_rows",
                "distinct_evaluated_buildings",
                "duplicate_rows",
            ],
            "value": [
                len(eval_rows),
                eval_rows["building_id"].nunique(),
                len(eval_rows) - eval_rows["building_id"].nunique(),
            ],
        }
    )
    save_table(dup_summary, "02_evaluation_duplicate_summary")

    rating_dist = pd.concat(
        [
            eval_rows["original"]
            .value_counts()
            .reindex(RATING_ORDER, fill_value=0)
            .rename_axis("rating")
            .reset_index(name="count")
            .assign(stage="Original Google"),
            eval_rows["post"]
            .value_counts()
            .reindex(RATING_ORDER, fill_value=0)
            .rename_axis("rating")
            .reset_index(name="count")
            .assign(stage="Postprocessed"),
        ],
        ignore_index=True,
    )
    rating_dist["percentage"] = rating_dist.groupby("stage")["count"].transform(
        lambda s: (100 * s / s.sum()).round(1)
    )
    save_table(rating_dist, "02_rating_distribution")

    transition = (
        pd.crosstab(eval_rows["original"], eval_rows["post"])
        .reindex(index=RATING_ORDER, columns=RATING_ORDER, fill_value=0)
        .reset_index()
    )
    save_table(transition, "02_original_to_post_transition_matrix")

    improvement_overall = (
        eval_rows.groupby("change")
        .size()
        .reindex(["improved", "unchanged", "degraded"], fill_value=0)
        .rename("count")
        .reset_index()
    )
    improvement_overall["percentage"] = (
        100 * improvement_overall["count"] / improvement_overall["count"].sum()
    ).round(1)
    improvement_overall.loc[len(improvement_overall)] = [
        "avg_original_score",
        round(eval_rows["original_score"].mean(), 2),
        np.nan,
    ]
    improvement_overall.loc[len(improvement_overall)] = [
        "avg_post_score",
        round(eval_rows["post_score"].mean(), 2),
        np.nan,
    ]
    save_table(improvement_overall, "02_overall_improvement")

    country_improvement = (
        eval_rows.pivot_table(
            index="country", columns="change", values="id", aggfunc="count", fill_value=0
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["improved", "unchanged", "degraded"]:
        if col not in country_improvement:
            country_improvement[col] = 0
    country_improvement["n"] = country_improvement[
        ["improved", "unchanged", "degraded"]
    ].sum(axis=1)
    country_improvement["improved_pct"] = (
        100 * country_improvement["improved"] / country_improvement["n"]
    ).round(1)
    country_improvement["degraded_pct"] = (
        100 * country_improvement["degraded"] / country_improvement["n"]
    ).round(1)
    country_scores = (
        eval_rows.groupby("country")
        .agg(avg_original_score=("original_score", "mean"), avg_post_score=("post_score", "mean"))
        .round(2)
        .reset_index()
    )
    country_improvement = country_improvement.merge(country_scores, on="country", how="left")
    save_table(
        country_improvement[
            [
                "country",
                "n",
                "improved",
                "unchanged",
                "degraded",
                "improved_pct",
                "degraded_pct",
                "avg_original_score",
                "avg_post_score",
            ]
        ],
        "02_country_improvement",
    )

    new_errors = (
        eval_rows["post_introduced_new_errors"]
        .value_counts()
        .rename_axis("post_introduced_new_errors")
        .reset_index(name="count")
    )
    new_errors["percentage"] = (100 * new_errors["count"] / new_errors["count"].sum()).round(1)
    save_table(new_errors, "02_postprocessing_new_error_frequency")

    original_errors = read_sql(
        """
        select x.err as error_type, count(*)::int as count
        from src_google.evaluation e
        cross join lateral jsonb_array_elements_text(coalesce(e.tags->'original_errors', '[]'::jsonb)) as x(err)
        group by x.err order by count desc
        """
    )
    post_errors = read_sql(
        """
        select x.err as error_type, count(*)::int as count
        from src_google.evaluation e
        cross join lateral jsonb_array_elements_text(coalesce(e.tags->'post_errors', '[]'::jsonb)) as x(err)
        group by x.err order by count desc
        """
    )
    error_compare = pd.concat(
        [original_errors.assign(stage="Original Google"), post_errors.assign(stage="Postprocessed")],
        ignore_index=True,
    )
    save_table(error_compare[["stage", "error_type", "count"]], "02_error_categories_before_after")

    country_original_errors = read_sql(
        f"""
        select {COUNTRY_CASE_B} as country, x.err as error_type, count(*)::int as count
        from src_google.evaluation e
        join src_google.buildings b on b.id=e.building_id
        cross join lateral jsonb_array_elements_text(coalesce(e.tags->'original_errors', '[]'::jsonb)) as x(err)
        where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
        group by country, x.err order by country, count desc
        """
    )
    country_post_errors = read_sql(
        f"""
        select {COUNTRY_CASE_B} as country, x.err as error_type, count(*)::int as count
        from src_google.evaluation e
        join src_google.buildings b on b.id=e.building_id
        cross join lateral jsonb_array_elements_text(coalesce(e.tags->'post_errors', '[]'::jsonb)) as x(err)
        where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
        group by country, x.err order by country, count desc
        """
    )
    save_table(country_original_errors, "02_country_original_errors")
    save_table(country_post_errors, "02_country_post_errors")
    eval_case_counts = (
        eval_rows.groupby("country")
        .size()
        .rename("evaluated_cases")
        .reset_index()
    )
    country_original_error_cases = read_sql(
        f"""
        select {COUNTRY_CASE_B} as country, x.err as error_type, count(distinct e.id)::int as count
        from src_google.evaluation e
        join src_google.buildings b on b.id=e.building_id
        cross join lateral jsonb_array_elements_text(coalesce(e.tags->'original_errors', '[]'::jsonb)) as x(err)
        where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
        group by country, x.err order by country, count desc
        """
    )
    country_post_error_cases = read_sql(
        f"""
        select {COUNTRY_CASE_B} as country, x.err as error_type, count(distinct e.id)::int as count
        from src_google.evaluation e
        join src_google.buildings b on b.id=e.building_id
        cross join lateral jsonb_array_elements_text(coalesce(e.tags->'post_errors', '[]'::jsonb)) as x(err)
        where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
        group by country, x.err order by country, count desc
        """
    )
    country_error_profile = pd.concat(
        [
            country_original_error_cases.assign(stage="Original Google"),
            country_post_error_cases.assign(stage="Postprocessed"),
        ],
        ignore_index=True,
    )
    total_error_profile = (
        country_error_profile.groupby(["stage", "error_type"], as_index=False)["count"]
        .sum()
        .assign(country="Total")
    )
    country_error_profile = pd.concat(
        [country_error_profile, total_error_profile],
        ignore_index=True,
    )
    full_error_index = pd.MultiIndex.from_product(
        [
            INCLUDED_COUNTRIES + ["Total"],
            ["Original Google", "Postprocessed"],
            ERROR_ORDER,
        ],
        names=["country", "stage", "error_type"],
    )
    country_error_profile = (
        country_error_profile.set_index(["country", "stage", "error_type"])
        .reindex(full_error_index, fill_value=0)
        .reset_index()
    )
    case_counts = pd.concat(
        [
            eval_case_counts,
            pd.DataFrame({"country": ["Total"], "evaluated_cases": [int(eval_rows.shape[0])]})
        ],
        ignore_index=True,
    )
    country_error_profile = country_error_profile.merge(case_counts, on="country", how="left")
    country_error_profile["share_pct"] = np.where(
        country_error_profile["evaluated_cases"] > 0,
        100 * country_error_profile["count"] / country_error_profile["evaluated_cases"],
        0,
    ).round(1)
    country_error_profile["display_location"] = pd.Categorical(
        country_error_profile["country"],
        categories=INCLUDED_COUNTRIES + ["Total"],
        ordered=True,
    )
    country_error_profile["error_type"] = pd.Categorical(
        country_error_profile["error_type"], categories=ERROR_ORDER, ordered=True
    )
    country_error_profile = country_error_profile.sort_values(
        ["display_location", "stage", "error_type"]
    ).drop(columns=["display_location"])
    save_table(country_error_profile, "02_error_categories_by_location_total")

    geom_eval = read_sql(
        f"""
        with ev_latest as (
          select distinct on (e.building_id)
            e.id as evaluation_id, e.building_id, e.original, e.post, e.sam as post_introduced_new_errors,
            {SCORE_ORIGINAL} as original_score, {SCORE_POST} as post_score,
            case when {SCORE_POST} > {SCORE_ORIGINAL} then 'improved'
                 when {SCORE_POST} < {SCORE_ORIGINAL} then 'degraded'
                 else 'unchanged' end as change,
            e.created_at
          from src_google.evaluation e
          join src_google.buildings b on b.id = e.building_id
          where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
          order by e.building_id, e.created_at desc, e.id desc
        ), sam as (
          select building_id, ST_UnaryUnion(ST_Collect(ST_MakeValid(geom))) geom
          from src_google.detected_house group by building_id
        ), post as (
          select building_id, ST_UnaryUnion(ST_Collect(ST_MakeValid(geom))) geom
          from src_google.detected_house_regularized group by building_id
        ), mlqa as (
          select distinct on (building_id)
            building_id, house_present, full_house_present
          from src_google.building_mlqa
          order by building_id, analyzed_at desc
        ), trees as (
          select
            building_id,
            count(*)::int as tree_count,
            sum(ST_Area(ST_MakeValid(geom)::geography)) as tree_area_m2
          from src_google.detected_tree
          group by building_id
        )
        select ev.building_id,
          {COUNTRY_CASE_B} as country,
          b.confidence as google_confidence,
          ev.original, ev.post, ev.change, ev.original_score, ev.post_score,
          ev.post_introduced_new_errors,
          mlqa.house_present, mlqa.full_house_present,
          exists (
            select 1
            from jsonb_array_elements_text(coalesce(e.tags->'original_errors', '[]'::jsonb)) as x(err)
            where x.err = 'SHIFTED'
          ) as original_shifted,
          exists (
            select 1
            from jsonb_array_elements_text(coalesce(e.tags->'post_errors', '[]'::jsonb)) as x(err)
            where x.err = 'MISSING_PARTS'
          ) as post_missing_parts,
          exists (
            select 1
            from jsonb_array_elements_text(coalesce(e.tags->'post_errors', '[]'::jsonb)) as x(err)
            where x.err = 'EXTRA_PARTS'
          ) as post_extra_parts,
          ST_Area(ST_MakeValid(b.geom)::geography) as original_area_m2,
          ST_Area(s.geom::geography) as sam_area_m2,
          ST_Area(p.geom::geography) as post_area_m2,
          ST_NPoints(ST_MakeValid(b.geom)) as original_vertices,
          ST_NPoints(s.geom) as sam_vertices,
          ST_NPoints(p.geom) as post_vertices,
          ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(s.geom)::geography) as orig_sam_shift_m,
          ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(p.geom)::geography) as orig_post_shift_m,
          ST_Distance(ST_Centroid(s.geom)::geography, ST_Centroid(p.geom)::geography) as sam_post_shift_m,
          coalesce(trees.tree_count, 0) as tree_count,
          coalesce(trees.tree_area_m2, 0) as tree_area_m2
          from ev_latest ev
        join src_google.evaluation e on e.id=ev.evaluation_id
        join src_google.buildings b on b.id=ev.building_id
        left join mlqa on mlqa.building_id=ev.building_id
        left join sam s on s.building_id=ev.building_id
        left join post p on p.building_id=ev.building_id
        left join trees on trees.building_id=ev.building_id
        """
    )
    geom_eval["sam_original_area_ratio"] = geom_eval["sam_area_m2"] / geom_eval[
        "original_area_m2"
    ].replace(0, np.nan)
    geom_eval["post_original_area_ratio"] = geom_eval["post_area_m2"] / geom_eval[
        "original_area_m2"
    ].replace(0, np.nan)
    geom_eval["post_sam_area_ratio"] = geom_eval["post_area_m2"] / geom_eval[
        "sam_area_m2"
    ].replace(0, np.nan)
    geom_eval["sam_vertex_delta"] = geom_eval["sam_vertices"] - geom_eval["original_vertices"]
    geom_eval["post_vertex_delta"] = geom_eval["post_vertices"] - geom_eval["original_vertices"]
    geom_eval["post_sam_vertex_delta"] = geom_eval["post_vertices"] - geom_eval["sam_vertices"]
    geom_eval["post_area_bias_pct"] = 100 * (geom_eval["post_original_area_ratio"] - 1)
    geom_eval["post_abs_area_error_pct"] = geom_eval["post_area_bias_pct"].abs()
    geom_eval["sam_area_bias_pct"] = 100 * (geom_eval["sam_original_area_ratio"] - 1)
    geom_eval["sam_abs_area_error_pct"] = geom_eval["sam_area_bias_pct"].abs()
    geom_eval["post_shift_delta_vs_sam_m"] = (
        geom_eval["orig_post_shift_m"] - geom_eval["orig_sam_shift_m"]
    )
    geom_eval["post_closer_than_sam"] = (
        geom_eval["orig_post_shift_m"] < geom_eval["orig_sam_shift_m"]
    )
    geom_eval["tree_area_original_ratio"] = geom_eval["tree_area_m2"] / geom_eval[
        "original_area_m2"
    ].replace(0, np.nan)

    area_error_bins = [
        -np.inf,
        -50,
        -25,
        -10,
        10,
        25,
        50,
        np.inf,
    ]
    area_error_labels = [
        "more than 50% smaller",
        "25-50% smaller",
        "10-25% smaller",
        "within +/-10%",
        "10-25% larger",
        "25-50% larger",
        "more than 50% larger",
    ]
    geom_eval["post_area_agreement"] = pd.cut(
        geom_eval["post_area_bias_pct"],
        bins=area_error_bins,
        labels=area_error_labels,
    )
    geom_eval["mlqa_visibility"] = np.select(
        [
            geom_eval["house_present"].isna(),
            geom_eval["house_present"].eq(False),
            geom_eval["house_present"].eq(True)
            & geom_eval["full_house_present"].eq(False),
            geom_eval["house_present"].eq(True)
            & geom_eval["full_house_present"].eq(True),
        ],
        ["not analyzed", "no visible house", "partial house", "full house"],
        default="not analyzed",
    )
    geom_eval["tree_context"] = np.select(
        [
            geom_eval["tree_count"].eq(0),
            geom_eval["tree_area_original_ratio"].le(0.25),
            geom_eval["tree_area_original_ratio"].le(1.0),
        ],
        ["no detected tree", "low tree context", "medium tree context"],
        default="high tree context",
    )
    geom_eval["tree_detected"] = np.where(
        geom_eval["tree_count"].gt(0), "tree detected", "no detected tree"
    )
    geom_eval["post_new_error_flag"] = (
        geom_eval["post_introduced_new_errors"]
        .fillna("no")
        .astype(str)
        .str.lower()
        .eq("yes")
    )
    geom_eval["google_confidence_pct"] = np.where(
        geom_eval["google_confidence"].le(1),
        geom_eval["google_confidence"] * 100,
        geom_eval["google_confidence"],
    )
    geom_eval.to_csv(TABLE_DIR / "03_geometry_eval_building_level_raw.csv", index=False)

    geometry_summary = pd.DataFrame(
        {
            "geometry_stage": ["Original Google", "SAM", "Postprocessed"],
            "n_with_geometry": [
                geom_eval["original_area_m2"].notna().sum(),
                geom_eval["sam_area_m2"].notna().sum(),
                geom_eval["post_area_m2"].notna().sum(),
            ],
            "mean_area_m2": [
                geom_eval["original_area_m2"].mean(),
                geom_eval["sam_area_m2"].mean(),
                geom_eval["post_area_m2"].mean(),
            ],
            "median_area_m2": [
                geom_eval["original_area_m2"].median(),
                geom_eval["sam_area_m2"].median(),
                geom_eval["post_area_m2"].median(),
            ],
            "mean_vertices": [
                geom_eval["original_vertices"].mean(),
                geom_eval["sam_vertices"].mean(),
                geom_eval["post_vertices"].mean(),
            ],
            "median_vertices": [
                geom_eval["original_vertices"].median(),
                geom_eval["sam_vertices"].median(),
                geom_eval["post_vertices"].median(),
            ],
        }
    ).round(2)
    save_table(geometry_summary, "03_geometry_stage_summary")

    geometry_country = (
        geom_eval.groupby("country")
        .agg(
            n=("building_id", "count"),
            avg_original_vertices=("original_vertices", "mean"),
            avg_sam_vertices=("sam_vertices", "mean"),
            avg_post_vertices=("post_vertices", "mean"),
            avg_orig_sam_shift_m=("orig_sam_shift_m", "mean"),
            avg_orig_post_shift_m=("orig_post_shift_m", "mean"),
            avg_sam_post_shift_m=("sam_post_shift_m", "mean"),
            avg_sam_original_area_ratio=("sam_original_area_ratio", "mean"),
            avg_post_original_area_ratio=("post_original_area_ratio", "mean"),
        )
        .round(2)
        .reset_index()
    )
    save_table(geometry_country, "03_country_geometry_summary")

    post_area_agreement = (
        geom_eval.dropna(subset=["post_area_agreement"])
        .groupby(["country", "post_area_agreement"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    country_totals = post_area_agreement.groupby("country")["n"].transform("sum")
    post_area_agreement["share_pct"] = (
        100 * post_area_agreement["n"] / country_totals.replace(0, np.nan)
    ).round(1)
    save_table(post_area_agreement, "03_post_area_agreement_by_country")

    geometry_result_factors = (
        geom_eval.assign(
            post_good_or_perfect=geom_eval["post"].isin(["good", "perfect"]),
            original_shifted=geom_eval["original_shifted"].fillna(False),
            post_closer_than_sam=geom_eval["post_closer_than_sam"].fillna(False),
        )
        .groupby("country")
        .agg(
            n=("building_id", "count"),
            good_or_perfect_pct=("post_good_or_perfect", lambda s: 100 * s.mean()),
            median_abs_area_error_pct=("post_abs_area_error_pct", "median"),
            median_post_shift_m=("orig_post_shift_m", "median"),
            shifted_originals=("original_shifted", "sum"),
            shifted_originals_closer_than_sam_pct=(
                "post_closer_than_sam",
                lambda s: 100 * s[geom_eval.loc[s.index, "original_shifted"].fillna(False)].mean()
                if geom_eval.loc[s.index, "original_shifted"].fillna(False).any()
                else np.nan,
            ),
            post_missing_parts_pct=("post_missing_parts", lambda s: 100 * s.fillna(False).mean()),
            post_extra_parts_pct=("post_extra_parts", lambda s: 100 * s.fillna(False).mean()),
            medium_high_tree_context_pct=(
                "tree_context",
                lambda s: 100 * s.isin(["medium tree context", "high tree context"]).mean(),
            ),
        )
        .round(1)
        .reset_index()
    )
    save_table(geometry_result_factors, "03_geometry_result_factors_by_country")

    geometry_by_quality = (
        geom_eval.assign(post_good_or_perfect=geom_eval["post"].isin(["good", "perfect"]))
        .groupby("post_good_or_perfect")
        .agg(
            n=("building_id", "count"),
            avg_sam_original_area_ratio=("sam_original_area_ratio", "mean"),
            avg_post_original_area_ratio=("post_original_area_ratio", "mean"),
            avg_sam_vertex_delta=("sam_vertex_delta", "mean"),
            avg_post_vertex_delta=("post_vertex_delta", "mean"),
            avg_post_sam_vertex_delta=("post_sam_vertex_delta", "mean"),
        )
        .round(2)
        .reset_index()
    )
    save_table(geometry_by_quality, "03_geometry_by_post_quality")

    geometry_by_visibility = (
        geom_eval.assign(
            post_good_or_perfect=geom_eval["post"].isin(["good", "perfect"]),
        )
        .groupby("mlqa_visibility")
        .agg(
            n=("building_id", "count"),
            good_or_perfect_pct=("post_good_or_perfect", lambda s: 100 * s.mean()),
            median_abs_area_error_pct=("post_abs_area_error_pct", "median"),
            missing_parts_pct=("post_missing_parts", lambda s: 100 * s.fillna(False).mean()),
            extra_parts_pct=("post_extra_parts", lambda s: 100 * s.fillna(False).mean()),
        )
        .round(1)
        .reset_index()
    )
    save_table(geometry_by_visibility, "03_geometry_by_mlqa_visibility")

    geometry_by_tree_context = (
        geom_eval.assign(post_good_or_perfect=geom_eval["post"].isin(["good", "perfect"]))
        .groupby("tree_context")
        .agg(
            n=("building_id", "count"),
            good_or_perfect_pct=("post_good_or_perfect", lambda s: 100 * s.mean()),
            median_abs_area_error_pct=("post_abs_area_error_pct", "median"),
            missing_parts_pct=("post_missing_parts", lambda s: 100 * s.fillna(False).mean()),
            extra_parts_pct=("post_extra_parts", lambda s: 100 * s.fillna(False).mean()),
        )
        .round(1)
        .reset_index()
    )
    save_table(geometry_by_tree_context, "03_geometry_by_tree_context")

    confidence_eval = geom_eval.dropna(subset=["google_confidence_pct"]).copy()
    confidence_eval["post_good_or_perfect"] = confidence_eval["post"].isin(["good", "perfect"])
    confidence_eval["improved_flag"] = confidence_eval["change"].eq("improved")
    confidence_eval["degraded_flag"] = confidence_eval["change"].eq("degraded")
    confidence_eval["score_delta"] = (
        confidence_eval["post_score"] - confidence_eval["original_score"]
    )
    confidence_eval["confidence_quartile"] = pd.qcut(
        confidence_eval["google_confidence_pct"],
        q=4,
        labels=[
            "Q1 lowest confidence",
            "Q2 lower-middle",
            "Q3 upper-middle",
            "Q4 highest confidence",
        ],
        duplicates="drop",
    )

    confidence_summary_country = (
        confidence_eval.groupby("country")
        .agg(
            n=("building_id", "count"),
            mean_confidence_pct=("google_confidence_pct", "mean"),
            median_confidence_pct=("google_confidence_pct", "median"),
            min_confidence_pct=("google_confidence_pct", "min"),
            max_confidence_pct=("google_confidence_pct", "max"),
            good_or_perfect_pct=("post_good_or_perfect", lambda s: 100 * s.mean()),
            degraded_pct=("degraded_flag", lambda s: 100 * s.mean()),
            post_new_error_pct=("post_new_error_flag", lambda s: 100 * s.mean()),
        )
        .round(2)
        .reset_index()
    )
    save_table(confidence_summary_country, "03_google_confidence_by_country")

    confidence_by_quartile = (
        confidence_eval.groupby("confidence_quartile", observed=False)
        .agg(
            n=("building_id", "count"),
            confidence_min_pct=("google_confidence_pct", "min"),
            confidence_median_pct=("google_confidence_pct", "median"),
            confidence_max_pct=("google_confidence_pct", "max"),
            good_or_perfect_pct=("post_good_or_perfect", lambda s: 100 * s.mean()),
            improved_pct=("improved_flag", lambda s: 100 * s.mean()),
            degraded_pct=("degraded_flag", lambda s: 100 * s.mean()),
            post_new_error_pct=("post_new_error_flag", lambda s: 100 * s.mean()),
            median_abs_area_error_pct=("post_abs_area_error_pct", "median"),
            median_post_shift_m=("orig_post_shift_m", "median"),
            missing_parts_pct=("post_missing_parts", lambda s: 100 * s.fillna(False).mean()),
            extra_parts_pct=("post_extra_parts", lambda s: 100 * s.fillna(False).mean()),
        )
        .round(2)
        .reset_index()
    )
    save_table(confidence_by_quartile, "03_google_confidence_by_quartile")

    def correlation_row(metric: str, column: str) -> dict[str, float | str | int]:
        subset = confidence_eval[["google_confidence_pct", "country", column]].dropna().copy()
        subset[column] = subset[column].astype(float)
        if len(subset) < 3:
            return {
                "metric": metric,
                "n": len(subset),
                "pearson": np.nan,
                "spearman": np.nan,
                "within_country_pearson": np.nan,
            }
        x = subset["google_confidence_pct"]
        y = subset[column]
        x_centered = x - subset.groupby("country")["google_confidence_pct"].transform("mean")
        y_centered = y - subset.groupby("country")[column].transform("mean")
        return {
            "metric": metric,
            "n": len(subset),
            "pearson": x.corr(y),
            "spearman": x.rank().corr(y.rank()),
            "within_country_pearson": x_centered.corr(y_centered),
        }

    confidence_correlations = pd.DataFrame(
        [
            correlation_row("original_manual_score", "original_score"),
            correlation_row("postprocessed_manual_score", "post_score"),
            correlation_row("score_delta_post_minus_original", "score_delta"),
            correlation_row("post_good_or_perfect_flag", "post_good_or_perfect"),
            correlation_row("improved_flag", "improved_flag"),
            correlation_row("degraded_flag", "degraded_flag"),
            correlation_row("post_introduced_new_error_flag", "post_new_error_flag"),
            correlation_row("post_abs_area_error_pct", "post_abs_area_error_pct"),
            correlation_row("orig_post_shift_m", "orig_post_shift_m"),
            correlation_row("post_missing_parts_flag", "post_missing_parts"),
            correlation_row("post_extra_parts_flag", "post_extra_parts"),
        ]
    ).round(3)
    save_table(confidence_correlations, "03_google_confidence_correlations")

    new_errors_by_tree_detection = (
        geom_eval.groupby("tree_detected")
        .agg(
            n=("building_id", "count"),
            post_new_errors=("post_new_error_flag", "sum"),
            post_new_error_pct=("post_new_error_flag", lambda s: 100 * s.mean()),
            good_or_perfect_pct=("post", lambda s: 100 * s.isin(["good", "perfect"]).mean()),
            median_abs_area_change_pct=("post_abs_area_error_pct", "median"),
            missing_parts_pct=("post_missing_parts", lambda s: 100 * s.fillna(False).mean()),
            extra_parts_pct=("post_extra_parts", lambda s: 100 * s.fillna(False).mean()),
        )
        .round(1)
        .reset_index()
    )
    save_table(new_errors_by_tree_detection, "03_new_errors_by_tree_detection")

    new_errors_by_tree_context = (
        geom_eval.groupby("tree_context")
        .agg(
            n=("building_id", "count"),
            post_new_errors=("post_new_error_flag", "sum"),
            post_new_error_pct=("post_new_error_flag", lambda s: 100 * s.mean()),
            good_or_perfect_pct=("post", lambda s: 100 * s.isin(["good", "perfect"]).mean()),
            median_abs_area_change_pct=("post_abs_area_error_pct", "median"),
            missing_parts_pct=("post_missing_parts", lambda s: 100 * s.fillna(False).mean()),
            extra_parts_pct=("post_extra_parts", lambda s: 100 * s.fillna(False).mean()),
        )
        .round(1)
        .reset_index()
    )
    save_table(new_errors_by_tree_context, "03_new_errors_by_tree_context")

    tree_yes = geom_eval["tree_count"].gt(0)
    error_yes = geom_eval["post_new_error_flag"]
    a = int((tree_yes & error_yes).sum())
    b = int((tree_yes & ~error_yes).sum())
    c = int((~tree_yes & error_yes).sum())
    d = int((~tree_yes & ~error_yes).sum())
    risk_tree = a / (a + b) if (a + b) else np.nan
    risk_no_tree = c / (c + d) if (c + d) else np.nan
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = ((a * d - b * c) / denom) if denom else np.nan
    new_error_tree_association = pd.DataFrame(
        [
            {"metric": "tree_detected_and_new_error", "value": a},
            {"metric": "tree_detected_no_new_error", "value": b},
            {"metric": "no_tree_detected_and_new_error", "value": c},
            {"metric": "no_tree_detected_no_new_error", "value": d},
            {"metric": "new_error_rate_with_tree_pct", "value": 100 * risk_tree},
            {"metric": "new_error_rate_without_tree_pct", "value": 100 * risk_no_tree},
            {"metric": "risk_difference_percentage_points", "value": 100 * (risk_tree - risk_no_tree)},
            {"metric": "risk_ratio_tree_vs_no_tree", "value": risk_tree / risk_no_tree if risk_no_tree else np.nan},
            {"metric": "phi_correlation_tree_detected_new_error", "value": phi},
        ]
    )
    save_table(new_error_tree_association.round(3), "03_new_error_tree_association")

    shift_eval = read_sql(
        f"""
        with ev_latest as (
          select distinct on (e.building_id) e.building_id, e.post
          from src_google.evaluation e
          join src_google.buildings b on b.id = e.building_id
          where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
            and exists (
              select 1
              from jsonb_array_elements_text(coalesce(e.tags->'original_errors', '[]'::jsonb)) as x(err)
              where x.err = 'SHIFTED'
            )
          order by e.building_id, e.created_at desc, e.id desc
        ), sam as (
          select building_id, ST_UnaryUnion(ST_Collect(ST_MakeValid(geom))) geom
          from src_google.detected_house group by building_id
        ), post as (
          select building_id, ST_UnaryUnion(ST_Collect(ST_MakeValid(geom))) geom
          from src_google.detected_house_regularized group by building_id
        ), vectors as (
          select ev.building_id, ev.post, {COUNTRY_CASE_B} as country,
            ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(s.geom)::geography) sam_dist,
            ST_Azimuth(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(s.geom)::geography) sam_az,
            ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(p.geom)::geography) post_dist,
            ST_Azimuth(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(p.geom)::geography) post_az
          from ev_latest ev
          join src_google.buildings b on b.id=ev.building_id
          left join sam s on s.building_id=ev.building_id
          left join post p on p.building_id=ev.building_id
        )
        select building_id, country, post, sam_dist, sam_az,
               sam_dist * sin(sam_az) as sam_dx_m,
               sam_dist * cos(sam_az) as sam_dy_m,
               post_dist, post_az,
               post_dist * sin(post_az) as post_dx_m,
               post_dist * cos(post_az) as post_dy_m
        from vectors
        """
    )
    shift_eval.to_csv(TABLE_DIR / "04_shift_vectors_raw.csv", index=False)
    shift_eval_good_post = shift_eval[shift_eval["post"].isin(["good", "perfect"])].copy()
    shift_eval_good_post.to_csv(
        TABLE_DIR / "04_shift_vectors_good_perfect_post_raw.csv",
        index=False,
    )
    shift_country = (
        shift_eval.groupby("country")
        .agg(
            n=("building_id", "count"),
            avg_sam_shift_m=("sam_dist", "mean"),
            median_sam_shift_m=("sam_dist", "median"),
            avg_sam_dx_m=("sam_dx_m", "mean"),
            avg_sam_dy_m=("sam_dy_m", "mean"),
            sd_sam_dx_m=("sam_dx_m", "std"),
            sd_sam_dy_m=("sam_dy_m", "std"),
            avg_post_shift_m=("post_dist", "mean"),
            median_post_shift_m=("post_dist", "median"),
            avg_post_dx_m=("post_dx_m", "mean"),
            avg_post_dy_m=("post_dy_m", "mean"),
            sd_post_dx_m=("post_dx_m", "std"),
            sd_post_dy_m=("post_dy_m", "std"),
        )
        .round(2)
        .reset_index()
    )
    save_table(shift_country, "04_country_shift_vectors")

    def _circular_mean_deg(series: pd.Series) -> float:
        values = series.dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return np.nan
        sin_mean = np.sin(values).mean()
        cos_mean = np.cos(values).mean()
        return float((np.degrees(np.arctan2(sin_mean, cos_mean)) + 360) % 360)

    def _directional_concentration(series: pd.Series) -> float:
        values = series.dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return np.nan
        return float(np.hypot(np.sin(values).mean(), np.cos(values).mean()))

    shift_stats = (
        shift_eval.groupby("country")
        .agg(
            n=("building_id", "count"),
            mean_shift_m=("post_dist", "mean"),
            median_shift_m=("post_dist", "median"),
            p75_shift_m=("post_dist", lambda s: s.quantile(0.75)),
            p90_shift_m=("post_dist", lambda s: s.quantile(0.90)),
            max_shift_m=("post_dist", "max"),
            share_le_1m_pct=("post_dist", lambda s: 100 * (s <= 1).mean()),
            share_le_2m_pct=("post_dist", lambda s: 100 * (s <= 2).mean()),
            share_le_5m_pct=("post_dist", lambda s: 100 * (s <= 5).mean()),
            mean_dx_m=("post_dx_m", "mean"),
            mean_dy_m=("post_dy_m", "mean"),
            sd_dx_m=("post_dx_m", "std"),
            sd_dy_m=("post_dy_m", "std"),
            mean_direction_deg=("post_az", _circular_mean_deg),
            directional_concentration=("post_az", _directional_concentration),
        )
        .reset_index()
    )
    shift_stats["resultant_shift_m"] = np.hypot(
        shift_stats["mean_dx_m"], shift_stats["mean_dy_m"]
    )
    shift_stats["resultant_direction_deg"] = (
        np.degrees(np.arctan2(shift_stats["mean_dx_m"], shift_stats["mean_dy_m"])) + 360
    ) % 360
    shift_stats = shift_stats[
        [
            "country",
            "n",
            "mean_shift_m",
            "median_shift_m",
            "p75_shift_m",
            "p90_shift_m",
            "max_shift_m",
            "share_le_1m_pct",
            "share_le_2m_pct",
            "share_le_5m_pct",
            "mean_dx_m",
            "mean_dy_m",
            "resultant_shift_m",
            "resultant_direction_deg",
            "mean_direction_deg",
            "directional_concentration",
            "sd_dx_m",
            "sd_dy_m",
        ]
    ].round(2)
    save_table(shift_stats, "04_shift_summary_for_discussion")

    mlqa_error_counts = read_sql(
        """
        select x.err as mlqa_error_type, count(*)::int as count
        from src_google.building_mlqa m
        cross join lateral jsonb_array_elements_text(coalesce(m.errors, '[]'::jsonb)) as x(err)
        group by x.err order by count desc
        """
    )
    save_table(mlqa_error_counts, "05_mlqa_error_counts")

    semantic_error_flags = read_sql(
        f"""
        with categories(category) as (
          values
            ('SHIFTED'),
            ('SHAPE_MISMATCH'),
            ('OVERSIMPLIFIED'),
            ('MISSING_PARTS'),
            ('EXTRA_PARTS')
        ), ev_latest as (
          select distinct on (e.building_id)
            e.building_id,
            {COUNTRY_CASE_B} as country,
            coalesce(e.tags->'original_errors', '[]'::jsonb) as manual_errors
          from src_google.evaluation e
          join src_google.buildings b on b.id = e.building_id
          where {COUNTRY_CASE_B} in {INCLUDED_COUNTRY_SQL}
          order by e.building_id, e.created_at desc, e.id desc
        ), mlqa_latest as (
          select distinct on (m.building_id)
            m.building_id,
            coalesce(m.errors, '[]'::jsonb) as mlqa_errors
          from src_google.building_mlqa m
          order by m.building_id, m.analyzed_at desc
        ), paired as (
          select
            ev.building_id,
            ev.country,
            ev.manual_errors,
            mlqa.mlqa_errors
          from ev_latest ev
          join mlqa_latest mlqa on mlqa.building_id = ev.building_id
        )
        select
          p.building_id,
          p.country,
          c.category,
          exists (
            select 1
            from jsonb_array_elements_text(p.manual_errors) as x(err)
            where x.err = c.category
          ) as manual_present,
          exists (
            select 1
            from jsonb_array_elements_text(p.mlqa_errors) as x(err)
            where case x.err
              when 'MISALIGNED' then 'SHIFTED'
              when 'ORIENTATION_MISMATCH' then 'SHAPE_MISMATCH'
              else x.err
            end = c.category
          ) as mlqa_present
        from paired p
        cross join categories c
        order by p.country, p.building_id, c.category
        """
    )
    save_table(semantic_error_flags, "05_semantic_error_building_level_flags")

    if semantic_error_flags.empty:
        semantic_error_agreement = pd.DataFrame()
        semantic_error_frequency = pd.DataFrame()
        semantic_error_case_summary = pd.DataFrame()
    else:
        semantic_error_flags["manual_present"] = semantic_error_flags["manual_present"].astype(bool)
        semantic_error_flags["mlqa_present"] = semantic_error_flags["mlqa_present"].astype(bool)
        paired_case_count = semantic_error_flags["building_id"].nunique()

        agreement_rows = []
        for category, group in semantic_error_flags.groupby("category", sort=False):
            manual = group["manual_present"]
            mlqa = group["mlqa_present"]
            tp = int((manual & mlqa).sum())
            fp = int((~manual & mlqa).sum())
            fn = int((manual & ~mlqa).sum())
            tn = int((~manual & ~mlqa).sum())
            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
            agreement_rows.append(
                {
                    "category": category,
                    "paired_cases": int(group["building_id"].nunique()),
                    "manual_count": int(manual.sum()),
                    "mlqa_count": int(mlqa.sum()),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_negative": tn,
                    "precision": round(precision, 3) if not pd.isna(precision) else np.nan,
                    "recall": round(recall, 3) if not pd.isna(recall) else np.nan,
                    "f1": round(f1, 3) if not pd.isna(f1) else np.nan,
                    "accuracy": round((tp + tn) / len(group), 3) if len(group) else np.nan,
                    "mlqa_minus_manual_count": int(mlqa.sum() - manual.sum()),
                    "mlqa_manual_count_ratio": round(mlqa.sum() / manual.sum(), 2) if manual.sum() else np.nan,
                }
            )
        semantic_error_agreement = pd.DataFrame(agreement_rows)
        semantic_error_agreement["category"] = pd.Categorical(
            semantic_error_agreement["category"], categories=ERROR_ORDER, ordered=True
        )
        semantic_error_agreement = semantic_error_agreement.sort_values("category")
        save_table(semantic_error_agreement, "05_semantic_error_category_agreement")

        frequency_rows = []
        total_manual_labels = int(semantic_error_flags["manual_present"].sum())
        total_mlqa_labels = int(semantic_error_flags["mlqa_present"].sum())
        for category, group in semantic_error_flags.groupby("category", sort=False):
            manual_count = int(group["manual_present"].sum())
            mlqa_count = int(group["mlqa_present"].sum())
            frequency_rows.extend(
                [
                    {
                        "source": "Manual original errors",
                        "category": category,
                        "count": manual_count,
                        "share_of_cases_pct": round(100 * manual_count / paired_case_count, 1),
                        "share_of_all_labels_pct": round(100 * manual_count / total_manual_labels, 1)
                        if total_manual_labels
                        else np.nan,
                    },
                    {
                        "source": "MLQA normalized errors",
                        "category": category,
                        "count": mlqa_count,
                        "share_of_cases_pct": round(100 * mlqa_count / paired_case_count, 1),
                        "share_of_all_labels_pct": round(100 * mlqa_count / total_mlqa_labels, 1)
                        if total_mlqa_labels
                        else np.nan,
                    },
                ]
            )
        semantic_error_frequency = pd.DataFrame(frequency_rows)
        semantic_error_frequency["category"] = pd.Categorical(
            semantic_error_frequency["category"], categories=ERROR_ORDER, ordered=True
        )
        semantic_error_frequency = semantic_error_frequency.sort_values(["category", "source"])
        save_table(semantic_error_frequency, "05_semantic_error_category_frequency_compare")

        semantic_error_sets = (
            semantic_error_flags.assign(
                manual_label=lambda df: np.where(df["manual_present"], df["category"].astype(str), None),
                mlqa_label=lambda df: np.where(df["mlqa_present"], df["category"].astype(str), None),
            )
            .groupby(["building_id", "country"], as_index=False)
            .agg(
                manual_labels=("manual_label", lambda s: sorted(x for x in s if pd.notna(x))),
                mlqa_labels=("mlqa_label", lambda s: sorted(x for x in s if pd.notna(x))),
            )
        )
        semantic_error_sets["exact_set_match"] = (
            semantic_error_sets["manual_labels"].map(tuple)
            == semantic_error_sets["mlqa_labels"].map(tuple)
        )
        semantic_error_sets["jaccard_similarity"] = semantic_error_sets.apply(
            lambda row: (
                len(set(row["manual_labels"]) & set(row["mlqa_labels"]))
                / len(set(row["manual_labels"]) | set(row["mlqa_labels"]))
                if set(row["manual_labels"]) | set(row["mlqa_labels"])
                else 1.0
            ),
            axis=1,
        )
        semantic_error_case_summary = pd.DataFrame(
            [
                {
                    "paired_cases": int(len(semantic_error_sets)),
                    "exact_set_match_count": int(semantic_error_sets["exact_set_match"].sum()),
                    "exact_set_match_pct": round(100 * semantic_error_sets["exact_set_match"].mean(), 1),
                    "mean_jaccard_similarity": round(semantic_error_sets["jaccard_similarity"].mean(), 3),
                    "median_jaccard_similarity": round(semantic_error_sets["jaccard_similarity"].median(), 3),
                    "mean_manual_labels_per_case": round(semantic_error_sets["manual_labels"].map(len).mean(), 2),
                    "mean_mlqa_labels_per_case": round(semantic_error_sets["mlqa_labels"].map(len).mean(), 2),
                }
            ]
        )
        save_table(semantic_error_case_summary, "05_semantic_error_case_set_agreement")

    missing_outputs = read_sql(
        f"""
        with b as (
          select id, {COUNTRY_CASE_B} as country
          from src_google.buildings b
        ), sam as (select distinct building_id from src_google.detected_house),
        post as (select distinct building_id from src_google.detected_house_regularized)
        select b.country,
          count(*)::int as total_buildings,
          count(*) filter (where sam.building_id is null)::int as no_sam_detection,
          count(*) filter (where post.building_id is null)::int as no_post_output,
          count(*) filter (where sam.building_id is not null and post.building_id is null)::int as sam_but_no_post,
          count(*) filter (where post.building_id is not null)::int as with_post
        from b
        left join sam on sam.building_id=b.id
        left join post on post.building_id=b.id
        where b.country in {INCLUDED_COUNTRY_SQL}
        group by b.country order by b.country
        """
    )
    save_table(missing_outputs, "05_missing_outputs_by_country")

    # ------------------------------------------------------------------
    # Thesis results integration tables
    # ------------------------------------------------------------------
    # These tables collect the five result additions that are most useful
    # for the written results chapter. They deliberately reuse the core
    # statistics above so the numbers stay reproducible from one generator.
    pipeline_dropoff = coverage.merge(
        missing_outputs[
            [
                "country",
                "no_sam_detection",
                "sam_but_no_post",
                "no_post_output",
            ]
        ],
        on="country",
        how="left",
    )
    pipeline_dropoff["mlqa_rate_of_total_pct"] = (
        100 * pipeline_dropoff["mlqa_analyzed"] / pipeline_dropoff["total_buildings"].replace(0, np.nan)
    ).round(1)
    pipeline_dropoff["sam_rate_of_mlqa_pct"] = (
        100 * pipeline_dropoff["with_sam"] / pipeline_dropoff["mlqa_analyzed"].replace(0, np.nan)
    ).round(1)
    pipeline_dropoff["post_rate_of_total_pct"] = (
        100 * pipeline_dropoff["with_post"] / pipeline_dropoff["total_buildings"].replace(0, np.nan)
    ).round(1)
    pipeline_dropoff = pipeline_dropoff[
        [
            "country",
            "total_buildings",
            "mlqa_analyzed",
            "with_sam",
            "with_post",
            "manually_evaluated",
            "no_sam_detection",
            "sam_but_no_post",
            "no_post_output",
            "mlqa_rate_of_total_pct",
            "sam_rate_of_mlqa_pct",
            "post_rate_of_sam_pct",
            "manual_eval_rate_of_post_pct",
        ]
    ]
    save_table(pipeline_dropoff, "06_results_pipeline_dropoff")

    stage_comparison = geometry_summary.copy()
    original = stage_comparison.loc[
        stage_comparison["geometry_stage"] == "Original Google"
    ].iloc[0]
    stage_comparison["mean_area_change_vs_original_pct"] = (
        100 * (stage_comparison["mean_area_m2"] / original["mean_area_m2"] - 1)
    ).round(1)
    stage_comparison["mean_vertex_change_vs_original_pct"] = (
        100 * (stage_comparison["mean_vertices"] / original["mean_vertices"] - 1)
    ).round(1)
    save_table(stage_comparison, "06_results_stage_comparison")

    tree_context_results = geometry_by_tree_context.copy()
    tree_context_results = tree_context_results.rename(
        columns={"median_abs_area_error_pct": "median_abs_area_change_pct"}
    )
    tree_context_results = tree_context_results[
        [
            "tree_context",
            "n",
            "good_or_perfect_pct",
            "median_abs_area_change_pct",
            "missing_parts_pct",
            "extra_parts_pct",
        ]
    ]
    save_table(tree_context_results, "06_results_tree_context")

    mlqa_visibility_results = geometry_by_visibility.copy()
    mlqa_visibility_results = mlqa_visibility_results.rename(
        columns={"median_abs_area_error_pct": "median_abs_area_change_pct"}
    )
    mlqa_visibility_results = mlqa_visibility_results[
        [
            "mlqa_visibility",
            "n",
            "good_or_perfect_pct",
            "median_abs_area_change_pct",
            "missing_parts_pct",
            "extra_parts_pct",
        ]
    ]
    save_table(mlqa_visibility_results, "06_results_mlqa_visibility")

    country_synthesis = geometry_result_factors.copy()
    country_synthesis = country_synthesis.rename(
        columns={"median_abs_area_error_pct": "median_abs_area_change_pct"}
    )
    country_synthesis = country_synthesis[
        [
            "country",
            "n",
            "good_or_perfect_pct",
            "median_abs_area_change_pct",
            "median_post_shift_m",
            "post_missing_parts_pct",
            "post_extra_parts_pct",
            "medium_high_tree_context_pct",
        ]
    ]
    save_table(country_synthesis, "06_results_country_synthesis")

    return {
        "table_counts": table_counts,
        "coverage": coverage,
        "eval_rows": eval_rows,
        "rating_dist": rating_dist,
        "transition": transition,
        "country_improvement": country_improvement,
        "error_compare": error_compare,
        "country_error_profile": country_error_profile,
        "geom_eval": geom_eval,
        "geometry_summary": geometry_summary,
        "geometry_result_factors": geometry_result_factors,
        "geometry_by_visibility": geometry_by_visibility,
        "geometry_by_tree_context": geometry_by_tree_context,
        "confidence_by_quartile": confidence_by_quartile,
        "confidence_correlations": confidence_correlations,
        "new_errors_by_tree_detection": new_errors_by_tree_detection,
        "new_errors_by_tree_context": new_errors_by_tree_context,
        "new_error_tree_association": new_error_tree_association,
        "shift_eval": shift_eval,
        "shift_eval_good_post": shift_eval_good_post,
        "shift_country": shift_country,
        "semantic_error_flags": semantic_error_flags,
        "semantic_error_agreement": semantic_error_agreement,
        "semantic_error_frequency": semantic_error_frequency,
        "semantic_error_case_summary": semantic_error_case_summary,
        "missing_outputs": missing_outputs,
        "pipeline_dropoff": pipeline_dropoff,
        "stage_comparison": stage_comparison,
        "tree_context_results": tree_context_results,
        "mlqa_visibility_results": mlqa_visibility_results,
        "country_synthesis": country_synthesis,
    }


def build_figures(data: dict[str, pd.DataFrame]) -> None:
    coverage = data["coverage"]
    rating_dist = data["rating_dist"]
    eval_rows = data["eval_rows"]
    country_improvement = data["country_improvement"]
    error_compare = data["error_compare"]
    country_error_profile = data["country_error_profile"]
    geom_eval = data["geom_eval"]
    shift_eval = data["shift_eval"]
    shift_eval_good_post = data["shift_eval_good_post"]
    shift_country = data["shift_country"]
    missing_outputs = data["missing_outputs"]
    pipeline_dropoff = data["pipeline_dropoff"]
    stage_comparison = data["stage_comparison"]
    tree_context_results = data["tree_context_results"]
    mlqa_visibility_results = data["mlqa_visibility_results"]
    country_synthesis = data["country_synthesis"]
    new_errors_by_tree_detection = data["new_errors_by_tree_detection"]
    new_errors_by_tree_context = data["new_errors_by_tree_context"]
    confidence_by_quartile = data["confidence_by_quartile"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_df = coverage.melt(
        id_vars="country",
        value_vars=["total_buildings", "mlqa_analyzed", "with_sam", "with_post", "manually_evaluated"],
        var_name="stage",
        value_name="count",
    )
    sns.barplot(data=plot_df, x="country", y="count", hue="stage", ax=ax, palette="muted")
    ax.set_title("Pipeline coverage by country")
    ax.set_xlabel("Country / AOI")
    ax.set_ylabel("Number of buildings")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Stage", loc="upper right")
    save_fig(fig, "01_pipeline_coverage_by_country")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    sns.barplot(
        data=rating_dist,
        x="rating",
        y="count",
        hue="stage",
        order=RATING_ORDER,
        ax=ax,
        palette=[PALETTE["original"], PALETTE["post"]],
    )
    ax.set_title("Manual quality ratings before and after postprocessing")
    ax.set_xlabel("Quality rating")
    ax.set_ylabel("Number of evaluated samples")
    ax.legend(title="Geometry")
    save_fig(fig, "02_rating_distribution_before_after")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    trans_plot = pd.crosstab(eval_rows["original"], eval_rows["post"]).reindex(
        index=RATING_ORDER, columns=RATING_ORDER, fill_value=0
    )
    sns.heatmap(
        trans_plot,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar_kws={"label": "Number of samples"},
        ax=ax,
    )
    ax.set_title("Transition from original to postprocessed quality")
    ax.set_xlabel("Postprocessed rating")
    ax.set_ylabel("Original Google rating")
    save_fig(fig, "02_original_to_post_transition_heatmap")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    country_long = country_improvement.melt(
        id_vars="country",
        value_vars=["improved", "unchanged", "degraded"],
        var_name="change",
        value_name="count",
    )
    country_long = country_long.merge(country_improvement[["country", "n"]], on="country")
    country_long["percentage"] = 100 * country_long["count"] / country_long["n"]
    sns.barplot(
        data=country_long,
        x="country",
        y="percentage",
        hue="change",
        hue_order=["improved", "unchanged", "degraded"],
        ax=ax,
        palette=[PALETTE["improved"], PALETTE["unchanged"], PALETTE["degraded"]],
    )
    ax.set_title("Postprocessing effect by country")
    ax.set_xlabel("Country / AOI")
    ax.set_ylabel("Share of evaluated samples (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Change")
    save_fig(fig, "02_country_improvement_rates")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    err_plot = error_compare.copy()
    err_plot["error_type"] = pd.Categorical(err_plot["error_type"], categories=ERROR_ORDER, ordered=True)
    sns.barplot(
        data=err_plot.sort_values("error_type"),
        x="error_type",
        y="count",
        hue="stage",
        ax=ax,
        palette=[PALETTE["original"], PALETTE["post"]],
    )
    ax.set_title("Geometry error categories before and after postprocessing")
    ax.set_xlabel("Error category")
    ax.set_ylabel("Number of tagged errors")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Geometry")
    save_fig(fig, "02_error_categories_before_after")

    error_profile = country_error_profile.copy()
    location_order = INCLUDED_COUNTRIES + ["Total"]
    stage_order = ["Original Google", "Postprocessed"]
    error_profile["country"] = pd.Categorical(
        error_profile["country"], categories=location_order, ordered=True
    )
    error_profile["stage"] = pd.Categorical(
        error_profile["stage"], categories=stage_order, ordered=True
    )
    error_profile["error_type"] = pd.Categorical(
        error_profile["error_type"], categories=ERROR_ORDER, ordered=True
    )
    error_profile = error_profile.sort_values(["country", "stage", "error_type"])

    fig = plt.figure(figsize=(11.4, 7.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.50)
    ax_bar = fig.add_subplot(grid[0])
    ax_heat = fig.add_subplot(grid[1])

    bar_rows = []
    y_positions = []
    y_labels = []
    y = 0
    for country in location_order:
        for stage in stage_order:
            bar_rows.append((country, stage))
            y_positions.append(y)
            y_labels.append(f"{country} | {'Original' if stage == 'Original Google' else 'Post'}")
            y += 1
        y += 0.45

    profile_idx = error_profile.set_index(["country", "stage", "error_type"])
    for error_type in ERROR_ORDER:
        left = []
        widths = []
        for country, stage in bar_rows:
            previous = sum(
                profile_idx.loc[(country, stage, earlier), "share_pct"]
                for earlier in ERROR_ORDER[: ERROR_ORDER.index(error_type)]
            )
            left.append(previous)
            widths.append(profile_idx.loc[(country, stage, error_type), "share_pct"])
        ax_bar.barh(
            y_positions,
            widths,
            left=left,
            height=0.78,
            color=ERROR_COLORS[error_type],
            edgecolor="white",
            linewidth=0.7,
            label=error_type.replace("_", " ").title(),
        )

    ax_bar.set_xlim(0, 100)
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(y_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Share of evaluated cases (%)")
    ax_bar.set_title("Error profile by location and total", pad=14)
    ax_bar.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        columnspacing=1.2,
        handlelength=1.8,
    )

    original_share = (
        error_profile[error_profile["stage"] == "Original Google"]
        .pivot(index="country", columns="error_type", values="share_pct")
        .reindex(index=location_order, columns=ERROR_ORDER)
    )
    post_share = (
        error_profile[error_profile["stage"] == "Postprocessed"]
        .pivot(index="country", columns="error_type", values="share_pct")
        .reindex(index=location_order, columns=ERROR_ORDER)
    )
    delta_share = post_share - original_share
    max_abs_delta = float(np.nanmax(np.abs(delta_share.to_numpy())))
    sns.heatmap(
        delta_share,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs_delta,
        vmax=max_abs_delta,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Share change (percentage points)"},
        ax=ax_heat,
    )
    ax_heat.set_title("Shift in error composition after postprocessing")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_xticklabels(
        [label.get_text().replace("_", " ").title() for label in ax_heat.get_xticklabels()],
        rotation=25,
        ha="right",
    )
    fig.subplots_adjust(left=0.16, right=0.92, top=0.86, bottom=0.14)
    save_fig(fig, "02_error_categories_by_location_total", tight=False)

    agreement_order = [
        "more than 50% smaller",
        "25-50% smaller",
        "10-25% smaller",
        "within +/-10%",
        "10-25% larger",
        "25-50% larger",
        "more than 50% larger",
    ]
    area_agreement = (
        geom_eval.dropna(subset=["post_area_agreement"])
        .groupby(["country", "post_area_agreement"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    area_agreement["share_pct"] = 100 * area_agreement["n"] / area_agreement.groupby(
        "country"
    )["n"].transform("sum")
    area_pivot = (
        area_agreement.pivot(index="country", columns="post_area_agreement", values="share_pct")
        .reindex(index=INCLUDED_COUNTRIES, columns=agreement_order)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    area_pivot.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=["#8C3B46", "#C7545A", "#E7A95B", "#5DA271", "#7BAFD4", "#4C78A8", "#5B4B8A"],
        width=0.72,
    )
    ax.set_title("Postprocessed area agreement with original geometry")
    ax.set_xlabel("Share of evaluated postprocessed buildings (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(
        title="Post / original area",
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=3,
    )
    save_fig(fig, "03_post_area_agreement_by_country", tight=False)

    area_ratio_long = geom_eval[
        ["building_id", "sam_original_area_ratio", "post_original_area_ratio"]
    ].melt(id_vars="building_id", var_name="ratio_type", value_name="area_ratio")
    area_ratio_long = area_ratio_long.replace([np.inf, -np.inf], np.nan).dropna()
    area_ratio_long = area_ratio_long[area_ratio_long["area_ratio"].between(0, 5)]
    area_ratio_long["ratio_type"] = area_ratio_long["ratio_type"].map(
        {
            "sam_original_area_ratio": "SAM / Original",
            "post_original_area_ratio": "Post / Original",
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.boxplot(
        data=area_ratio_long,
        x="ratio_type",
        y="area_ratio",
        hue="ratio_type",
        showfliers=False,
        ax=ax,
        palette=[PALETTE["sam"], PALETTE["post"]],
        legend=False,
    )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Area ratio relative to original geometry")
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Area ratio, outliers hidden")
    save_fig(fig, "03_area_ratio_distribution")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    area_quality_df = geom_eval.dropna(subset=["post", "post_abs_area_error_pct"]).copy()
    area_quality_df = area_quality_df[
        area_quality_df["post_abs_area_error_pct"]
        <= area_quality_df["post_abs_area_error_pct"].quantile(0.98)
    ]
    sns.boxplot(
        data=area_quality_df,
        x="post",
        y="post_abs_area_error_pct",
        hue="post",
        order=RATING_ORDER,
        hue_order=RATING_ORDER,
        showfliers=False,
        ax=ax,
        palette=[PALETTE[r] for r in RATING_ORDER],
        legend=False,
    )
    ax.set_title("Area error by manual postprocessed quality")
    ax.set_xlabel("Manual postprocessed rating")
    ax.set_ylabel("Absolute area error vs original (%)")
    save_fig(fig, "03_post_area_error_by_quality")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    scatter_df = geom_eval.dropna(subset=["original_area_m2", "post_area_m2"]).copy()
    scatter_df = scatter_df[
        (scatter_df["original_area_m2"] <= scatter_df["original_area_m2"].quantile(0.98))
        & (scatter_df["post_area_m2"] <= scatter_df["post_area_m2"].quantile(0.98))
    ]
    sns.scatterplot(
        data=scatter_df,
        x="original_area_m2",
        y="post_area_m2",
        hue="country",
        s=22,
        alpha=0.7,
        ax=ax,
    )
    max_lim = max(scatter_df["original_area_m2"].max(), scatter_df["post_area_m2"].max())
    ax.plot([0, max_lim], [0, max_lim], color="black", linestyle="--", linewidth=1)
    ax.set_title("Original vs postprocessed building area")
    ax.set_xlabel("Original area (m2)")
    ax.set_ylabel("Postprocessed area (m2)")
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, "03_original_vs_post_area_scatter")

    context_df = geom_eval.assign(
        post_good_or_perfect=geom_eval["post"].isin(["good", "perfect"])
    )
    context_summary = (
        context_df.groupby("mlqa_visibility")
        .agg(good_or_perfect_pct=("post_good_or_perfect", lambda s: 100 * s.mean()))
        .reindex(["full house", "partial house", "no visible house", "not analyzed"])
        .dropna()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    sns.barplot(
        data=context_summary,
        x="mlqa_visibility",
        y="good_or_perfect_pct",
        color=PALETTE["post"],
        ax=ax,
    )
    ax.set_title("Postprocessed quality by MLQA visibility")
    ax.set_xlabel("MLQA visibility class")
    ax.set_ylabel("Good or perfect postprocessed results (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "03_post_quality_by_mlqa_visibility")

    tree_summary = (
        context_df.groupby("tree_context")
        .agg(good_or_perfect_pct=("post_good_or_perfect", lambda s: 100 * s.mean()))
        .reindex(
            [
                "no detected tree",
                "low tree context",
                "medium tree context",
                "high tree context",
            ]
        )
        .dropna()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    sns.barplot(
        data=tree_summary,
        x="tree_context",
        y="good_or_perfect_pct",
        color="#6B8F71",
        ax=ax,
    )
    ax.set_title("Postprocessed quality by tree context")
    ax.set_xlabel("Detected tree context")
    ax.set_ylabel("Good or perfect postprocessed results (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "03_post_quality_by_tree_context")

    confidence_plot = confidence_by_quartile[
        [
            "confidence_quartile",
            "good_or_perfect_pct",
            "improved_pct",
            "degraded_pct",
            "post_new_error_pct",
        ]
    ].melt(
        id_vars="confidence_quartile",
        var_name="metric",
        value_name="share_pct",
    )
    confidence_plot["metric"] = confidence_plot["metric"].map(
        {
            "good_or_perfect_pct": "good/perfect",
            "improved_pct": "improved",
            "degraded_pct": "degraded",
            "post_new_error_pct": "new post error",
        }
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    sns.lineplot(
        data=confidence_plot,
        x="confidence_quartile",
        y="share_pct",
        hue="metric",
        marker="o",
        ax=ax,
    )
    ax.set_title("Postprocessed result by Google Open Buildings confidence")
    ax.set_xlabel("Google Open Buildings confidence quartile")
    ax.set_ylabel("Share of evaluated buildings (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="")
    save_fig(fig, "03_post_quality_by_google_confidence")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.boxplot(data=shift_eval, x="country", y="post_dist", showfliers=False, ax=ax, color="#7BAFD4")
    ax.set_title("Centroid shift for shifted originals by country")
    ax.set_xlabel("Country / AOI")
    ax.set_ylabel("Shift distance (m), outliers hidden")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "04_post_shift_distance_by_country")

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for _, row in shift_country.iterrows():
        ax.arrow(
            0,
            0,
            row["avg_post_dx_m"],
            row["avg_post_dy_m"],
            head_width=0.12,
            length_includes_head=True,
            linewidth=1.8,
        )
        ax.text(row["avg_post_dx_m"] * 1.07, row["avg_post_dy_m"] * 1.07, row["country"], fontsize=9)
    lim = max(abs(shift_country["avg_post_dx_m"]).max(), abs(shift_country["avg_post_dy_m"]).max()) + 0.8
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mean original-to-post shift vector for shifted originals")
    ax.set_xlabel("Mean east-west shift dx (m)")
    ax.set_ylabel("Mean north-south shift dy (m)")
    save_fig(fig, "04_mean_post_shift_vectors")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter = shift_eval.dropna(subset=["post_dx_m", "post_dy_m"]).copy()
    sns.scatterplot(data=scatter, x="post_dx_m", y="post_dy_m", hue="country", s=22, alpha=0.55, ax=ax)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title("Shift-vector cloud for shifted originals")
    ax.set_xlabel("East-west shift dx (m)")
    ax.set_ylabel("North-south shift dy (m)")
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, "04_post_shift_vector_scatter")

    countries = [c for c in INCLUDED_COUNTRIES if c in set(scatter["country"])]
    if countries:
        lim = float(
            np.nanmax(
                np.abs(scatter[["post_dx_m", "post_dy_m"]].to_numpy(dtype=float))
            )
        )
        lim = min(max(lim + 0.5, 1.0), 12.0)
        major_tick_step = 4.0
        minor_tick_step = 2.0
        major_ticks = np.arange(
            np.floor(-lim / major_tick_step) * major_tick_step,
            np.ceil(lim / major_tick_step) * major_tick_step + major_tick_step,
            major_tick_step,
        )
        minor_ticks = np.arange(
            np.floor(-lim / minor_tick_step) * minor_tick_step,
            np.ceil(lim / minor_tick_step) * minor_tick_step + minor_tick_step,
            minor_tick_step,
        )
        fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), sharex=True, sharey=True)
        flat_axes = axes.ravel()
        for ax, country in zip(flat_axes, countries):
            country_df = scatter[scatter["country"] == country]
            sns.scatterplot(
                data=country_df,
                x="post_dx_m",
                y="post_dy_m",
                s=18,
                alpha=0.55,
                color=PALETTE["post"],
                edgecolor=None,
                ax=ax,
            )
            ax.axhline(0, color="grey", linewidth=0.7)
            ax.axvline(0, color="grey", linewidth=0.7)
            ax.set_title(f"{country} (n={len(country_df)})", fontsize=10)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(minor_ticks, minor=True)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axisbelow(True)
            ax.grid(True, which="major", linewidth=0.65, alpha=0.45, color="#8A8A8A")
            ax.grid(True, which="minor", linewidth=0.5, alpha=0.38, color="#A8A8A8")

        for ax in flat_axes[len(countries):]:
            ax.axis("off")

        fig.subplots_adjust(bottom=0.16, top=0.9, left=0.09, right=0.98, hspace=0.32, wspace=0.18)
        fig.suptitle("Shift-vector cloud for shifted originals by country", fontweight="bold")
        fig.supxlabel("East-west shift dx (m)", y=0.005)
        fig.supylabel("North-south shift dy (m)")
        save_fig(fig, "04_post_shift_vector_scatter_by_country", tight=False)

    scatter_good = shift_eval_good_post.dropna(subset=["post_dx_m", "post_dy_m"]).copy()
    if not scatter_good.empty:
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        sns.scatterplot(
            data=scatter_good,
            x="post_dx_m",
            y="post_dy_m",
            hue="country",
            s=24,
            alpha=0.6,
            ax=ax,
        )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_title("Shift-vector cloud for shifted originals with good/perfect post result")
        ax.set_xlabel("East-west shift dx (m)")
        ax.set_ylabel("North-south shift dy (m)")
        ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
        save_fig(fig, "04_post_shift_vector_scatter_good_perfect")

        countries_good = [c for c in INCLUDED_COUNTRIES if c in set(scatter_good["country"])]
        if countries_good:
            lim_good = float(
                np.nanmax(
                    np.abs(scatter_good[["post_dx_m", "post_dy_m"]].to_numpy(dtype=float))
                )
            )
            lim_good = min(max(lim_good + 0.5, 1.0), 12.0)
            major_tick_step = 4.0
            minor_tick_step = 2.0
            major_ticks_good = np.arange(
                np.floor(-lim_good / major_tick_step) * major_tick_step,
                np.ceil(lim_good / major_tick_step) * major_tick_step + major_tick_step,
                major_tick_step,
            )
            minor_ticks_good = np.arange(
                np.floor(-lim_good / minor_tick_step) * minor_tick_step,
                np.ceil(lim_good / minor_tick_step) * minor_tick_step + minor_tick_step,
                minor_tick_step,
            )
            fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), sharex=True, sharey=True)
            flat_axes = axes.ravel()
            for ax, country in zip(flat_axes, countries_good):
                country_df = scatter_good[scatter_good["country"] == country]
                sns.scatterplot(
                    data=country_df,
                    x="post_dx_m",
                    y="post_dy_m",
                    s=18,
                    alpha=0.6,
                    color=PALETTE["good"],
                    edgecolor=None,
                    ax=ax,
                )
                ax.axhline(0, color="grey", linewidth=0.7)
                ax.axvline(0, color="grey", linewidth=0.7)
                ax.set_title(f"{country} (n={len(country_df)})", fontsize=10)
                ax.set_xlim(-lim_good, lim_good)
                ax.set_ylim(-lim_good, lim_good)
                ax.set_xticks(major_ticks_good)
                ax.set_yticks(major_ticks_good)
                ax.set_xticks(minor_ticks_good, minor=True)
                ax.set_yticks(minor_ticks_good, minor=True)
                ax.set_aspect("equal", adjustable="box")
                ax.set_axisbelow(True)
                ax.grid(True, which="major", linewidth=0.65, alpha=0.45, color="#8A8A8A")
                ax.grid(True, which="minor", linewidth=0.5, alpha=0.38, color="#A8A8A8")

            for ax in flat_axes[len(countries_good):]:
                ax.axis("off")

            fig.subplots_adjust(bottom=0.16, top=0.88, left=0.09, right=0.98, hspace=0.32, wspace=0.18)
            fig.suptitle(
                "Shift-vector cloud for shifted originals with good/perfect post result by country",
                fontweight="bold",
            )
            fig.supxlabel("East-west shift dx (m)", y=0.005)
            fig.supylabel("North-south shift dy (m)")
            save_fig(fig, "04_post_shift_vector_scatter_good_perfect_by_country", tight=False)

    direction_df = shift_eval.dropna(subset=["post_dist", "post_az"]).copy()
    direction_df["post_direction_deg"] = np.degrees(direction_df["post_az"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    sns.boxplot(
        data=direction_df,
        x="country",
        y="post_dist",
        order=INCLUDED_COUNTRIES,
        showfliers=False,
        ax=axes[0],
        color="#7BAFD4",
    )
    axes[0].set_title("Shift distance")
    axes[0].set_xlabel("Country / AOI")
    axes[0].set_ylabel("Original-to-post shift (m)")
    axes[0].tick_params(axis="x", rotation=25)

    sns.boxplot(
        data=direction_df,
        x="country",
        y="post_direction_deg",
        order=INCLUDED_COUNTRIES,
        showfliers=False,
        ax=axes[1],
        color="#D9A441",
    )
    axes[1].set_title("Shift direction")
    axes[1].set_xlabel("Country / AOI")
    axes[1].set_ylabel("Azimuth (degrees clockwise from north)")
    axes[1].set_ylim(0, 360)
    axes[1].set_yticks(np.arange(0, 361, 45))
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Shift distance and direction for shifted originals", fontweight="bold")
    save_fig(fig, "04_post_shift_distance_direction_by_country")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    mlqa_plot = coverage.melt(
        id_vars="country",
        value_vars=["no_visible_house_mlqa", "partial_house_mlqa", "full_house_mlqa"],
        var_name="mlqa_class",
        value_name="count",
    )
    sns.barplot(data=mlqa_plot, x="country", y="count", hue="mlqa_class", ax=ax, palette="Set2")
    ax.set_title("MLQA building-presence classes by country")
    ax.set_xlabel("Country / AOI")
    ax.set_ylabel("Number of MLQA-analyzed buildings")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="MLQA class")
    save_fig(fig, "05_mlqa_presence_classes_by_country")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    miss_plot = missing_outputs.melt(
        id_vars="country",
        value_vars=["no_sam_detection", "sam_but_no_post", "with_post"],
        var_name="output_status",
        value_name="count",
    )
    sns.barplot(data=miss_plot, x="country", y="count", hue="output_status", ax=ax, palette="Set1")
    ax.set_title("Pipeline output status by country")
    ax.set_xlabel("Country / AOI")
    ax.set_ylabel("Number of original buildings")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Output status")
    save_fig(fig, "05_output_status_by_country")

    # ------------------------------------------------------------------
    # Thesis results integration figures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    dropoff_plot = pipeline_dropoff.melt(
        id_vars="country",
        value_vars=["mlqa_analyzed", "with_sam", "with_post", "manually_evaluated"],
        var_name="pipeline_stage",
        value_name="count",
    )
    sns.barplot(
        data=dropoff_plot,
        x="country",
        y="count",
        hue="pipeline_stage",
        ax=ax,
        palette="muted",
    )
    ax.set_title("Result addition 1: pipeline coverage and drop-off")
    ax.set_xlabel("Country / AOI")
    ax.set_ylabel("Number of buildings")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Pipeline stage")
    save_fig(fig, "06_results_pipeline_dropoff")

    stage_long = stage_comparison.melt(
        id_vars="geometry_stage",
        value_vars=["mean_area_m2", "mean_vertices"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3))
    sns.barplot(
        data=stage_long[stage_long["metric"] == "mean_area_m2"],
        x="geometry_stage",
        y="value",
        ax=axes[0],
        palette=[PALETTE["original"], PALETTE["sam"], PALETTE["post"]],
    )
    axes[0].set_title("Mean area")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Area (m2)")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(
        data=stage_long[stage_long["metric"] == "mean_vertices"],
        x="geometry_stage",
        y="value",
        ax=axes[1],
        palette=[PALETTE["original"], PALETTE["sam"], PALETTE["post"]],
    )
    axes[1].set_title("Mean vertices")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Vertices")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Result addition 2: original vs SAM vs postprocessed geometry")
    save_fig(fig, "06_results_stage_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    sns.barplot(
        data=tree_context_results,
        x="tree_context",
        y="good_or_perfect_pct",
        ax=axes[0],
        color="#6B8F71",
    )
    axes[0].set_title("Tree context")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Good or perfect results (%)")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(
        data=mlqa_visibility_results,
        x="mlqa_visibility",
        y="good_or_perfect_pct",
        ax=axes[1],
        color=PALETTE["post"],
    )
    axes[1].set_title("MLQA visibility")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Result additions 3 and 4: context effects on final quality")
    save_fig(fig, "06_results_context_effects")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.scatterplot(
        data=country_synthesis,
        x="median_abs_area_change_pct",
        y="good_or_perfect_pct",
        size="medium_high_tree_context_pct",
        hue="country",
        sizes=(80, 360),
        ax=ax,
    )
    for _, row in country_synthesis.iterrows():
        ax.text(
            row["median_abs_area_change_pct"] + 0.4,
            row["good_or_perfect_pct"] + 0.4,
            row["country"],
            fontsize=9,
        )
    ax.set_title("Result addition 5: country synthesis")
    ax.set_xlabel("Median absolute area change vs original (%)")
    ax.set_ylabel("Good or perfect results (%)")
    ax.set_ylim(55, 90)
    ax.legend(title="Country / tree context", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, "06_results_country_synthesis")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    sns.barplot(
        data=new_errors_by_tree_detection,
        x="tree_detected",
        y="post_new_error_pct",
        ax=axes[0],
        color="#C7545A",
    )
    axes[0].set_title("Tree detected vs no tree")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Postprocessing introduced new errors (%)")
    axes[0].set_ylim(0, 30)
    axes[0].tick_params(axis="x", rotation=15)
    sns.barplot(
        data=new_errors_by_tree_context,
        x="tree_context",
        y="post_new_error_pct",
        ax=axes[1],
        color="#C7545A",
    )
    axes[1].set_title("Tree-context intensity")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, 30)
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Postprocessing-introduced errors by detected tree context")
    save_fig(fig, "06_results_new_errors_by_tree")


COMMON_SETUP_CODE = r"""
from pathlib import Path
import pandas as pd
from IPython.display import display, Image


def find_repo_root(start=Path.cwd()):
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "evaluation" / "statistics").exists():
            return p
    raise RuntimeError("Could not find repository root")


REPO_ROOT = find_repo_root()
STAT_DIR = REPO_ROOT / "evaluation" / "statistics"
TABLE_DIR = STAT_DIR / "tables"
FIG_DIR = STAT_DIR / "figures"


def show_table(name, n=None):
    df = pd.read_csv(TABLE_DIR / f"{name}.csv")
    display(df.head(n) if n else df)
    return df


def show_figure(name, width=850):
    display(Image(filename=str(FIG_DIR / f"{name}.png"), width=width))
"""


def write_nb(filename: str, title: str, markdown_sections: list[str], code_sections: list[str] | None = None) -> Path:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n")]
    for md in markdown_sections:
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(md).strip() + "\n"))
    for code in code_sections or []:
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(code).strip() + "\n"))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path = STAT_DIR / filename
    nbf.write(nb, path)
    return path


def build_notebooks() -> None:
    write_nb(
        "00_generate_all_statistics.ipynb",
        "Generate All Thesis Statistics",
        [
            """
            This notebook documents the generated statistics package. The reusable generator is
            `evaluation/statistics/generate_thesis_statistics.py`.

            Run it from the repository root with:

            ```bash
            uv run python evaluation/statistics/generate_thesis_statistics.py
            ```

            The generator uses explicit read-only transactions against `src_google`.
            """,
            f"""
            Generated outputs:

            - Tables: `{TABLE_DIR}`
            - Figures: `{FIG_DIR}`

            Figures are saved as PNG and SVG. Tables are saved as CSV, Markdown, and LaTeX where possible.
            """,
        ],
    )

    write_nb(
        "01_dataset_and_pipeline_overview.ipynb",
        "Dataset And Pipeline Overview",
        [
            """
            ## What this notebook shows

            This notebook grounds the thesis in the empirical dataset. Bangladesh and Nepal2 are intentionally excluded from these thesis statistics. It answers how many original
            Google/Open Buildings footprints are available, how many were assessed by MLQA, how many
            received SAM/postprocessed outputs, and how many were manually evaluated.
            """,
            """
            ## Interpretation guide

            `no_visible_house_mlqa` is a proxy for cases where no building is visible in the image patch.
            Treat this carefully: it can mean non-building reference data, temporal mismatch, occlusion,
            imagery problems, or misregistration.
            """,
        ],
        [
            COMMON_SETUP_CODE,
            """
            show_table("01_table_counts")
            # Bangladesh and Nepal2 are intentionally excluded from the thesis statistics.
            show_table("01_country_pipeline_coverage")
            show_figure("01_pipeline_coverage_by_country")
            """,
        ],
    )

    write_nb(
        "02_manual_evaluation_categories.ipynb",
        "Manual Evaluation Categories",
        [
            """
            ## What this notebook shows

            This notebook is the core evaluation-category analysis. It compares manually labelled
            original Google geometry against the postprocessed result and breaks down which geometric
            errors occur before and after postprocessing.
            """,
            """
            ## Main reading

            The strongest thesis statement from this part is that postprocessing improves most
            evaluated samples while changing the error profile. Original data is dominated by
            `SHIFTED`; postprocessed geometry has far fewer shift labels but more `MISSING_PARTS`
            and `EXTRA_PARTS`, which are typical segmentation/postprocessing failure modes.
            """,
            """
            ## Location-specific error profile

            The location-plus-total error profile should be used when the thesis needs a richer
            explanation than the global before/after bars alone. The stacked bars compare the
            prevalence of each error category over all evaluated cases for every AOI and for the
            complete evaluation set, while the heatmap shows how strongly each category's share
            changes after postprocessing.
            """,
            """
            ## Rating scale

            `bad = 1`, `ok = 2`, `good = 3`, `perfect = 4`. A sample is counted as improved when
            the postprocessed rating has a higher score than the original rating.
            """,
        ],
        [
            COMMON_SETUP_CODE,
            """
            show_table("02_evaluation_duplicate_summary")
            show_table("02_rating_distribution")
            show_figure("02_rating_distribution_before_after")
            show_table("02_original_to_post_transition_matrix")
            show_figure("02_original_to_post_transition_heatmap")
            show_table("02_overall_improvement")
            show_table("02_country_improvement")
            show_figure("02_country_improvement_rates")
            show_table("02_postprocessing_new_error_frequency")
            show_table("02_error_categories_before_after")
            show_figure("02_error_categories_before_after")
            show_table("02_error_categories_by_location_total")
            show_figure("02_error_categories_by_location_total", width=1050)
            """,
        ],
    )

    write_nb(
        "03_geometry_before_after.ipynb",
        "Geometry Before And After",
        [
            """
            ## What this notebook shows

            This notebook measures how the geometry changes numerically. It looks at building area,
            centroid displacement, MLQA visibility, and tree context for original Google footprints
            and postprocessed geometry. Raw SAM geometry is kept in the raw export where it helps
            interpret intermediate pipeline behavior, but the figures focus on result quality.
            """,
            """
            ## Main reading

            The most useful results signal is not that SAM has many vertices. The more relevant
            question is where postprocessing preserves area, reduces shift, and still receives a
            good manual rating. Visibility and nearby detected trees help explain difficult cases,
            especially missing or extra parts after postprocessing.
            """,
            """
            ## Method note

            Postprocessed rows do not contain a reliable `source_id` back to a specific SAM row.
            For building-level statistics, geometries are aggregated by `building_id` using
            `ST_UnaryUnion(ST_Collect(...))`.
            """,
        ],
        [
            COMMON_SETUP_CODE,
            """
            show_table("03_geometry_stage_summary")
            show_table("03_country_geometry_summary")
            show_table("03_geometry_result_factors_by_country")
            show_table("03_post_area_agreement_by_country")
            show_table("03_geometry_by_post_quality")
            show_table("03_geometry_by_mlqa_visibility")
            show_table("03_geometry_by_tree_context")
            show_table("03_new_errors_by_tree_detection")
            show_table("03_new_errors_by_tree_context")
            show_table("03_new_error_tree_association")
            show_figure("03_post_area_agreement_by_country", width=1050)
            show_figure("03_area_ratio_distribution")
            show_figure("03_post_area_error_by_quality")
            show_figure("03_original_vs_post_area_scatter")
            show_figure("03_post_quality_by_mlqa_visibility")
            show_figure("03_post_quality_by_tree_context")
            raw = pd.read_csv(TABLE_DIR / "03_geometry_eval_building_level_raw.csv")
            display(raw.head())
            print(f"Raw building-level geometry rows: {len(raw)}")
            """,
        ],
    )

    write_nb(
        "04_shift_analysis.ipynb",
        "Shift Analysis",
        [
            """
            ## What this notebook shows

            This notebook tests whether the spatial shift between Google/Open Buildings and the
            image-derived geometry is constant. It now restricts the analysis to buildings whose
            original Google geometry was manually tagged `SHIFTED`, then computes centroid
            displacement vectors from original geometry to postprocessed geometry and summarizes
            distance and direction by country.
            """,
            """
            ## Main reading

            The shift is not globally constant even within the `SHIFTED` subset. Several countries
            show systematic local directions, but the direction changes by AOI. This supports a
            geography/data-quality interpretation: imagery/reference misalignment is spatially
            heterogeneous, so a single global correction is not defensible.
            """,
            """
            ## Coordinate convention

            `dx` is east-west shift in meters; positive means east. `dy` is north-south shift in
            meters; positive means north.
            """,
            """
            ## Scope note

            The tables and plots only include the latest evaluation row per building where the
            original geometry carries the `SHIFTED` label.
            """,
        ],
        [
            COMMON_SETUP_CODE,
            """
            show_table("04_country_shift_vectors")
            show_table("04_shift_summary_for_discussion")
            show_figure("04_post_shift_distance_by_country")
            show_figure("04_post_shift_distance_direction_by_country", width=950)
            show_figure("04_mean_post_shift_vectors")
            show_figure("04_post_shift_vector_scatter")
            show_figure("04_post_shift_vector_scatter_by_country", width=950)
            show_figure("04_post_shift_vector_scatter_good_perfect")
            show_figure("04_post_shift_vector_scatter_good_perfect_by_country", width=950)
            raw = pd.read_csv(TABLE_DIR / "04_shift_vectors_raw.csv")
            display(raw.head())
            print(f"Raw shift-vector rows: {len(raw)}")
            raw_good = pd.read_csv(TABLE_DIR / "04_shift_vectors_good_perfect_post_raw.csv")
            display(raw_good.head())
            print(f"Raw shifted-original rows with good/perfect post result: {len(raw_good)}")
            """,
        ],
    )

    write_nb(
        "05_non_buildings_and_dropoff.ipynb",
        "Non-Buildings And Pipeline Drop-Off",
        [
            """
            ## What this notebook shows

            This notebook looks at cases where the reference footprint may not correspond to a visible
            building and where the pipeline does not produce downstream geometry.
            """,
            """
            ## Main reading

            Use careful wording in the thesis. `house_present = false` should not automatically be
            called a deleted building. It means no visible building was found in the patch by MLQA.
            Possible explanations include non-building reference data, temporal mismatch, imagery
            problems, occlusion, or misregistration.
            """,
            """
            ## Recommended thesis framing

            This section supports the argument that automated footprint correction is a geospatial
            data-quality problem, not only an AI model-performance problem.
            """,
        ],
        [
            COMMON_SETUP_CODE,
            """
            show_table("01_country_pipeline_coverage")
            show_table("05_missing_outputs_by_country")
            show_table("05_mlqa_error_counts")
            show_table("05_semantic_error_category_frequency_compare")
            show_table("05_semantic_error_category_agreement")
            show_table("05_semantic_error_case_set_agreement")
            show_figure("05_mlqa_presence_classes_by_country")
            show_figure("05_output_status_by_country")
            """,
        ],
    )

    write_nb(
        "06_results_section_additions.ipynb",
        "Results Section Additions",
        [
            """
            ## Purpose

            This notebook collects the five additional result blocks that strengthen the written
            results chapter without mixing them into the lower-level exploratory notebooks.
            The same tables and figures are generated by `generate_thesis_statistics.py`, so each
            number can be reproduced from the database.
            """,
            """
            ## Result addition 1: pipeline coverage and drop-off

            Use this near the beginning of the results chapter. It explains how many buildings pass
            each stage: original candidates, MLQA-analyzed buildings, SAM outputs, postprocessed
            outputs, and manually evaluated cases.
            """,
            """
            ## Result addition 2: original vs SAM vs postprocessed geometry

            Use this after the main improvement rates. It shows the role of the intermediate SAM
            geometry: SAM creates detailed image-derived polygons, while postprocessing reduces
            complexity and produces cleaner footprints.
            """,
            """
            ## Result additions 3 and 4: failure conditions

            Tree context and MLQA visibility help explain where the workflow performs worse. In the
            thesis text, call the area metric an absolute area change relative to the original
            footprint, not a ground-truth error.
            """,
            """
            ## Result addition 5: country synthesis

            Use the country synthesis table as a compact explanation of why the study areas differ.
            It combines quality, area change, shift, postprocessing errors, and tree context.
            """,
            """
            ## Additional check: new errors and detected trees

            This checks whether postprocessing-introduced errors are more frequent when tree masks
            were detected around a building. The association is descriptive, not causal: detected
            trees are also a proxy for visually complex scenes.
            """,
        ],
        [
            COMMON_SETUP_CODE,
            """
            show_table("06_results_pipeline_dropoff")
            show_figure("06_results_pipeline_dropoff")
            """,
            """
            show_table("06_results_stage_comparison")
            show_figure("06_results_stage_comparison")
            """,
            """
            show_table("06_results_tree_context")
            show_table("06_results_mlqa_visibility")
            show_figure("06_results_context_effects")
            """,
            """
            show_table("06_results_country_synthesis")
            show_figure("06_results_country_synthesis")
            """,
            """
            show_table("03_new_errors_by_tree_detection")
            show_table("03_new_errors_by_tree_context")
            show_table("03_new_error_tree_association")
            show_figure("06_results_new_errors_by_tree")
            """,
        ],
    )

    write_nb(
        "README_statistics_index.ipynb",
        "Statistics Notebook Index",
        [
            """
            ## Files created

            - `00_generate_all_statistics.ipynb`: generator documentation.
            - `01_dataset_and_pipeline_overview.ipynb`: dataset and stage coverage.
            - `02_manual_evaluation_categories.ipynb`: manual ratings and error categories.
            - `03_geometry_before_after.ipynb`: postprocessed area agreement, geometry shift,
              MLQA visibility, and tree-context signals.
            - `04_shift_analysis.ipynb`: centroid shift distance and direction.
            - `05_non_buildings_and_dropoff.ipynb`: no-house proxy and pipeline drop-off.
            - `06_results_section_additions.ipynb`: thesis-facing result additions 1--5.
            """,
            """
            ## Suggested thesis storyline

            1. Start with dataset coverage and spatial sampling.
            2. Show that postprocessing improves most manually evaluated buildings.
            3. Explain that the original error profile is dominated by spatial shift, and use the
               location-plus-total profile to show how the error mix differs between AOIs.
            4. Show where postprocessed geometry preserves area and quality, and use MLQA/tree
               context to explain missing or extra parts.
            5. Use the shift-vector analysis to argue that misalignment is spatially heterogeneous.
            6. Close with no-house/drop-off analysis as evidence that data quality matters.
            """,
        ],
        [
            COMMON_SETUP_CODE,
            """
            show_table("01_table_counts")
            show_table("02_overall_improvement")
            show_table("04_country_shift_vectors")
            show_table("06_results_country_synthesis")
            """,
        ],
    )


def main() -> None:
    data = build_tables()
    build_figures(data)
    build_notebooks()
    print(f"Created statistics notebooks and outputs in {STAT_DIR}")
    print(f"Tables: {len(list(TABLE_DIR.glob('*.csv')))} CSV files")
    print(f"Figures: {len(list(FIG_DIR.glob('*.png')))} PNG files")


if __name__ == "__main__":
    main()



