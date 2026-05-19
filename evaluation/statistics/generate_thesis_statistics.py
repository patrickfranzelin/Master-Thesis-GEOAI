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
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (TABLE_DIR / f"{name}.md").write_text(md_table(df), encoding="utf-8")
    try:
        df.to_latex(TABLE_DIR / f"{name}.tex", index=False, escape=True)
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
            e.building_id, e.original, e.post, e.sam as post_introduced_new_errors,
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
        )
        select ev.building_id,
          {COUNTRY_CASE_B} as country,
          ev.original, ev.post, ev.change, ev.original_score, ev.post_score,
          ST_Area(ST_MakeValid(b.geom)::geography) as original_area_m2,
          ST_Area(s.geom::geography) as sam_area_m2,
          ST_Area(p.geom::geography) as post_area_m2,
          ST_NPoints(ST_MakeValid(b.geom)) as original_vertices,
          ST_NPoints(s.geom) as sam_vertices,
          ST_NPoints(p.geom) as post_vertices,
          ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(s.geom)::geography) as orig_sam_shift_m,
          ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(p.geom)::geography) as orig_post_shift_m,
          ST_Distance(ST_Centroid(s.geom)::geography, ST_Centroid(p.geom)::geography) as sam_post_shift_m
        from ev_latest ev
        join src_google.buildings b on b.id=ev.building_id
        left join sam s on s.building_id=ev.building_id
        left join post p on p.building_id=ev.building_id
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

    shift_eval = read_sql(
        f"""
        with ev_latest as (
          select distinct on (e.building_id) e.building_id
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
          select ev.building_id, {COUNTRY_CASE_B} as country,
            ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(s.geom)::geography) sam_dist,
            ST_Azimuth(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(s.geom)::geography) sam_az,
            ST_Distance(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(p.geom)::geography) post_dist,
            ST_Azimuth(ST_Centroid(ST_MakeValid(b.geom))::geography, ST_Centroid(p.geom)::geography) post_az
          from ev_latest ev
          join src_google.buildings b on b.id=ev.building_id
          left join sam s on s.building_id=ev.building_id
          left join post p on p.building_id=ev.building_id
        )
        select building_id, country, sam_dist, sam_az,
               sam_dist * sin(sam_az) as sam_dx_m,
               sam_dist * cos(sam_az) as sam_dy_m,
               post_dist, post_az,
               post_dist * sin(post_az) as post_dx_m,
               post_dist * cos(post_az) as post_dy_m
        from vectors
        """
    )
    shift_eval.to_csv(TABLE_DIR / "04_shift_vectors_raw.csv", index=False)
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

    mlqa_error_counts = read_sql(
        """
        select x.err as mlqa_error_type, count(*)::int as count
        from src_google.building_mlqa m
        cross join lateral jsonb_array_elements_text(coalesce(m.errors, '[]'::jsonb)) as x(err)
        group by x.err order by count desc
        """
    )
    save_table(mlqa_error_counts, "05_mlqa_error_counts")

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
        "shift_eval": shift_eval,
        "shift_country": shift_country,
        "missing_outputs": missing_outputs,
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
    shift_country = data["shift_country"]
    missing_outputs = data["missing_outputs"]

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

    vertex_long = geom_eval[
        ["building_id", "original_vertices", "sam_vertices", "post_vertices"]
    ].melt(id_vars="building_id", var_name="stage", value_name="vertices").dropna()
    vertex_long["stage"] = vertex_long["stage"].map(
        {
            "original_vertices": "Original Google",
            "sam_vertices": "SAM",
            "post_vertices": "Postprocessed",
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.boxplot(
        data=vertex_long,
        x="stage",
        y="vertices",
        showfliers=False,
        ax=ax,
        palette=[PALETTE["original"], PALETTE["sam"], PALETTE["post"]],
    )
    ax.set_title("Vertex count distribution")
    ax.set_xlabel("Geometry stage")
    ax.set_ylabel("Number of vertices, outliers hidden")
    save_fig(fig, "03_vertex_count_distribution")

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
        showfliers=False,
        ax=ax,
        palette=[PALETTE["sam"], PALETTE["post"]],
    )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Area ratio relative to original geometry")
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Area ratio, outliers hidden")
    save_fig(fig, "03_area_ratio_distribution")

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
        lim = max(lim + 0.5, 1.0)
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
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.35, alpha=0.25)

        for ax in flat_axes[len(countries):]:
            ax.axis("off")

        fig.suptitle("Shift-vector cloud for shifted originals by country", fontweight="bold")
        fig.supxlabel("East-west shift dx (m)")
        fig.supylabel("North-south shift dy (m)")
        save_fig(fig, "04_post_shift_vector_scatter_by_country", tight=False)

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
            vertex count, and centroid displacement for original Google footprints, raw SAM geometry,
            and postprocessed geometry.
            """,
            """
            ## Main reading

            Original Google footprints are usually very simple. SAM produces much more detailed
            boundaries. Postprocessing reduces the SAM vertex count strongly, which makes the geometry
            more GIS-like, but it can also remove roof parts or add unwanted parts in difficult scenes.
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
            show_table("03_geometry_by_post_quality")
            show_figure("03_vertex_count_distribution")
            show_figure("03_area_ratio_distribution")
            show_figure("03_original_vs_post_area_scatter")
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
            show_figure("04_post_shift_distance_by_country")
            show_figure("04_mean_post_shift_vectors")
            show_figure("04_post_shift_vector_scatter")
            show_figure("04_post_shift_vector_scatter_by_country", width=950)
            raw = pd.read_csv(TABLE_DIR / "04_shift_vectors_raw.csv")
            display(raw.head())
            print(f"Raw shift-vector rows: {len(raw)}")
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
            show_figure("05_mlqa_presence_classes_by_country")
            show_figure("05_output_status_by_country")
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
            - `03_geometry_before_after.ipynb`: area, vertices, and geometry complexity.
            - `04_shift_analysis.ipynb`: centroid shift distance and direction.
            - `05_non_buildings_and_dropoff.ipynb`: no-house proxy and pipeline drop-off.
            """,
            """
            ## Suggested thesis storyline

            1. Start with dataset coverage and spatial sampling.
            2. Show that postprocessing improves most manually evaluated buildings.
            3. Explain that the original error profile is dominated by spatial shift, and use the
               location-plus-total profile to show how the error mix differs between AOIs.
            4. Show that postprocessing regularizes geometry, especially vertices, but can introduce
               missing or extra parts.
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



