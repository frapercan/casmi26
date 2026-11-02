"""Merge external structure DBs into one mass-sorted candidate table.

Sources (public Kaggle datasets):
  rishavae/casmi-2026-candidate-database   COCONUT + PubChem shards (~18M)
  xjoannax88/casmi-pubchemlite-candidates  PubChemLite (+ n_entries popularity)
Output columns: inchikey14, smiles, exact_mass, in_coconut, in_pcl, pcl_entries
"""
from __future__ import annotations

import argparse

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--coconut", default="/home/user/data/cands/coconut.parquet")
ap.add_argument("--pubchem", default="/home/user/data/cands/pubchem/*.parquet")
ap.add_argument("--pcl", default="/home/user/data/cands/pcl.parquet")
ap.add_argument("--out", default="/home/user/work/candidates.parquet")
a = ap.parse_args()

con = duckdb.connect()
con.execute("SET memory_limit='8GB'; SET threads=4;")
con.execute(f"""
COPY (
  WITH coco AS (SELECT inchikey14, smiles, exact_mass FROM '{a.coconut}'),
       pc   AS (SELECT inchikey14, smiles, exact_mass FROM '{a.pubchem}'),
       pcl  AS (SELECT key AS inchikey14, smiles, exact_mass, n_entries
                FROM '{a.pcl}' WHERE charge = 0),
       allm AS (
         SELECT inchikey14, smiles, exact_mass, 0 AS pri FROM coco
         UNION ALL SELECT inchikey14, smiles, exact_mass, 1 FROM pcl
         UNION ALL SELECT inchikey14, smiles, exact_mass, 2 FROM pc),
       best AS (
         SELECT inchikey14, arg_min(smiles, pri) AS smiles, arg_min(exact_mass, pri) AS exact_mass
         FROM allm GROUP BY inchikey14)
  SELECT b.*,
         b.inchikey14 IN (SELECT inchikey14 FROM coco) AS in_coconut,
         p.n_entries IS NOT NULL AS in_pcl,
         coalesce(p.n_entries, 0) AS pcl_entries
  FROM best b LEFT JOIN pcl p USING (inchikey14)
  ORDER BY exact_mass
) TO '{a.out}' (FORMAT parquet, ROW_GROUP_SIZE 500000)
""")
print(con.execute(f"""SELECT count(*), sum(in_coconut::int), sum(in_pcl::int)
                      FROM '{a.out}'""").fetchall())
