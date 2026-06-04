'use client';

import React, { useMemo, useState } from "react";
import styles from "./SortableTable.module.css";

/* ── Public types ──────────────────────────────────────── */

export interface Column {
  /** Property key on each data record. */
  key: string;
  /** Display label in the header. */
  label: string;
  /** If true, sort numerically (strips commas, parses floats). */
  numeric?: boolean;
}

export interface SortableTableProps {
  columns: Column[];
  data: Record<string, string | number | null | undefined>[];
  /** Show a filter text-input above the table. */
  filterable?: boolean;
  filterPlaceholder?: string;
  /** Make the header row sticky when scrolling. */
  stickyHeader?: boolean;
}

/* ── Sort direction cycle: asc → desc → none ───────────── */

type SortDir = "asc" | "desc" | null;

function nextDir(current: SortDir): SortDir {
  if (current === "asc") return "desc";
  if (current === "desc") return null;
  return "asc";
}

function dirArrow(dir: SortDir): string {
  if (dir === "asc") return "▲";
  if (dir === "desc") return "▼";
  return "⇅";
}

/* ── Helpers ───────────────────────────────────────────── */

function parseNumeric(v: unknown): number {
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const cleaned = v.replace(/,/g, "");
    const n = parseFloat(cleaned);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

function cellText(v: unknown): string {
  if (v == null) return "";
  return String(v);
}

/* ── Component ─────────────────────────────────────────── */

export default function SortableTable({
  columns,
  data,
  filterable = false,
  filterPlaceholder = "Filter rows…",
  stickyHeader = false,
}: SortableTableProps): React.ReactElement {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>(null);
  const [filter, setFilter] = useState("");

  /* Click a header: cycle sort on that column. */
  function handleHeaderClick(key: string) {
    if (key === sortKey) {
      const nd = nextDir(sortDir);
      setSortDir(nd);
      if (nd === null) setSortKey(null);
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  /* Filtered rows. */
  const filtered = useMemo(() => {
    if (!filter) return data;
    const lc = filter.toLowerCase();
    return data.filter((row) =>
      columns.some((col) => cellText(row[col.key]).toLowerCase().includes(lc)),
    );
  }, [data, columns, filter]);

  /* Sorted rows. */
  const rows = useMemo(() => {
    if (!sortKey || !sortDir) return filtered;
    const col = columns.find((c) => c.key === sortKey);
    if (!col) return filtered;

    const copy = [...filtered];
    const mul = sortDir === "asc" ? 1 : -1;

    if (col.numeric) {
      copy.sort((a, b) => mul * (parseNumeric(a[col.key]) - parseNumeric(b[col.key])));
    } else {
      copy.sort((a, b) => mul * cellText(a[col.key]).localeCompare(cellText(b[col.key])));
    }
    return copy;
  }, [filtered, columns, sortKey, sortDir]);

  return (
    <div className={styles.wrapper}>
      {/* Filter bar */}
      {filterable && (
        <div className={styles.filterRow}>
          <input
            className={styles.filterInput}
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={filterPlaceholder}
            aria-label={filterPlaceholder}
          />
          <span className={styles.rowCount}>
            {rows.length} of {data.length} rows
          </span>
        </div>
      )}

      {/* Table */}
      <div className={styles.tableContainer}>
        <table
          className={`${styles.table} ${stickyHeader ? styles.stickyHeader : ""}`}
        >
          <thead>
            <tr>
              {columns.map((col) => {
                const active = sortKey === col.key;
                return (
                  <th
                    key={col.key}
                    onClick={() => handleHeaderClick(col.key)}
                    aria-sort={
                      active && sortDir === "asc"
                        ? "ascending"
                        : active && sortDir === "desc"
                          ? "descending"
                          : "none"
                    }
                  >
                    {col.label}
                    <span
                      className={`${styles.sortIndicator} ${active && sortDir ? styles.sortActive : ""}`}
                    >
                      {active ? dirArrow(sortDir) : "⇅"}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td key={col.key}>{cellText(row[col.key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
