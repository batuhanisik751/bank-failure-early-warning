/** Minimal RFC 4180 CSV writer shared by the download routes. Pure and synthetic-testable. */

export type CsvCell = string | number | boolean | null | undefined;

/**
 * Text that Excel, Sheets and LibreOffice would evaluate as a formula when the file is opened:
 * a leading =, +, -, @, tab or carriage return. Numeric cells (typed numbers, or text that is
 * a plain number such as "-0.5") are never formulas.
 */
const FORMULA_START = /^[=+\-@\t\r]/;

export function csvCell(value: CsvCell): string {
  if (value == null) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  let text = String(value);
  if (typeof value === "string" && FORMULA_START.test(text) && !Number.isFinite(Number(text))) {
    // Neutralise spreadsheet formula injection (bank names come from the FDIC feed): a leading
    // apostrophe makes the cell literal text, and the quotes keep the apostrophe in the file.
    text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  }
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Header row first, then one line per row, CRLF-terminated, no trailing blank line. */
export function toCsv(headers: readonly string[], rows: ReadonlyArray<ReadonlyArray<CsvCell>>): string {
  const lines = [headers.map(csvCell).join(",")];
  for (const row of rows) lines.push(row.map(csvCell).join(","));
  return `${lines.join("\r\n")}\r\n`;
}

/**
 * Headers for a CSV response the browser saves under `filename`. The download routes are
 * `force-dynamic` and read the same `unstable_cache` data the pages do, so the file is never
 * stored by a shared cache: after `POST /api/revalidate` the next download is the new quarter,
 * which an `s-maxage` on a CDN would defeat for up to an hour.
 */
export function csvHeaders(filename: string): HeadersInit {
  return {
    "content-type": "text/csv; charset=utf-8",
    "content-disposition": `attachment; filename="${filename}"`,
    "cache-control": "no-store",
  };
}
