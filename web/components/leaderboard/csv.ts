/** Minimal RFC 4180 CSV writer shared by the download routes. Pure and synthetic-testable. */

export type CsvCell = string | number | boolean | null | undefined;

export function csvCell(value: CsvCell): string {
  if (value == null) return "";
  const text = typeof value === "number" ? (Number.isFinite(value) ? String(value) : "") : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Header row first, then one line per row, CRLF-terminated, no trailing blank line. */
export function toCsv(headers: readonly string[], rows: ReadonlyArray<ReadonlyArray<CsvCell>>): string {
  const lines = [headers.map(csvCell).join(",")];
  for (const row of rows) lines.push(row.map(csvCell).join(","));
  return `${lines.join("\r\n")}\r\n`;
}

/** Headers for a CSV response the browser saves under `filename`; cached like the pages. */
export function csvHeaders(filename: string): HeadersInit {
  return {
    "content-type": "text/csv; charset=utf-8",
    "content-disposition": `attachment; filename="${filename}"`,
    "cache-control": "public, max-age=0, s-maxage=3600",
  };
}
