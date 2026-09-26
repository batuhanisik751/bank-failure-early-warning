import { readFileSync } from "node:fs";
import path from "node:path";
import { getTableColumns, getTableName } from "drizzle-orm";
import { describe, expect, it } from "vitest";
import { tables } from "@/lib/db/schema";

// Python owns the DDL. This test parses it and fails when the Drizzle mirror drifts.
const SQL_PATH = path.resolve(__dirname, "../../src/bankcanary/publish/schema.sql");

/** SQL type keyword -> the Drizzle columnType that mirrors it. */
const TYPE_MAP: Record<string, string> = {
  text: "PgText",
  bigint: "PgBigInt53",
  integer: "PgInteger",
  real: "PgReal",
  "double precision": "PgDoublePrecision",
  boolean: "PgBoolean",
  date: "PgDateString",
  timestamptz: "PgTimestampString",
  jsonb: "PgJsonb",
};

type SqlColumn = { name: string; type: string };

function parseSchema(sqlText: string): Map<string, SqlColumn[]> {
  const out = new Map<string, SqlColumn[]>();
  const re = /CREATE TABLE IF NOT EXISTS (\w+) \(([\s\S]*?)\n\);/g;
  for (const match of sqlText.matchAll(re)) {
    const columns: SqlColumn[] = [];
    for (const raw of match[2].split("\n")) {
      const line = raw.trim().replace(/,$/, "");
      if (!line || line.startsWith("PRIMARY KEY")) continue;
      const m = /^(\w+)\s+(double precision|\w+)/.exec(line);
      if (m) columns.push({ name: m[1], type: m[2] });
    }
    out.set(match[1], columns);
  }
  return out;
}

describe("Drizzle schema mirrors schema.sql", () => {
  const sql = parseSchema(readFileSync(SQL_PATH, "utf8"));

  it("parses every table the contract names", () => {
    expect(sql.size).toBe(14);
  });

  it("has exactly the tables of schema.sql", () => {
    expect(Object.keys(tables).sort()).toEqual([...sql.keys()].sort());
    for (const [name, table] of Object.entries(tables)) {
      expect(getTableName(table)).toBe(name);
    }
  });

  for (const [name, table] of Object.entries(tables)) {
    it(`${name}: column names and types match`, () => {
      const expected = sql.get(name) ?? [];
      const actual = Object.values(getTableColumns(table)).map((c) => ({
        name: c.name,
        type: c.columnType,
      }));
      expect(actual.map((c) => c.name)).toEqual(expected.map((c) => c.name));
      expect(actual.map((c) => c.type)).toEqual(
        expected.map((c) => TYPE_MAP[c.type] ?? `unmapped:${c.type}`),
      );
    });
  }
});
