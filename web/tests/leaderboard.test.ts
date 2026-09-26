import { describe, expect, it } from "vitest";
import { csvCell, toCsv } from "@/components/leaderboard/csv";
import {
  leaderboardHref,
  pageWindow,
  parseLeaderboardParams,
  sortHref,
} from "@/components/leaderboard/params";

describe("parseLeaderboardParams", () => {
  it("defaults to rank ascending on page 1 with no filters", () => {
    expect(parseLeaderboardParams({})).toEqual({ filters: {}, page: 1, sort: "rank", dir: "asc" });
  });
  it("keeps only known values and normalises case", () => {
    const p = parseLeaderboardParams({
      state: "tx", size: "1b_10b", charter: "nm", band: "high", q: " First ", sort: "assets", page: "3",
    });
    expect(p.filters).toEqual({ state: "TX", sizeBucket: "1b_10b", bkclass: "NM", band: "high", search: "First" });
    expect(p).toMatchObject({ page: 3, sort: "assets", dir: "desc" });
  });
  it("ignores malformed values", () => {
    const p = parseLeaderboardParams({ state: "Texas", size: "huge", charter: "ZZ", band: "red", sort: "x", dir: "up", page: "-2" });
    expect(p).toEqual({ filters: {}, page: 1, sort: "rank", dir: "asc" });
    expect(parseLeaderboardParams({ page: ["7", "8"] }).page).toBe(7);
  });
});

describe("leaderboardHref and sortHref", () => {
  const current = parseLeaderboardParams({ state: "CA", q: "bank", sort: "assets", page: "2" });
  it("serialises filters and drops defaults", () => {
    expect(leaderboardHref(current)).toBe("/?state=CA&q=bank&sort=assets&page=2");
    expect(leaderboardHref(parseLeaderboardParams({}))).toBe("/");
  });
  it("resets the page when the sort changes and flips direction on the same column", () => {
    expect(sortHref(current, "name")).toBe("/?state=CA&q=bank&sort=name");
    expect(sortHref(current, "assets")).toBe("/?state=CA&q=bank&sort=assets&dir=asc");
    expect(leaderboardHref(current, { page: 5 }, "/api/download/leaderboard.csv")).toBe(
      "/api/download/leaderboard.csv?state=CA&q=bank&sort=assets&page=5",
    );
  });
});

describe("pageWindow", () => {
  it("shows first, last and a window around the current page", () => {
    expect(pageWindow(1, 1)).toEqual([1]);
    expect(pageWindow(5, 20)).toEqual([1, 3, 4, 5, 6, 7, 20]);
    expect(pageWindow(20, 20)).toEqual([1, 18, 19, 20]);
  });
});

describe("csv", () => {
  it("quotes only when needed and writes CRLF lines", () => {
    expect(csvCell('a "b", c')).toBe('"a ""b"", c"');
    expect(csvCell(null)).toBe("");
    expect(csvCell(Number.NaN)).toBe("");
    expect(toCsv(["a", "b"], [[1, "x,y"], [null, true]])).toBe('a,b\r\n1,"x,y"\r\n,true\r\n');
  });
});
