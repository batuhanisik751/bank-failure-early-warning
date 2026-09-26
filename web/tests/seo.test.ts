import { afterEach, describe, expect, it } from "vitest";
import { absoluteUrl, pageMetadata, siteUrl } from "@/lib/seo";

const original = process.env.NEXT_PUBLIC_SITE_URL;

afterEach(() => {
  if (original === undefined) delete process.env.NEXT_PUBLIC_SITE_URL;
  else process.env.NEXT_PUBLIC_SITE_URL = original;
});

describe("siteUrl and absoluteUrl", () => {
  it("falls back to the local port and strips a trailing slash", () => {
    delete process.env.NEXT_PUBLIC_SITE_URL;
    expect(siteUrl()).toBe("http://localhost:3100");
    process.env.NEXT_PUBLIC_SITE_URL = "https://example.org/";
    expect(siteUrl()).toBe("https://example.org");
    expect(absoluteUrl("/bank/14")).toBe("https://example.org/bank/14");
    expect(absoluteUrl("/time-machine?quarter=2023Q1")).toBe("https://example.org/time-machine?quarter=2023Q1");
  });
});

describe("pageMetadata", () => {
  it("fills title, description, canonical, Open Graph, Twitter and robots", () => {
    const m = pageMetadata({ title: "Leaderboard", description: "Ranked banks.", path: "/" });
    expect(m.title).toEqual({ absolute: "Leaderboard | BankCanary" });
    expect(m.description).toBe("Ranked banks.");
    expect(m.alternates).toEqual({ canonical: "/" });
    expect(m.openGraph).toMatchObject({
      title: "Leaderboard | BankCanary",
      description: "Ranked banks.",
      url: "/",
      siteName: "BankCanary",
      type: "website",
    });
    expect(m.twitter).toMatchObject({ card: "summary", title: "Leaderboard | BankCanary" });
    expect(m.robots).toEqual({ index: true, follow: true });
  });

  it("marks duplicate or missing views noindex but keeps following links", () => {
    const m = pageMetadata({ title: "Bank not found", description: "x", path: "/bank/0", index: false });
    expect(m.robots).toEqual({ index: false, follow: true });
    expect(m.alternates).toEqual({ canonical: "/bank/0" });
  });
});
