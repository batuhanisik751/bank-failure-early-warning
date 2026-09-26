import { describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

describe("stripTlsParams", () => {
  it("leaves a URL without a query string alone", async () => {
    const { stripTlsParams } = await import("@/lib/db/client");
    const url = "postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db";
    expect(stripTlsParams(url)).toEqual({ url, dropped: [] });
  });

  it("keeps parameters that are not about TLS", async () => {
    const { stripTlsParams } = await import("@/lib/db/client");
    const url = "postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db?application_name=bankcanary&channel_binding=require";
    expect(stripTlsParams(url)).toEqual({ url, dropped: [] });
  });

  it("drops every TLS parameter and names the dropped keys, never their values", async () => {
    const { stripTlsParams } = await import("@/lib/db/client");
    const out = stripTlsParams(
      "postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db?sslmode=disable&application_name=bankcanary&uselibpqcompat=true&ssl=false&sslrootcert=%2Fetc%2Fca.pem",
    );
    expect(out.url).toBe("postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db?application_name=bankcanary");
    expect(out.dropped).toEqual(["ssl", "sslmode", "sslrootcert", "uselibpqcompat"]);
    expect(out.dropped.join(" ")).not.toContain("disable");
  });

  it("removes the whole query string when only TLS parameters were in it", async () => {
    const { stripTlsParams } = await import("@/lib/db/client");
    expect(stripTlsParams("postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db?sslmode=verify-full").url).toBe(
      "postgresql://u:p@ep-x.us-east-2.aws.neon.tech/db",
    );
  });
});
