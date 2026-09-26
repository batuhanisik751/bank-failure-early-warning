import type { MetadataRoute } from "next";
import { absoluteUrl } from "@/lib/seo";

/** Pages are crawlable; the JSON, CSV and revalidation endpoints are not. */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: "*", allow: "/", disallow: ["/api/"] }],
    sitemap: absoluteUrl("/sitemap.xml"),
  };
}
