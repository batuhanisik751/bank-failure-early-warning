import type { MetadataRoute } from "next";
import { quarters } from "@/lib/queries/quarters";
import { scoredCerts } from "@/lib/queries/sitemap";
import { scoredQuarters } from "@/lib/queries/timeMachine";
import { absoluteUrl } from "@/lib/seo";

export const dynamic = "force-dynamic";

/**
 * Every indexable page: the six sections, one time-machine entry per scored quarter and one
 * profile per bank the production model has scored (about 9,000 URLs, well under the 50,000
 * limit of a single sitemap). Reads the same cached queries as the pages.
 */
export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  // With the database unreachable the six sections still list; the per-bank and per-quarter
  // entries come back once it does.
  const [published, scored, certs] = await Promise.all([quarters(), scoredQuarters(), scoredCerts()]).catch(
    (error: unknown) => {
      console.error("sitemap: database unavailable", error instanceof Error ? error.message : error);
      return [[], [], []] as const;
    },
  );
  const latest = published[0];
  const lastModified = latest?.availDate ? new Date(latest.availDate) : new Date();
  const sections: MetadataRoute.Sitemap = [
    { url: absoluteUrl("/"), lastModified, changeFrequency: "weekly", priority: 1 },
    { url: absoluteUrl("/time-machine"), lastModified, changeFrequency: "weekly", priority: 0.8 },
    { url: absoluteUrl("/map"), lastModified, changeFrequency: "weekly", priority: 0.7 },
    { url: absoluteUrl("/case-study-2023"), lastModified, changeFrequency: "yearly", priority: 0.6 },
    { url: absoluteUrl("/rate-shock"), lastModified, changeFrequency: "weekly", priority: 0.6 },
    { url: absoluteUrl("/methodology"), lastModified, changeFrequency: "monthly", priority: 0.7 },
  ];
  const quarterPages: MetadataRoute.Sitemap = scored.map((q) => ({
    url: absoluteUrl(`/time-machine?quarter=${q.label}`),
    lastModified,
    changeFrequency: "yearly",
    priority: 0.3,
  }));
  const bankPages: MetadataRoute.Sitemap = certs.map((cert) => ({
    url: absoluteUrl(`/bank/${cert}`),
    lastModified,
    changeFrequency: "weekly",
    priority: 0.5,
  }));
  return [...sections, ...quarterPages, ...bankPages];
}
