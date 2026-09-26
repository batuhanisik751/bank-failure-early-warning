import type { Metadata } from "next";

export const SITE_NAME = "BankCanary";

/** The public origin; deployments set NEXT_PUBLIC_SITE_URL, local runs fall back to the dev port. */
export function siteUrl(): string {
  const raw = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3100";
  return raw.replace(/\/+$/, "");
}

/** An absolute URL for a site path such as "/bank/14" or "/time-machine?quarter=2023Q1". */
export function absoluteUrl(path: string): string {
  return new URL(path, `${siteUrl()}/`).toString();
}

export type PageMeta = {
  title: string;
  description: string;
  /** Canonical path for this page; filtered or duplicated views name their canonical parent. */
  path: string;
  /** false marks the page noindex,follow (filtered views, error states). */
  index?: boolean;
};

/**
 * Per-page metadata with the pieces every page needs: an absolute title with the site name,
 * description, canonical link, Open Graph and robots. The metadataBase in app/layout.tsx
 * resolves the relative URLs.
 */
export function pageMetadata({ title, description, path, index = true }: PageMeta): Metadata {
  return {
    // Absolute: the root segment (app/page.tsx) is not a child of the layout, so its
    // title.template would not apply there; every page names the site the same way.
    title: { absolute: `${title} | ${SITE_NAME}` },
    description,
    alternates: { canonical: path },
    openGraph: {
      title: `${title} | ${SITE_NAME}`,
      description,
      url: path,
      siteName: SITE_NAME,
      type: "website",
      locale: "en_US",
    },
    twitter: { card: "summary", title: `${title} | ${SITE_NAME}`, description },
    robots: index ? { index: true, follow: true } : { index: false, follow: true },
  };
}
