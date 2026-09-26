"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export const SECTIONS = [
  { href: "/", label: "Leaderboard" },
  { href: "/time-machine", label: "Time machine" },
  { href: "/map", label: "Map" },
  { href: "/case-study-2023", label: "2023 case study" },
  { href: "/rate-shock", label: "Rate shock" },
  { href: "/methodology", label: "Methodology" },
] as const;

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/" || pathname.startsWith("/bank/");
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function SiteNav() {
  const pathname = usePathname() ?? "/";
  return (
    <nav aria-label="Sections" className="-mx-4 overflow-x-auto px-4">
      <ul className="flex gap-1 whitespace-nowrap text-sm">
        {SECTIONS.map(({ href, label }) => {
          const active = isActive(pathname, href);
          return (
            <li key={href}>
              <Link
                href={href}
                aria-current={active ? "page" : undefined}
                className={
                  "inline-block rounded-md px-3 py-1.5 font-medium no-underline hover:bg-surface hover:no-underline " +
                  (active ? "bg-surface text-fg shadow-sm ring-1 ring-border" : "text-muted")
                }
              >
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
