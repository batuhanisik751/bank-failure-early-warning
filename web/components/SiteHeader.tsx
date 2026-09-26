import Link from "next/link";
import { SiteNav } from "./SiteNav";
import { ThemeToggle } from "./ThemeToggle";

export function SiteHeader() {
  return (
    <header className="border-b border-border bg-bg">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center justify-between gap-4">
          <Link href="/" className="text-lg font-bold tracking-tight text-fg no-underline hover:no-underline">
            <span aria-hidden="true" className="mr-2 inline-block h-3 w-3 rounded-full bg-accent align-middle" />
            BankCanary
          </Link>
          <div className="sm:hidden">
            <ThemeToggle />
          </div>
        </div>
        <SiteNav />
        <div className="hidden sm:block">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
