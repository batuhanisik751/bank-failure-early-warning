"use client";

import { useEffect } from "react";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <section role="alert" className="space-y-4">
      <h1 className="text-2xl font-bold tracking-tight">Something went wrong</h1>
      <p className="max-w-2xl text-muted">
        The page could not load its data. This is usually a database hiccup; trying again in a
        moment normally works.
      </p>
      {error.digest ? (
        <p className="text-xs text-muted">
          Reference <code className="font-mono">{error.digest}</code>
        </p>
      ) : null}
      <button
        type="button"
        onClick={reset}
        className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-fg"
      >
        Try again
      </button>
    </section>
  );
}
