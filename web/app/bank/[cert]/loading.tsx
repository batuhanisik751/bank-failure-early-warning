export default function Loading() {
  return (
    <div role="status" aria-live="polite" className="space-y-4">
      <p className="sr-only">Loading bank profile</p>
      <div aria-hidden="true" className="h-4 w-40 animate-pulse rounded bg-surface" />
      <div aria-hidden="true" className="h-9 w-80 max-w-full animate-pulse rounded bg-surface" />
      <div aria-hidden="true" className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-16 animate-pulse rounded-lg bg-surface" />
        ))}
      </div>
      <div aria-hidden="true" className="h-80 animate-pulse rounded-lg bg-surface" />
      <div aria-hidden="true" className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-44 animate-pulse rounded-lg bg-surface" />
        ))}
      </div>
    </div>
  );
}
