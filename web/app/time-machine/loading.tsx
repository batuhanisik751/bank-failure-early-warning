export default function Loading() {
  return (
    <div role="status" aria-live="polite" className="min-h-screen space-y-4">
      <p className="sr-only">Loading</p>
      <div aria-hidden="true" className="h-8 w-56 animate-pulse rounded bg-surface" />
      <div aria-hidden="true" className="h-4 w-full max-w-2xl animate-pulse rounded bg-surface" />
      <div aria-hidden="true" className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-16 animate-pulse rounded-lg bg-surface" />
        ))}
      </div>
      <div aria-hidden="true" className="h-72 animate-pulse rounded-lg bg-surface" />
    </div>
  );
}
