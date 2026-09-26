"use client";

import { useTheme } from "@/lib/useTheme";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={theme ? `Switch to ${next} theme` : "Switch theme"}
      aria-pressed={theme === "dark"}
      className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-fg hover:border-accent"
    >
      <span aria-hidden="true" className="mr-1">
        {theme === "dark" ? "☽" : "☀"}
      </span>
      {theme === "dark" ? "Dark" : "Light"}
    </button>
  );
}
