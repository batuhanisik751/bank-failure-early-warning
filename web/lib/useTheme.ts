"use client";

import { useCallback, useSyncExternalStore } from "react";

export type Theme = "light" | "dark";
const STORAGE_KEY = "theme";

function readTheme(): Theme {
  const stored = document.documentElement.dataset.theme;
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Notifies on a system preference change and on any change to <html data-theme>. */
function subscribe(onChange: () => void): () => void {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const observer = new MutationObserver(onChange);
  media.addEventListener("change", onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => {
    media.removeEventListener("change", onChange);
    observer.disconnect();
  };
}

/**
 * The theme in effect: the stored choice on <html data-theme>, else the system preference.
 * `theme` is null during server rendering and hydration so both sides agree.
 */
export function useTheme(): { theme: Theme | null; setTheme: (t: Theme) => void } {
  const theme = useSyncExternalStore<Theme | null>(subscribe, readTheme, () => null);

  const setTheme = useCallback((next: Theme) => {
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Storage may be unavailable; the attribute still applies for this page.
    }
  }, []);

  return { theme, setTheme };
}

/** Inline script for <head>: applies the stored choice before first paint. */
export const THEME_INIT_SCRIPT =
  `(function(){try{var t=localStorage.getItem("${STORAGE_KEY}");` +
  `if(t==="light"||t==="dark"){document.documentElement.setAttribute("data-theme",t)}}catch(e){}})();`;
