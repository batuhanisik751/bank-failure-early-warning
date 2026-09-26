"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapQuarter, MapTimelineEntry } from "@/lib/queries/map";
import { FailureList } from "./FailureList";
import { MapCanvas } from "./MapCanvas";

/** Time per quarter while playing. */
export const FRAME_MS = 1000;

type Props = { timeline: MapTimelineEntry[]; initial: MapQuarter };

async function fetchQuarter(label: string): Promise<MapQuarter> {
  const res = await fetch(`/api/map/${label}`);
  if (!res.ok) throw new Error(`Map data for ${label} could not be loaded (HTTP ${res.status}).`);
  return (await res.json()) as MapQuarter;
}

/**
 * Play, pause and scrub through every scored quarter. Each quarter is fetched once from
 * /api/map/[quarter] and kept in memory; the next quarter is prefetched so playback does
 * not stall, and playback waits for a quarter that has not arrived instead of skipping it.
 */
export function FailureReplayMap({ timeline, initial }: Props) {
  const labels = useMemo(() => timeline.map((t) => t.label), [timeline]);
  const [index, setIndex] = useState(() => Math.max(0, labels.indexOf(initial.quarter)));
  const [playing, setPlaying] = useState(false);
  const [shown, setShown] = useState<MapQuarter>(initial);
  const [error, setError] = useState<string | null>(null);
  const cache = useRef(new Map<string, MapQuarter>([[initial.quarter, initial]]));
  const inflight = useRef(new Map<string, Promise<MapQuarter>>());
  const indexRef = useRef(index);
  useEffect(() => {
    indexRef.current = index;
  }, [index]);

  const load = useCallback((label: string): Promise<MapQuarter> => {
    const hit = cache.current.get(label);
    if (hit) return Promise.resolve(hit);
    let pending = inflight.current.get(label);
    if (!pending) {
      pending = fetchQuarter(label).then(
        (data) => {
          cache.current.set(label, data);
          inflight.current.delete(label);
          return data;
        },
        (err: unknown) => {
          inflight.current.delete(label);
          throw err;
        },
      );
      inflight.current.set(label, pending);
    }
    return pending;
  }, []);

  // Show the selected quarter, prefetch the one after it, and keep the URL shareable.
  useEffect(() => {
    const label = labels[index];
    if (!label) return;
    let cancelled = false;
    load(label).then(
      (data) => {
        if (cancelled) return;
        setShown(data);
        setError(null);
      },
      (err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setPlaying(false);
      },
    );
    const next = labels[index + 1];
    if (next) load(next).catch(() => undefined);
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("quarter", label);
      window.history.replaceState(null, "", url);
    } catch {
      // The URL is a convenience; the page works without it.
    }
    return () => {
      cancelled = true;
    };
  }, [index, labels, load]);

  // Playback: one quarter per frame, only once that quarter's data is in hand; stops at the end.
  useEffect(() => {
    if (!playing) return;
    if (indexRef.current >= labels.length - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setInterval(() => {
      const next = labels[indexRef.current + 1];
      if (!next) {
        setPlaying(false);
        return;
      }
      if (cache.current.has(next)) setIndex(indexRef.current + 1);
      else load(next).catch(() => setPlaying(false));
    }, FRAME_MS);
    return () => window.clearInterval(id);
  }, [playing, labels, load]);

  const label = labels[index] ?? initial.quarter;
  const entry = timeline[index];
  const loading = shown.quarter !== label;
  const last = index >= labels.length - 1;
  const step = (delta: number) => {
    setPlaying(false);
    setIndex((i) => Math.min(labels.length - 1, Math.max(0, i + delta)));
  };
  const button =
    "rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-fg hover:border-accent disabled:cursor-not-allowed disabled:opacity-50";
  return (
    <div className="space-y-4">
      <div role="group" aria-label="Playback controls" className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={() => setPlaying((p) => !p)} aria-pressed={playing} disabled={last && !playing} className={`${button} min-w-[5rem] bg-accent text-accent-fg`}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={() => step(-1)} disabled={index === 0} className={button} aria-label="Previous quarter">
          <span aria-hidden="true">◀</span>
        </button>
        <button type="button" onClick={() => step(1)} disabled={last} className={button} aria-label="Next quarter">
          <span aria-hidden="true">▶</span>
        </button>
        <label className="flex min-w-[14rem] flex-1 items-center gap-3 text-sm">
          <span>Quarter</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, labels.length - 1)}
            value={index}
            onChange={(e) => {
              setPlaying(false);
              setIndex(Number(e.target.value));
            }}
            aria-valuetext={label}
            className="w-full accent-[var(--accent)]"
            data-testid="quarter-slider"
          />
        </label>
        <output className="font-mono text-lg font-semibold tabular-nums" data-testid="map-quarter">
          {label}
        </output>
      </div>
      <p aria-live={playing ? "off" : "polite"} className="text-sm text-muted" data-testid="map-status">
        {label}: {entry?.nBanks ?? shown.dots.length} banks on the map, {entry?.nFailures ?? 0} failed in the following quarter
        {loading ? " (loading)" : ""}
        {last ? ". The quarter after the latest report has not ended, so its failures may still be added" : ""}.
        {" "}Model version <code className="font-mono text-fg" data-testid="map-model-version">{entry?.modelVersion ?? "unknown"}</code>.
      </p>
      {error ? (
        <p role="alert" className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-fg">
          {error} The last quarter loaded stays on the map.
        </p>
      ) : null}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <MapCanvas quarter={shown.quarter} dots={shown.dots} />
        <FailureList quarter={shown.quarter} dots={shown.dots} />
      </div>
    </div>
  );
}
