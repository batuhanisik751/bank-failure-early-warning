import type { Metadata } from "next";
import { SectionPlaceholder } from "@/components/SectionPlaceholder";
import { DISCLAIMER } from "@/lib/disclaimer";

export const metadata: Metadata = {
  title: "Time machine",
  description: "What the model said before each past quarter, with hindsight about which banks failed. " + DISCLAIMER,
};

export default function Page() {
  return (
    <SectionPlaceholder title="Time machine">
      <p>Pick any quarter since 2008 and see the ranking the walk-forward model of that year produced at the time, next to the failures that followed within four quarters.</p>
    </SectionPlaceholder>
  );
}
