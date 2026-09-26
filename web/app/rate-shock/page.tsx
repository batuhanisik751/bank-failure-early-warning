import type { Metadata } from "next";
import { SectionPlaceholder } from "@/components/SectionPlaceholder";
import { DISCLAIMER } from "@/lib/disclaimer";

export const metadata: Metadata = {
  title: "Rate shock",
  description: "The latest quarter re-scored under parallel interest-rate shocks. " + DISCLAIMER,
};

export default function Page() {
  return (
    <SectionPlaceholder title="Rate shock">
      <p>Re-score the latest quarter under a 100 to 400 basis point parallel shock across securities durations of two to six years and watch the bands move.</p>
    </SectionPlaceholder>
  );
}
