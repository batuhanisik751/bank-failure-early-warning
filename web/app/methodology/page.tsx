import type { Metadata } from "next";
import { SectionPlaceholder } from "@/components/SectionPlaceholder";
import { DISCLAIMER } from "@/lib/disclaimer";

export const metadata: Metadata = {
  title: "Methodology",
  description: "How the model is trained, validated and calibrated, and where it fails. " + DISCLAIMER,
};

export default function Page() {
  return (
    <SectionPlaceholder title="Methodology">
      <p>Walk-forward training, calibration, the metrics per test year and the limits of the approach, in one place.</p>
    </SectionPlaceholder>
  );
}
