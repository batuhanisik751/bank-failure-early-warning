import type { Metadata } from "next";
import { SectionPlaceholder } from "@/components/SectionPlaceholder";
import { DISCLAIMER } from "@/lib/disclaimer";

export const metadata: Metadata = {
  title: "2023 case study",
  description: "Silicon Valley Bank, Signature and First Republic under a credit-only and a rate-aware model. " + DISCLAIMER,
};

export default function Page() {
  return (
    <SectionPlaceholder title="2023 case study">
      <p>How the credit-only and the rate-aware model ranked Silicon Valley Bank, Signature Bank and First Republic Bank in the quarters before March 2023.</p>
    </SectionPlaceholder>
  );
}
