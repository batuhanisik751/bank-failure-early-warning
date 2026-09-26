import type { Metadata } from "next";
import { SectionPlaceholder } from "@/components/SectionPlaceholder";
import { DISCLAIMER } from "@/lib/disclaimer";

export const metadata: Metadata = {
  title: "Map",
  description: "Every scored bank on a map, coloured by risk band, quarter by quarter. " + DISCLAIMER,
};

export default function Page() {
  return (
    <SectionPlaceholder title="Map">
      <p>Head-office locations of every scored bank, coloured and labelled by risk band, with the failures of each quarter marked.</p>
    </SectionPlaceholder>
  );
}
