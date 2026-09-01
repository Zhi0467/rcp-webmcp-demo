export type ScreenStoryKind = "film_ip" | "series";
export type EstimateConfidence = "measured" | "high" | "medium";

export interface ScreenStoryComparison {
  id: string;
  label: string;
  kind: ScreenStoryKind;
  estimatedScriptWords: number;
  confidence: EstimateConfidence;
  basis: string;
  sources: readonly string[];
}

const ENGLISH_TOKENS_PER_WORD = 4 / 3;

/**
 * One row per complete screen-story unit. No transcript text is bundled.
 * Rounded estimates deliberately trade false precision for a stable, playful
 * comparison; `basis`, `confidence`, and `sources` keep every row auditable.
 */
export const SCREEN_STORY_COMPARISONS: readonly ScreenStoryComparison[] = [
  {
    id: "before-trilogy",
    label: "the Before trilogy",
    kind: "film_ip",
    estimatedScriptWords: 48_000,
    confidence: "high",
    basis: "Two measured full screenplays plus the published third screenplay length.",
    sources: [
      "https://www.story24.film/screenplays/before-sunrise.pdf",
      "https://screenplays.io/screenplay/before-sunset",
      "https://assets.scriptslug.com/live/pdf/scripts/before-midnight-2013.pdf",
    ],
  },
  {
    id: "back-to-the-future-trilogy",
    label: "the Back to the Future trilogy",
    kind: "film_ip",
    estimatedScriptWords: 74_000,
    confidence: "medium",
    basis: "Measured first-film screenplay, scaled with the two sequel screenplay lengths.",
    sources: [
      "https://www.scifiscripts.com/scripts/backtothefuture_script.pdf",
      "https://8flix.com/scripts/film/back-to-the-future-part-2-1989-screenplay/",
      "https://8flix.com/scripts/film/back-to-the-future-part-3-1990-screenplay/",
    ],
  },
  {
    id: "lord-of-the-rings-trilogy",
    label: "the Lord of the Rings trilogy",
    kind: "film_ip",
    estimatedScriptWords: 98_716,
    confidence: "measured",
    basis: "Full screenplay text measured across all three films.",
    sources: [
      "https://assets.scriptslug.com/live/pdf/scripts/the-lord-of-the-rings-the-fellowship-of-the-ring-2001.pdf",
      "https://assets.scriptslug.com/live/pdf/scripts/the-lord-of-the-rings-the-two-towers-2002.pdf",
      "https://assets.scriptslug.com/live/pdf/scripts/the-lord-of-the-rings-the-3-return-of-the-king-2003.pdf",
    ],
  },
  {
    id: "stranger-things",
    label: "Stranger Things",
    kind: "series",
    estimatedScriptWords: 562_500,
    confidence: "medium",
    basis:
      "Complete-series estimate from transcript volume and measured one-hour teleplay expansion.",
    sources: ["https://github.com/filmicaesthetic/stringr-things"],
  },
  {
    id: "breaking-bad",
    label: "Breaking Bad",
    kind: "series",
    estimatedScriptWords: 787_500,
    confidence: "medium",
    basis:
      "Complete transcript corpus expanded by the measured pilot teleplay-to-transcript ratio.",
    sources: [
      "https://www.springfieldspringfield.co.uk/episode_scripts.php?tv-show=breaking-bad",
      "https://kinodramaturg.ru/wp-content/uploads/2014/09/Breaking-Bad-pilot-script.pdf",
    ],
  },
  {
    id: "better-call-saul",
    label: "Better Call Saul",
    kind: "series",
    estimatedScriptWords: 810_000,
    confidence: "medium",
    basis: "Complete transcript corpus expanded with the one-hour teleplay profile.",
    sources: [
      "https://www.springfieldspringfield.co.uk/episode_scripts.php?tv-show=better-call-saul-2015",
      "https://nofilmschool.com/read-and-download-better-call-saul-pilot-script-plus-other-episodes",
    ],
  },
  {
    id: "big-bang-theory",
    label: "The Big Bang Theory",
    kind: "series",
    estimatedScriptWords: 1_087_500,
    confidence: "medium",
    basis:
      "Complete measured transcript corpus expanded with a dialogue-heavy sitcom script profile.",
    sources: [
      "https://www.springfieldspringfield.co.uk/episode_scripts.php?tv-show=big-bang-theory",
      "https://pmc.ncbi.nlm.nih.gov/articles/PMC6874063/",
    ],
  },
  {
    id: "the-simpsons",
    label: "The Simpsons",
    kind: "series",
    estimatedScriptWords: 3_412_500,
    confidence: "medium",
    basis: "Series transcript corpus expanded with an animation screenplay profile.",
    sources: ["https://github.com/gastonstat/simpsons-transcripts"],
  },
];

export function estimatedScreenplayTokens(comparison: ScreenStoryComparison): number {
  return Math.round(comparison.estimatedScriptWords * ENGLISH_TOKENS_PER_WORD);
}

export function projectUsageTokens(processedInput: number, generated: number): number {
  return Math.max(0, processedInput) + Math.max(0, generated);
}

export function pickScreenStoryComparison(
  random: () => number = Math.random,
): ScreenStoryComparison {
  const index = Math.min(
    SCREEN_STORY_COMPARISONS.length - 1,
    Math.floor(Math.max(0, random()) * SCREEN_STORY_COMPARISONS.length),
  );
  return SCREEN_STORY_COMPARISONS[index];
}

export function screenStoryComparisonCopy(
  projectTokens: number,
  comparison: ScreenStoryComparison,
): string | null {
  if (projectTokens <= 0) return null;
  const ratio = projectTokens / estimatedScreenplayTokens(comparison);
  if (ratio < 1) {
    const percent = ratio * 100;
    const formatted = percent < 1 ? "<1%" : `${Math.round(percent)}%`;
    return `This project has used about ${formatted} as many tokens as the scripts for ${comparison.label}.`;
  }
  const formatted =
    ratio <= 10
      ? `${ratio.toFixed(1)}×`
      : `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(ratio)}×`;
  return `This project has used about ${formatted} as many tokens as the scripts for ${comparison.label}.`;
}
