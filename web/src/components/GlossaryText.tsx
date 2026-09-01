import { Fragment } from "react";
import { segmentGlossaryText, type GlossaryIndex } from "../glossary";

interface Props {
  text: string;
  glossaryIndex?: GlossaryIndex;
}

export function GlossaryText({ text, glossaryIndex }: Props) {
  if (!glossaryIndex) return text;

  return segmentGlossaryText(text, glossaryIndex).map((segment, index) =>
    segment.kind === "text" ? (
      <Fragment key={`${index}:${segment.text}`}>{segment.text}</Fragment>
    ) : (
      <dfn
        className="glossary-definition"
        data-definition={segment.plainDefinition}
        tabIndex={0}
        key={`${index}:${segment.term}`}
      >
        {segment.text}
      </dfn>
    ),
  );
}
