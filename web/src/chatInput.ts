export interface TextSpan {
  start: number;
  end: number;
}

export function replaceTextSpan(current: string, span: TextSpan, replacement: string) {
  return {
    value: `${current.slice(0, span.start)}${replacement}${current.slice(span.end)}`,
    end: span.start + replacement.length,
  };
}
