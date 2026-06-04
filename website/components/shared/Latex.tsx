import katex from 'katex';
import 'katex/dist/katex.min.css';

export function stripMathDelimiters(math: string): string {
  return math.replace(/^\$+|\$+$/g, '');
}

interface LatexProps {
  math: string;
  display?: boolean;
}

export default function Latex({math, display = false}: LatexProps) {
  const html = katex.renderToString(math, {
    displayMode: display,
    throwOnError: false,
    trust: false,
  });

  return display ? (
    <div dangerouslySetInnerHTML={{__html: html}} />
  ) : (
    <span dangerouslySetInnerHTML={{__html: html}} />
  );
}
