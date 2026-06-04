import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

// whestbench wordmark — lowercase, with `whest` highlighted in coral and a
// permanent coral period (mirrors flopscope's wordmark treatment; coral is the
// shared design-system foundation brand).
function Wordmark() {
  return (
    <span className="flopscope-wordmark text-[22px]" aria-label="whestbench.">
      <span className="flopscope-wordmark__flop">whest</span>bench
      <span className="flopscope-wordmark__dot">.</span>
    </span>
  );
}

export function baseOptions(): BaseLayoutProps {
  return {
    nav: { title: <Wordmark /> },
    githubUrl: 'https://github.com/AIcrowd/whestbench',
  };
}
