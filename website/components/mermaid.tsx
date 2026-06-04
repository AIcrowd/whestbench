'use client';

import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({ startOnLoad: false, theme: 'neutral' });

export default function Mermaid({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState('');

  useEffect(() => {
    const id = `mermaid-${Math.random().toString(36).slice(2)}`;
    mermaid.render(id, chart).then(({ svg }) => setSvg(svg));
  }, [chart]);

  return <div ref={ref} dangerouslySetInnerHTML={{ __html: svg }} />;
}
