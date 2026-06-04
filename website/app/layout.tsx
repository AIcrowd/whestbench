import { Inter, Newsreader, Source_Serif_4, JetBrains_Mono } from 'next/font/google';
import { Provider } from '@/components/provider';
import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import 'katex/dist/katex.css';
import './global.css';
import { cn } from '@/lib/utils';
import { withBasePath } from '@/lib/base-path';

export const metadata: Metadata = {
  icons: {
    // Per design-system §03 size ladder, anything under 48px collapses
    // to the coral dot — the SVG scales cleanly for all favicon sizes.
    icon: [
      { url: withBasePath('/favicon.svg'), type: 'image/svg+xml' },
      { url: withBasePath('/logo.png'), type: 'image/png' },
    ],
    apple: withBasePath('/logo.png'),
  },
};

// Four type registers from the Flopscope Design System.
// - App (UI chrome, buttons, dense panels):      Inter
// - Editorial / display (wordmark, h1–h3):        Newsreader (variable opsz 6–72)
// - Paper (long-form body prose ≥ 15px):          Source Serif 4 (variable opsz 8–60)
// - Code (blocks, inline, formulas):              JetBrains Mono
const inter = Inter({
  subsets: ['latin'],
  variable: '--font-app-sans',
  display: 'swap',
});

const newsreader = Newsreader({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  style: ['normal', 'italic'],
  variable: '--font-display-serif',
  display: 'swap',
});

const sourceSerif = Source_Serif_4({
  subsets: ['latin'],
  weight: ['400', '600', '700'],
  style: ['normal', 'italic'],
  variable: '--font-paper-serif',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '700'],
  style: ['normal', 'italic'],
  variable: '--font-mono',
  display: 'swap',
});

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={cn(
        inter.className,
        'font-sans',
        inter.variable,
        newsreader.variable,
        sourceSerif.variable,
        jetbrainsMono.variable,
      )}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
