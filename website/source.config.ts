import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { whestbenchLight, whestbenchDark } from './lib/shiki-themes';

export const docs = defineDocs({
  dir: 'content/docs',
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    rehypePlugins: (v) => [rehypeKatex, ...v],
    rehypeCodeOptions: {
      // Shiki accepts raw TextMate themes (`tokenColors`-shaped); TS's union
      // defaults to the stricter `ThemeRegistrationResolved` branch which
      // expects `settings` instead. The raw shape works at runtime.
      themes: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        light: whestbenchLight as any,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        dark: whestbenchDark as any,
      },
    },
  },
});
