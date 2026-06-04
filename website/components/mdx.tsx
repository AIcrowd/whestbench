import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps } from 'react';
import { Accordions, Accordion } from 'fumadocs-ui/components/accordion';
import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import ApiReference from './api-reference';
import ApiNamespaceHub from './api-reference/ApiNamespaceHub';
import NamespaceInventory from './api-reference/NamespaceInventory';
import SortableTable from './shared/SortableTable';
import Mermaid from './mermaid';
import StaticFileLink from './static-file-link';
import { Callout } from './ui/callout';

function DocsPre(props: ComponentProps<'pre'>) {
  return (
    <CodeBlock {...props}>
      <Pre>{props.children}</Pre>
    </CodeBlock>
  );
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    pre: DocsPre,
    ApiReference,
    ApiNamespaceHub,
    NamespaceInventory,
    OperationCostIndex: ApiReference,
    Callout,
    SortableTable,
    Mermaid,
    Accordions,
    Accordion,
    StaticFileLink,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
