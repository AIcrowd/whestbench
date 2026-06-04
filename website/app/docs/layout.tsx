import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';
import type { ReactNode } from 'react';
import { source } from '@/lib/source';
import DocsSidebarItem from '@/components/docs/DocsSidebarItem';

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout
      tree={source.getPageTree()}
      sidebar={{ components: { Item: DocsSidebarItem } }}
      {...baseOptions()}
    >
      {children}
    </DocsLayout>
  );
}
