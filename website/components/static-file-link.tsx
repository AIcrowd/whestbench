import type { AnchorHTMLAttributes, ReactNode } from 'react';
import { withBasePath } from '@/lib/base-path';

type StaticFileLinkProps = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> & {
  href: string;
  children: ReactNode;
};

export default function StaticFileLink({
  href,
  children,
  ...props
}: StaticFileLinkProps) {
  return (
    <a href={withBasePath(href)} {...props}>
      {children}
    </a>
  );
}
