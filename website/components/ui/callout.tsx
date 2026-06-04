import * as React from 'react';
import { cn } from '@/lib/utils';

type CalloutVariant = 'default' | 'accent';

export interface CalloutProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: string;
  variant?: CalloutVariant;
}

export function Callout({
  label,
  variant = 'default',
  className,
  children,
  ...props
}: CalloutProps) {
  return (
    <div
      data-variant={variant}
      className={cn(
        'not-prose my-6 rounded-[var(--radius-xl)] px-5 py-4 text-[13px] leading-[1.6]',
        variant === 'default'
          ? 'border border-[var(--gray-200)] bg-[var(--gray-50)] text-[var(--gray-900)]'
          : 'border bg-[var(--coral-light)] text-[var(--gray-900)] [border-color:color-mix(in_oklab,var(--coral)_25%,transparent)]',
        className,
      )}
      {...props}
    >
      {label ? (
        <div
          className={cn(
            'mb-2 font-sans text-[10px] font-semibold uppercase',
            variant === 'accent' ? 'text-[var(--coral)]' : 'text-[var(--gray-400)]',
          )}
          style={{ letterSpacing: 'var(--tracking-kicker-loose)' }}
        >
          {label}
        </div>
      ) : null}
      {children}
    </div>
  );
}
