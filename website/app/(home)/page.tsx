import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-6 py-24 text-center">
      <span className="flopscope-wordmark text-[40px]" aria-label="whestbench.">
        <span className="flopscope-wordmark__flop">whest</span>bench
        <span className="flopscope-wordmark__dot">.</span>
      </span>
      <p className="mt-6 max-w-xl text-lg text-fd-muted-foreground">
        White-box estimation of MLP output statistics under a FLOP budget.
        Library, CLI, and competition tooling for the ARC White-Box Estimation Challenge.
      </p>
      <div className="mt-10 flex gap-4">
        <Link
          href="/docs"
          className="rounded-md bg-fd-primary px-5 py-2.5 font-medium text-fd-primary-foreground"
        >
          Read the docs
        </Link>
        <Link
          href="/docs/participant-guide"
          className="rounded-md border px-5 py-2.5 font-medium"
        >
          Participant guide
        </Link>
      </div>
    </main>
  );
}
