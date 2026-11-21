"use client";

import { useState } from "react";
import Link from "next/link";

import CitationExplorer from "@/components/citation-explorer";
import RetrievalXRay from "@/components/retrieval-xray";

import {
  askQuestion,
  type AskResponse,
  type RetrievalMode,
} from "@/lib/api";

const DEMO_DOCUMENT_ID =
  "24b26eb7b9bba731af9746974c452a23f870814d31b7d55cd681b771dfe02fdc";

const retrievalModes: {
  value: RetrievalMode;
  label: string;
  description: string;
}[] = [
  {
    value: "dense",
    label: "Dense",
    description: "Semantic similarity search",
  },
  {
    value: "bm25",
    label: "BM25",
    description: "Keyword-based retrieval",
  },
  {
    value: "hybrid",
    label: "Hybrid",
    description: "Dense + BM25 with RRF fusion",
  },
];

const exampleQuestions = [
  "What is the primary purpose of the Common Pile?",
  "What types of data sources are included in the Common Pile?",
  "Why is copyright licensing important for LLM training datasets?",
  "Which language models were trained using the Common Pile?",
];

export default function Home() {
  const [question, setQuestion] = useState(
    exampleQuestions[0]
  );

  const [mode, setMode] = useState<RetrievalMode>("hybrid");

  const [result, setResult] = useState<AskResponse | null>(
    null
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!question.trim() || loading) {
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await askQuestion(
        question.trim(),
        mode,
        false,
        DEMO_DOCUMENT_ID
      );

      setResult(response);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to retrieve an answer."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100 sm:px-8">
      <div className="mx-auto max-w-6xl space-y-8">

        {/* Header */}
        <header className="flex flex-wrap items-start justify-between gap-5 border-b border-slate-800 pb-7">
          <div>
            <div className="mb-3 flex items-center gap-2">
              <span className="rounded-md bg-blue-500/15 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-blue-400">
                Explainable AI
              </span>

              <span className="text-xs text-slate-500">
                Public Demo
              </span>
            </div>

            <h1 className="text-4xl font-bold tracking-tight text-white">
              DocuMind
            </h1>

            <p className="mt-3 max-w-xl text-sm leading-6 text-slate-400">
              An explainable, evaluation-driven Retrieval-Augmented
              Generation platform with hybrid search, source-grounded
              answers, and transparent retrieval diagnostics.
            </p>
          </div>

          {/* Evaluation navigation */}
          <Link
            href="/evaluation"
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-blue-500/40 bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-400"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M3 3v18h18" />
              <path d="m7 16 4-5 4 3 5-7" />
            </svg>

            View Evaluations

            <span aria-hidden="true">↗</span>
          </Link>
        </header>

        {/* Demo document */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="rounded-lg bg-blue-500/10 p-3 text-blue-400">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
                  <path d="M14 2v6h6" />
                  <path d="M8 13h8" />
                  <path d="M8 17h8" />
                </svg>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Indexed Research Document
                </p>

                <h2 className="mt-1 text-lg font-semibold text-white">
                  The Common Pile
                </h2>

                <p className="mt-1 text-sm text-slate-400">
                  55 pages · 276 indexed chunks
                </p>

                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                  Explore a research paper describing an 8TB dataset
                  of public-domain and openly licensed text for
                  language model training.
                </p>
              </div>
            </div>

            <span className="inline-flex items-center gap-2 rounded-full border border-emerald-800/50 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-400">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              Pre-indexed
            </span>
          </div>
        </section>

        {/* Ask DocuMind */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 sm:p-7">
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-white">
              Ask DocuMind
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Ask questions about the indexed paper and inspect
              the evidence behind each answer.
            </p>
          </div>

          <form onSubmit={handleAsk} className="space-y-5">
            <div>
              <label
                htmlFor="question"
                className="mb-2 block text-sm font-medium text-slate-300"
              >
                Your question
              </label>

              <textarea
                id="question"
                value={question}
                onChange={(event) =>
                  setQuestion(event.target.value)
                }
                maxLength={500}
                rows={3}
                placeholder="Ask a question about the Common Pile..."
                className="w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm leading-6 text-white outline-none transition-colors placeholder:text-slate-600 focus:border-blue-500"
              />

              <p className="mt-1 text-right text-xs text-slate-500">
                {question.length}/500
              </p>
            </div>

            {/* Example questions */}
            <div>
              <p className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
                Example questions
              </p>

              <div className="flex flex-wrap gap-2">
                {exampleQuestions.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => setQuestion(example)}
                    className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-left text-xs text-slate-300 transition-colors hover:border-blue-500/50 hover:text-blue-300"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            {/* Retrieval mode */}
            <div>
              <p className="mb-3 text-sm font-medium text-slate-300">
                Retrieval strategy
              </p>

              <div className="grid gap-3 sm:grid-cols-3">
                {retrievalModes.map((retrievalMode) => {
                  const active = mode === retrievalMode.value;

                  return (
                    <button
                      key={retrievalMode.value}
                      type="button"
                      aria-pressed={active}
                      onClick={() =>
                        setMode(retrievalMode.value)
                      }
                      className={`rounded-lg border p-4 text-left transition-colors ${
                        active
                          ? "border-blue-500 bg-blue-500/10"
                          : "border-slate-700 bg-slate-950 hover:border-slate-500"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={`h-3 w-3 rounded-full border ${
                            active
                              ? "border-blue-400 bg-blue-400"
                              : "border-slate-500"
                          }`}
                        />

                        <span className="text-sm font-semibold">
                          {retrievalMode.label}
                        </span>
                      </div>

                      <p className="mt-2 text-xs leading-5 text-slate-400">
                        {retrievalMode.description}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Submit */}
            <div className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-800 pt-5">
              <p className="text-xs text-slate-500">
                Answers are generated using retrieved evidence.
                Cross-encoder reranking and claim verification
                are available in the full local version.
              </p>

              <button
                type="submit"
                disabled={loading || !question.trim()}
                className="inline-flex min-w-36 items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Thinking...
                  </>
                ) : (
                  <>
                    Ask DocuMind
                    <span aria-hidden="true">→</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </section>

        {/* Error */}
        {error && (
          <div
            role="alert"
            className="rounded-xl border border-red-800/50 bg-red-500/10 p-5 text-sm text-red-300"
          >
            <p className="font-semibold">
              Unable to complete your request
            </p>

            <p className="mt-2 whitespace-pre-wrap break-words">
              {error}
            </p>

            <p className="mt-3 text-xs text-red-300/70">
              The free cloud backend may need time to wake up,
              or an API quota may have been reached.
            </p>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-8">

            {/* Citation-grounded answer */}
            <CitationExplorer
              key={`${question}-${result.answer}`}
              answer={result.answer}
              sources={result.sources}
            />

            {/* Retrieval diagnostics */}
            {result.retrieval_trace && (
              <RetrievalXRay
                trace={result.retrieval_trace}
              />
            )}

            {/* Verification results, if available */}
            {result.verification && (
              <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
                <h2 className="text-xl font-semibold">
                  Evidence Verification
                </h2>

                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-lg bg-slate-900 p-4">
                    <p className="text-xs text-slate-400">
                      Supported
                    </p>
                    <p className="mt-1 text-2xl font-semibold text-emerald-400">
                      {result.verification.supported_count}
                    </p>
                  </div>

                  <div className="rounded-lg bg-slate-900 p-4">
                    <p className="text-xs text-slate-400">
                      Contradicted
                    </p>
                    <p className="mt-1 text-2xl font-semibold text-red-400">
                      {result.verification.contradicted_count}
                    </p>
                  </div>

                  <div className="rounded-lg bg-slate-900 p-4">
                    <p className="text-xs text-slate-400">
                      Insufficient evidence
                    </p>
                    <p className="mt-1 text-2xl font-semibold text-amber-400">
                      {result.verification.insufficient_count}
                    </p>
                  </div>
                </div>

                <div className="mt-5 space-y-3">
                  {result.verification.claims.map(
                    (claim, index) => (
                      <div
                        key={index}
                        className="rounded-lg border border-slate-800 bg-slate-950 p-4"
                      >
                        <p className="text-sm font-medium text-slate-200">
                          {claim.claim}
                        </p>

                        <p className="mt-2 text-xs font-semibold uppercase text-blue-400">
                          {claim.status.replaceAll("_", " ")}
                        </p>

                        <p className="mt-2 text-sm leading-6 text-slate-400">
                          {claim.explanation}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </section>
            )}
          </div>
        )}

        {/* Evaluation CTA */}
        <section className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-6 sm:p-8">
          <div className="flex flex-wrap items-center justify-between gap-5">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-blue-400">
                Quantitative Evaluation
              </p>

              <h2 className="mt-2 text-xl font-semibold text-white">
                How well does DocuMind retrieve evidence?
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                Explore measured Recall@5, Precision@5, MRR,
                nDCG@5, latency comparisons, and per-question
                failure analysis across retrieval strategies.
              </p>
            </div>

            <Link
              href="/evaluation"
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-blue-500/40 bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-500"
            >
              Explore Evaluations
              <span aria-hidden="true">→</span>
            </Link>
          </div>
        </section>

        {/* Footer */}
        <footer className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-800 py-6 text-xs text-slate-500">
          <p>
            DocuMind · Explainable Retrieval-Augmented Generation
          </p>

          <div className="flex items-center gap-5">
            <Link
              href="/evaluation"
              className="hover:text-blue-400"
            >
              Evaluations
            </Link>

            <a
              href="https://github.com/hamedhoney/docuMind"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-blue-400"
            >
              GitHub ↗
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}
