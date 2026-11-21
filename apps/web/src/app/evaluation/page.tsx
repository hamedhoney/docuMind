"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import resultsJson from "@/data/retrieval-results.json";
import Link from "next/link";

// --------------------------------------------------
// Types
// --------------------------------------------------

type MetricKey =
  | "precision_at_k"
  | "recall_at_k"
  | "ndcg_at_k"
  | "mrr";

interface ModeMetrics {
  mode: string;
  precision_at_k: number | null;
  recall_at_k: number | null;
  ndcg_at_k: number | null;
  mrr: number | null;
  mean_latency_ms: number | null;
  p95_latency_ms: number | null;
}

interface QuestionResult {
  question_id: string;
  mode: string;
  precision_at_5: number;
  recall_at_5: number;
  ndcg_at_5: number;
  mrr: number;
  latency_ms: number;
}

interface BenchmarkData {
  schema_version: string;
  benchmark: string;
  environment: string;
  question_count: number;
  k: number;
  embedding_model: string;
  collection: string;
  modes: ModeMetrics[];
  results: QuestionResult[];
}

const data = resultsJson as BenchmarkData;

const COLORS: Record<string, string> = {
  dense: "#60a5fa",
  bm25: "#f59e0b",
  hybrid: "#34d399",
};

const LABELS: Record<string, string> = {
  dense: "Dense",
  bm25: "BM25",
  hybrid: "Hybrid",
};

const METRICS: {
  key: MetricKey;
  label: string;
  description: string;
}[] = [
  {
    key: "precision_at_k",
    label: "Precision@5",
    description: "Fraction of retrieved passages that are relevant.",
  },
  {
    key: "recall_at_k",
    label: "Recall@5",
    description: "Fraction of labeled relevant passages retrieved.",
  },
  {
    key: "mrr",
    label: "MRR",
    description: "Reciprocal rank of the first relevant passage.",
  },
  {
    key: "ndcg_at_k",
    label: "nDCG@5",
    description: "Ranking quality accounting for relevance and position.",
  },
];

// --------------------------------------------------
// Helpers
// --------------------------------------------------

function formatMetric(value: number | null): string {
  return value === null ? "N/A" : value.toFixed(3);
}

function formatLatency(value: number | null): string {
  return value === null ? "N/A" : `${value.toFixed(1)} ms`;
}

function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;

  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * p;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);

  if (lower === upper) return sorted[lower];

  return (
    sorted[lower] +
    (sorted[upper] - sorted[lower]) * (position - lower)
  );
}

function Card({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-3xl font-semibold tracking-tight text-white">
        {value}
      </p>
      {description && (
        <p className="mt-2 text-xs text-slate-500">
          {description}
        </p>
      )}
    </div>
  );
}

// --------------------------------------------------
// Main page
// --------------------------------------------------

export default function EvaluationPage() {
  const [selectedMetric, setSelectedMetric] =
    useState<MetricKey>("recall_at_k");

  const [selectedMode, setSelectedMode] =
    useState<string>("all");

  const [failuresOnly, setFailuresOnly] = useState(false);

  const modes = useMemo(() => {
    return data.modes.map((mode) => {
      const latencies = data.results
        .filter((result) => result.mode === mode.mode)
        .map((result) => result.latency_ms);

      return {
        ...mode,
        p95_latency_ms:
          mode.p95_latency_ms ?? percentile(latencies, 0.95),
      };
    });
  }, []);

  const hybrid = modes.find((mode) => mode.mode === "hybrid");

  const bestMode = [...modes]
    .filter((mode) => mode.recall_at_k !== null)
    .sort(
      (a, b) =>
        (b.recall_at_k ?? 0) - (a.recall_at_k ?? 0)
    )[0];

  const selectedMetricInfo = METRICS.find(
    (metric) => metric.key === selectedMetric
  );

  const chartData = modes.map((mode) => ({
    mode: LABELS[mode.mode] ?? mode.mode,
    value: mode[selectedMetric],
    fill: COLORS[mode.mode] ?? "#94a3b8",
  }));

  const latencyData = modes.map((mode) => ({
    mode: LABELS[mode.mode] ?? mode.mode,
    mean: mode.mean_latency_ms,
    p95: mode.p95_latency_ms,
  }));

  const questionIds = Array.from(
    new Set(data.results.map((result) => result.question_id))
  ).sort();

  const questionRows = questionIds.map((questionId) => {
    const questionResults = data.results.filter(
      (result) => result.question_id === questionId
    );

    return {
      questionId,
      dense: questionResults.find(
        (result) => result.mode === "dense"
      ),
      bm25: questionResults.find(
        (result) => result.mode === "bm25"
      ),
      hybrid: questionResults.find(
        (result) => result.mode === "hybrid"
      ),
      allFailed: questionResults.every(
        (result) => result.recall_at_5 === 0
      ),
    };
  });

  const failedQuestions = questionRows.filter(
    (row) => row.allFailed
  );

  const filteredResults = data.results.filter((result) => {
    if (selectedMode !== "all" && result.mode !== selectedMode) {
      return false;
    }

    if (failuresOnly && result.recall_at_5 > 0) {
      return false;
    }

    return true;
  });

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-slate-100 sm:px-8">
      <div className="mx-auto max-w-7xl space-y-10">

        {/* Header */}
        <header className="space-y-4">
            <Link
            href="/"
            className="inline-flex text-sm text-blue-400 hover:text-blue-300"
            >
            ← Back to DocuMind
            </Link>

          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-blue-400">
              DocuMind / Evaluation
            </p>

            <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Retrieval Evaluation
            </h1>

            <p className="mt-3 max-w-3xl text-slate-400">
              A reproducible benchmark comparing semantic retrieval,
              BM25 keyword search, and hybrid reciprocal rank fusion.
              All displayed metrics come from recorded evaluation runs.
            </p>
          </div>
        </header>

        {/* Benchmark metadata */}
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card
            label="Benchmark questions"
            value={String(data.question_count)}
            description={data.benchmark}
          />
          <Card
            label="Retrieval depth"
            value={`Top ${data.k}`}
            description="Passages per query"
          />
          <Card
            label="Best Recall@5"
            value={
              bestMode?.recall_at_k == null
                ? "N/A"
                : `${(bestMode.recall_at_k * 100).toFixed(1)}%`
            }
            description={
              bestMode
                ? LABELS[bestMode.mode] ?? bestMode.mode
                : "No results"
            }
          />
          <Card
            label="Hybrid mean latency"
            value={formatLatency(hybrid?.mean_latency_ms ?? null)}
            description="Retrieval only"
          />
        </section>

        {/* Quality comparison */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 sm:p-7">
          <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold">
                Retrieval quality
              </h2>
              <p className="mt-1 text-sm text-slate-400">
                Compare retrieval strategies using measured metrics.
              </p>
            </div>

            <select
              aria-label="Select evaluation metric"
              value={selectedMetric}
              onChange={(event) =>
                setSelectedMetric(event.target.value as MetricKey)
              }
              className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-white"
            >
              {METRICS.map((metric) => (
                <option key={metric.key} value={metric.key}>
                  {metric.label}
                </option>
              ))}
            </select>
          </div>

          <p className="mb-5 text-sm text-slate-400">
            {selectedMetricInfo?.description}
          </p>

          <div className="h-80 min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#334155"
                  vertical={false}
                />
                <XAxis
                  dataKey="mode"
                  stroke="#94a3b8"
                />
                <YAxis
                  domain={[0, 1]}
                  stroke="#94a3b8"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                    color: "#f8fafc",
                  }}
                  formatter={(value) =>
                    typeof value === "number"
                      ? value.toFixed(3)
                      : String(value ?? "N/A")
                  }
                />
                <Bar
                  dataKey="value"
                  name={selectedMetricInfo?.label ?? "Metric"}
                  radius={[6, 6, 0, 0]}
                  maxBarSize={90}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Strategy comparison table */}
        <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40">
          <div className="p-6">
            <h2 className="text-xl font-semibold">
              Strategy comparison
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Mean performance across {data.question_count} questions.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[780px] text-left text-sm">
              <thead className="bg-slate-900 text-slate-400">
                <tr>
                  <th className="px-5 py-3">Strategy</th>
                  <th className="px-5 py-3">Precision@5</th>
                  <th className="px-5 py-3">Recall@5</th>
                  <th className="px-5 py-3">MRR</th>
                  <th className="px-5 py-3">nDCG@5</th>
                  <th className="px-5 py-3">Mean latency</th>
                  <th className="px-5 py-3">P95 latency</th>
                </tr>
              </thead>

              <tbody>
                {modes.map((mode) => (
                  <tr
                    key={mode.mode}
                    className="border-t border-slate-800"
                  >
                    <td className="px-5 py-4 font-semibold">
                      {LABELS[mode.mode] ?? mode.mode}
                    </td>
                    <td className="px-5 py-4">
                      {formatMetric(mode.precision_at_k)}
                    </td>
                    <td className="px-5 py-4">
                      {formatMetric(mode.recall_at_k)}
                    </td>
                    <td className="px-5 py-4">
                      {formatMetric(mode.mrr)}
                    </td>
                    <td className="px-5 py-4">
                      {formatMetric(mode.ndcg_at_k)}
                    </td>
                    <td className="px-5 py-4">
                      {formatLatency(mode.mean_latency_ms)}
                    </td>
                    <td className="px-5 py-4">
                      {formatLatency(mode.p95_latency_ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Latency comparison */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 sm:p-7">
          <h2 className="text-xl font-semibold">
            Retrieval latency
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Mean and P95 retrieval latency in milliseconds.
          </p>

          <div className="mt-6 h-80 min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#334155"
                  vertical={false}
                />
                <XAxis
                  dataKey="mode"
                  stroke="#94a3b8"
                />
                <YAxis
                  stroke="#94a3b8"
                  unit=" ms"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                  }}
                />
                <Legend />
                <Bar
                  dataKey="mean"
                  name="Mean"
                  fill="#60a5fa"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="p95"
                  name="P95"
                  fill="#34d399"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Failure analysis */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 sm:p-7">
          <div className="mb-6">
            <h2 className="text-xl font-semibold">
              Failure analysis
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Questions where no labeled relevant passage appeared
              in the top five results of any evaluated strategy.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <Card
              label="Questions missed by every strategy"
              value={String(failedQuestions.length)}
              description={`Out of ${data.question_count} benchmark questions`}
            />
            <Card
              label="Questions with at least one successful strategy"
              value={String(
                questionRows.length - failedQuestions.length
              )}
              description="At least one relevant passage retrieved"
            />
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            {failedQuestions.map((row) => (
              <span
                key={row.questionId}
                className="rounded-lg border border-amber-700/40 bg-amber-500/10 px-3 py-2 font-mono text-sm text-amber-300"
              >
                {row.questionId}
              </span>
            ))}
          </div>

          <p className="mt-5 text-sm leading-6 text-slate-400">
            A zero-recall result does not necessarily mean the
            question is unanswerable. It may reflect missing
            relevance labels, chunking issues, or retrieval
            ranking errors. These cases require manual review.
          </p>
        </section>

        {/* Per-question results */}
        <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40">
          <div className="flex flex-wrap items-center justify-between gap-4 p-6">
            <div>
              <h2 className="text-xl font-semibold">
                Per-question results
              </h2>
              <p className="mt-1 text-sm text-slate-400">
                Inspect individual benchmark measurements.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <select
                aria-label="Filter retrieval strategy"
                value={selectedMode}
                onChange={(event) =>
                  setSelectedMode(event.target.value)
                }
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
              >
                <option value="all">All strategies</option>
                {modes.map((mode) => (
                  <option key={mode.mode} value={mode.mode}>
                    {LABELS[mode.mode] ?? mode.mode}
                  </option>
                ))}
              </select>

              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={failuresOnly}
                  onChange={(event) =>
                    setFailuresOnly(event.target.checked)
                  }
                  className="accent-blue-500"
                />
                Misses only
              </label>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="bg-slate-900 text-slate-400">
                <tr>
                  <th className="px-5 py-3">Question</th>
                  <th className="px-5 py-3">Strategy</th>
                  <th className="px-5 py-3">Precision@5</th>
                  <th className="px-5 py-3">Recall@5</th>
                  <th className="px-5 py-3">MRR</th>
                  <th className="px-5 py-3">nDCG@5</th>
                  <th className="px-5 py-3">Latency</th>
                </tr>
              </thead>

              <tbody>
                {filteredResults.map((result) => (
                  <tr
                    key={`${result.question_id}-${result.mode}`}
                    className="border-t border-slate-800 hover:bg-slate-800/40"
                  >
                    <td className="px-5 py-4 font-mono">
                      {result.question_id}
                    </td>
                    <td className="px-5 py-4">
                      {LABELS[result.mode] ?? result.mode}
                    </td>
                    <td className="px-5 py-4">
                      {result.precision_at_5.toFixed(3)}
                    </td>
                    <td className="px-5 py-4">
                      <span
                        className={
                          result.recall_at_5 === 0
                            ? "font-medium text-amber-400"
                            : "text-emerald-400"
                        }
                      >
                        {result.recall_at_5.toFixed(3)}
                      </span>
                    </td>
                    <td className="px-5 py-4">
                      {result.mrr.toFixed(3)}
                    </td>
                    <td className="px-5 py-4">
                      {result.ndcg_at_5.toFixed(3)}
                    </td>
                    <td className="px-5 py-4">
                      {result.latency_ms.toFixed(1)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {filteredResults.length === 0 && (
              <p className="p-6 text-sm text-slate-400">
                No results match the selected filters.
              </p>
            )}
          </div>
        </section>

        {/* Reproducibility */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
          <h2 className="text-xl font-semibold">
            Benchmark reproducibility
          </h2>

          <div className="mt-5 grid gap-5 sm:grid-cols-2">
            <div>
              <p className="text-xs text-slate-500">
                Dataset
              </p>
              <p className="mt-1 font-mono text-sm">
                {data.benchmark}
              </p>
            </div>

            <div>
              <p className="text-xs text-slate-500">
                Environment
              </p>
              <p className="mt-1 font-mono text-sm">
                {data.environment}
              </p>
            </div>

            <div>
              <p className="text-xs text-slate-500">
                Embedding model
              </p>
              <p className="mt-1 break-all font-mono text-sm">
                {data.embedding_model}
              </p>
            </div>

            <div>
              <p className="text-xs text-slate-500">
                Qdrant collection
              </p>
              <p className="mt-1 break-all font-mono text-sm">
                {data.collection}
              </p>
            </div>
          </div>

          <p className="mt-6 text-sm leading-6 text-slate-400">
            Results are based on a small, ten-question benchmark.
            Metrics depend on the quality and completeness of
            manually reviewed relevance labels. Latency values
            represent retrieval measurements, not end-to-end
            answer generation time.
          </p>

          <a
            href="https://github.com/hamedhoney/docuMind"
            target="_blank"
            rel="noopener noreferrer"
            className="mt-5 inline-block text-sm font-medium text-blue-400 hover:text-blue-300"
          >
            View DocuMind on GitHub →
          </a>
        </section>

        <footer className="border-t border-slate-800 pt-6 text-center text-xs text-slate-500">
          DocuMind · Explainable Retrieval-Augmented Generation
        </footer>
      </div>
    </main>
  );
}
