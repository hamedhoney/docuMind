
"use client";

import { useState } from "react";

import type {
  RankedChunk,
  RetrievalTrace,
} from "@/lib/api";

type Stage = "dense" | "bm25" | "fusion" | "selected";

const stages: { id: Stage; label: string }[] = [
  { id: "dense", label: "Dense" },
  { id: "bm25", label: "BM25" },
  { id: "fusion", label: "RRF Fusion" },
  { id: "selected", label: "Selected" },
];

function formatMs(value: number): string {
  return `${value.toFixed(2)} ms`;
}

function PassageCard({
  passage,
}: {
  passage: RankedChunk;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="rounded-xl border border-slate-700 bg-slate-900 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="rounded-md bg-blue-500/15 px-2 py-1 text-xs font-semibold text-blue-300">
            #{passage.rank}
          </span>
          <span className="text-sm text-slate-300">
            Page {passage.page_number}
          </span>
        </div>

        <span className="font-mono text-xs text-slate-400">
          Score: {passage.score.toFixed(4)}
        </span>
      </div>

      <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-300">
        {expanded
          ? passage.text
          : passage.text.slice(0, 320) +
            (passage.text.length > 320 ? "..." : "")}
      </p>

      {passage.text.length > 320 && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="mt-3 text-sm font-medium text-blue-400 hover:text-blue-300"
        >
          {expanded ? "Show less" : "Show full passage"}
        </button>
      )}

      <p className="mt-3 break-all font-mono text-xs text-slate-500">
        {passage.chunk_id}
      </p>
    </article>
  );
}

export default function RetrievalXRay({
  trace,
}: {
  trace: RetrievalTrace;
}) {
  const [stage, setStage] = useState<Stage>("selected");

  const timings = [
    {
      label: "Dense",
      value: trace.timings.dense_ms,
    },
    {
      label: "BM25",
      value: trace.timings.bm25_ms,
    },
    {
      label: "Fusion",
      value: trace.timings.fusion_ms,
    },
    {
      label: "Rerank",
      value: trace.timings.rerank_ms,
    },
  ];

  const maxTime = Math.max(
    ...timings.map((item) => item.value),
    1
  );

  const passages = trace[stage] ?? [];

  return (
    <section className="space-y-6 rounded-xl border border-slate-800 p-6">
      <div>
        <h2 className="text-xl font-semibold">
          Retrieval X-Ray
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Inspect how DocuMind finds and ranks evidence.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {[
          ["Dense", trace.dense.length],
          ["BM25", trace.bm25.length],
          ["Fused", trace.fusion.length],
        ].map(([label, count]) => (
          <div
            key={label}
            className="rounded-lg bg-slate-900 p-4"
          >
            <p className="text-xs text-slate-400">
              {label}
            </p>
            <p className="mt-1 text-2xl font-semibold">
              {count}
            </p>
          </div>
        ))}
      </div>

      <div>
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-semibold">
            Retrieval latency
          </h3>
          <span className="text-sm text-slate-400">
            Total: {formatMs(trace.timings.total_ms)}
          </span>
        </div>

        <div className="space-y-3">
          {timings.map((item) => (
            <div key={item.label}>
              <div className="mb-1 flex justify-between text-sm">
                <span>{item.label}</span>
                <span className="text-slate-400">
                  {formatMs(item.value)}
                </span>
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-blue-500"
                  style={{
                    width: `${
                      (item.value / maxTime) * 100
                    }%`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="mb-3 font-semibold">
          Ranking stages
        </h3>

        <div className="flex flex-wrap gap-2">
          {stages.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setStage(item.id)}
              className={`rounded-lg px-4 py-2 text-sm ${
                stage === item.id
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        {passages.length === 0 ? (
          <p className="text-sm text-slate-400">
            No passages available for this stage.
          </p>
        ) : (
          passages.map((passage) => (
            <PassageCard
              key={passage.chunk_id}
              passage={passage}
            />
          ))
        )}
      </div>
    </section>
  );
}
