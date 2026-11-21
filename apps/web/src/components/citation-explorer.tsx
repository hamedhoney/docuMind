
"use client";

import { useState } from "react";

import type { AskResponse } from "@/lib/api";

type Source = AskResponse["sources"][number];

interface CitationExplorerProps {
  answer: string;
  sources: Source[];
}

function parseAnswer(answer: string) {
  // Supports [1], [2, 4], and [1][3].
  return answer.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
}

function CitationButton({
  number,
  active,
  onClick,
}: {
  number: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`View source ${number}`}
      className={`mx-0.5 inline-flex rounded px-1.5 py-0.5
        align-baseline text-xs font-semibold transition-colors ${
          active
            ? "bg-blue-500 text-white"
            : "bg-blue-500/15 text-blue-300 hover:bg-blue-500/30"
        }`}
    >
      [{number}]
    </button>
  );
}

export default function CitationExplorer({
  answer,
  sources,
}: CitationExplorerProps) {
  const [selected, setSelected] = useState<number | null>(
    null
  );

  const source =
    selected !== null
      ? sources[selected - 1]
      : null;

  const parts = parseAnswer(answer);

  return (
    <section className="space-y-5 rounded-xl border border-slate-800 p-6">
      <div>
        <h2 className="text-xl font-semibold">
          Answer
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Click a citation to inspect its evidence.
        </p>
      </div>

      <div className="whitespace-pre-wrap text-base leading-8 text-slate-200">
        {parts.map((part, index) => {
          const isCitation =
            /^\[\d+(?:\s*,\s*\d+)*\]$/.test(part);

          if (!isCitation) {
            return (
              <span key={index}>
                {part}
              </span>
            );
          }

          const numbers = part
            .slice(1, -1)
            .split(",")
            .map((value) => Number(value.trim()));

          return (
            <span key={index}>
              {" "}
              {numbers.map((number) => {
                const valid =
                  Number.isInteger(number) &&
                  number >= 1 &&
                  number <= sources.length;

                if (!valid) {
                  return (
                    <span
                      key={number}
                      className="text-amber-400"
                      title="Citation has no matching source"
                    >
                      [{number}]
                    </span>
                  );
                }

                return (
                  <CitationButton
                    key={number}
                    number={number}
                    active={selected === number}
                    onClick={() =>
                      setSelected(
                        selected === number
                          ? null
                          : number
                      )
                    }
                  />
                );
              })}
            </span>
          );
        })}
      </div>

      {source && selected !== null && (
        <div
          role="region"
          aria-label={`Evidence for citation ${selected}`}
          className="space-y-3 rounded-xl border border-blue-500/30 bg-slate-900 p-5"
        >
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="font-semibold">
                Source [{selected}]
              </h3>
              <p className="mt-1 text-sm text-slate-400">
                Page {source.page_number}
              </p>
            </div>

            <button
              type="button"
              onClick={() => setSelected(null)}
              aria-label="Close evidence"
              className="rounded-lg px-3 py-1 text-slate-400 hover:bg-slate-800"
            >
              Close
            </button>
          </div>

          <p className="whitespace-pre-wrap text-sm leading-7 text-slate-300">
            {source.text}
          </p>

          <div className="border-t border-slate-800 pt-3">
            <p className="break-all font-mono text-xs text-slate-500">
              Chunk: {source.chunk_id}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
