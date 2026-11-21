"use client";

import { useState } from "react";

import {
  askQuestion,
  uploadDocument,
  type AskResponse,
  type RetrievalMode,
} from "@/lib/api";

const modes: RetrievalMode[] = [
  "dense",
  "bm25",
  "hybrid",
  "hybrid_rerank",
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [documentId, setDocumentId] = useState(
  "24b26eb7b9bba731af9746974c452a23f870814d31b7d55cd681b771dfe02fdc"
);
  const [question, setQuestion] = useState("");
  const [mode, setMode] =
    useState<RetrievalMode>("hybrid");
  const [verify, setVerify] = useState(false);
  const [result, setResult] =
    useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  async function handleUpload() {
    if (!file) return;

    setLoading(true);
    setMessage("");

    try {
      const response = await uploadDocument(file);
      setDocumentId(response.document_id);
      setResult(null);
      setMessage(
        `Indexed ${response.pages} pages and ` +
        `${response.chunks} chunks.`
      );
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Upload failed."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk() {
    if (!question.trim()) return;

    setLoading(true);
    setMessage("");

    try {
      const response = await askQuestion(
        question,
        mode,
        verify,
        documentId
      );
      setResult(response);
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Question failed."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto max-w-5xl space-y-8 p-8">
        <header className="border-b border-slate-800 pb-6">
          <h1 className="text-3xl font-bold">
            DocuMind
          </h1>
          <p className="mt-2 text-slate-400">
            Explainable, self-evaluating RAG platform
          </p>
        </header>

        <section className="rounded-xl border border-slate-800 p-6">
          <h2 className="mb-4 text-xl font-semibold">
            Document Upload
          </h2>

          <input
            type="file"
            accept=".pdf,application/pdf"
            onChange={(event) =>
              setFile(event.target.files?.[0] ?? null)
            }
          />

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="mt-4 block rounded-lg bg-blue-600 px-5 py-2 disabled:opacity-50"
          >
            Upload PDF
          </button>

          {documentId && (
            <p className="mt-3 break-all text-xs text-slate-400">
              Document ID: {documentId}
            </p>
          )}
        </section>

        <section className="rounded-xl border border-slate-800 p-6">
          <h2 className="mb-4 text-xl font-semibold">
            Ask DocuMind
          </h2>

          <textarea
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            placeholder="Ask a question about your documents..."
            className="w-full rounded-lg border border-slate-700 bg-slate-900 p-4"
            rows={3}
          />

          <div className="mt-4 flex flex-wrap items-center gap-4">
            <select
              value={mode}
              onChange={(event) =>
                setMode(event.target.value as RetrievalMode)
              }
              className="rounded-lg bg-slate-800 p-3"
            >
              {modes.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>

            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={verify}
                onChange={(event) =>
                  setVerify(event.target.checked)
                }
              />
              Verify evidence
            </label>

            <button
              onClick={handleAsk}
              disabled={loading || !question.trim()}
              className="rounded-lg bg-blue-600 px-5 py-3 disabled:opacity-50"
            >
              {loading ? "Processing..." : "Ask"}
            </button>
          </div>
        </section>

        {message && (
          <p role="status" className="text-sm text-amber-300">
            {message}
          </p>
        )}

        {result && (
          <div className="space-y-6">
            <section className="rounded-xl border border-slate-800 p-6">
              <h2 className="mb-3 text-xl font-semibold">
                Answer
              </h2>
              <p className="whitespace-pre-wrap text-slate-200">
                {result.answer}
              </p>
            </section>

            {result.retrieval_trace && (
              <section className="rounded-xl border border-slate-800 p-6">
                <h2 className="mb-4 text-xl font-semibold">
                  Retrieval X-Ray
                </h2>

                <p className="mb-4 text-slate-400">
                  Total retrieval time:{" "}
                  {result.retrieval_trace.timings.total_ms.toFixed(2)}
                  {" "}ms
                </p>

                <div className="space-y-3">
                  {result.retrieval_trace.selected.map(
                    (chunk) => (
                      <div
                        key={chunk.chunk_id}
                        className="rounded-lg bg-slate-900 p-4"
                      >
                        <p className="text-sm font-semibold">
                          Rank #{chunk.rank} · Page{" "}
                          {chunk.page_number}
                        </p>
                        <p className="mt-1 text-xs text-slate-400">
                          Score: {chunk.score.toFixed(4)}
                        </p>
                        <p className="mt-3 text-sm text-slate-300">
                          {chunk.text}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </section>
            )}

            {result.verification && (
              <section className="rounded-xl border border-slate-800 p-6">
                <h2 className="mb-4 text-xl font-semibold">
                  Evidence Verification
                </h2>

                <p className="mb-4 text-slate-400">
                  Support rate:{" "}
                  {result.verification.support_rate === null
                    ? "N/A"
                    : `${(
                        result.verification.support_rate * 100
                      ).toFixed(1)}%`}
                </p>

                <div className="space-y-3">
                  {result.verification.claims.map(
                    (claim, index) => (
                      <div
                        key={index}
                        className="rounded-lg bg-slate-900 p-4"
                      >
                        <p className="font-medium">
                          {claim.claim}
                        </p>
                        <p className="mt-2 text-sm text-blue-300">
                          {claim.status.replaceAll("_", " ")}
                        </p>
                        <p className="mt-2 text-sm text-slate-400">
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
      </div>
    </main>
  );
}
