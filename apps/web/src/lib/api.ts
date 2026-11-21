const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

export type RetrievalMode =
  | "dense"
  | "bm25"
  | "hybrid"
  | "hybrid_rerank";

export interface RankedChunk {
  chunk_id: string;
  document_id: string;
  page_number: number;
  text: string;
  rank: number;
  score: number;
}

export interface RetrievalTrace {
  query: string;
  mode: RetrievalMode;
  top_k: number;
  timings: {
    dense_ms: number;
    bm25_ms: number;
    fusion_ms: number;
    rerank_ms: number;
    total_ms: number;
  };
  dense: RankedChunk[];
  bm25: RankedChunk[];
  fusion: RankedChunk[];
  reranked: RankedChunk[];
  selected: RankedChunk[];
}

export interface VerifiedClaim {
  claim: string;
  status:
    | "supported"
    | "contradicted"
    | "insufficient_evidence";
  citation_ids: number[];
  evidence: string[];
  explanation: string;
}

export interface VerificationReport {
  claims: VerifiedClaim[];
  supported_count: number;
  contradicted_count: number;
  insufficient_count: number;
  support_rate: number | null;
}

export interface AskResponse {
  answer: string;
  sources: Array<{
    chunk_id: string;
    document_id: string;
    page_number: number;
    text: string;
    score?: number;
    rrf_score?: number;
    rerank_score?: number;
  }>;
  retrieval_trace: RetrievalTrace | null;
  verification: VerificationReport | null;
}

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json() as Promise<{
    document_id: string;
    pages: number;
    chunks: number;
  }>;
}

export async function askQuestion(
  question: string,
  mode: RetrievalMode,
  verify: boolean,
  documentId?: string
): Promise<AskResponse> {
  const response = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: 5,
      retrieval_mode: mode,
      document_id: documentId || null,
      verify,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}
