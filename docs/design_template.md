# Design Template

## Problem

Hệ thống cần xử lý các câu hỏi nghiên cứu phức tạp (ví dụ: "GraphRAG là gì và hoạt động như thế nào?") và
trả lời bằng văn bản có cấu trúc, trích dẫn nguồn. Task yêu cầu: tìm kiếm thông tin, phân tích nội dung,
và tổng hợp câu trả lời — ba bước đòi hỏi kỹ năng khác nhau.

## Why multi-agent?

Single-agent không đủ vì:
- Một agent làm cả search + phân tích + viết sẽ dễ bị "context pollution" — các bước khác nhau
  cần góc nhìn khác nhau (factual vs. critical vs. editorial).
- Không có traceability: không biết bước nào gây lỗi nếu output kém.
- Khó thêm guardrail riêng cho từng bước (ví dụ: giới hạn token cho researcher khác writer).
- Multi-agent cho phép parallel execution và retry độc lập cho từng bước trong tương lai.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Routing: quyết định agent nào chạy tiếp theo | `ResearchState` (toàn bộ) | Cập nhật `route_history` | Max iterations exceeded → fallback "done" |
| Researcher | Tìm kiếm Wikipedia, tóm tắt nguồn | `request.query`, `request.max_sources` | `sources`, `research_notes` | Wikipedia 0 kết quả → fallback model knowledge |
| Analyst | Phân tích research notes, extract key claims | `research_notes` | `analysis_notes` | LLM timeout → tenacity retry 3 lần |
| Writer | Viết câu trả lời ~500 từ có citation | `research_notes`, `analysis_notes`, `sources` | `final_answer` | Hallucinate citations → post-process validation |

## Shared state

| Field | Kiểu | Lý do cần |
|---|---|---|
| `request` | `ResearchQuery` | Query gốc + config (max_sources, audience) |
| `iteration` | `int` | Đếm số vòng để enforce max_iterations |
| `route_history` | `list[str]` | Trace routing decisions, debug |
| `sources` | `list[SourceDocument]` | Dữ liệu từ search, dùng bởi researcher & writer |
| `research_notes` | `str \| None` | Output của researcher → input của analyst |
| `analysis_notes` | `str \| None` | Output của analyst → input của writer |
| `final_answer` | `str \| None` | Output cuối cùng trả về user |
| `agent_results` | `list[AgentResult]` | Token usage, cost per agent — dùng cho benchmark |
| `trace` | `list[dict]` | Lightweight trace events |
| `errors` | `list[str]` | Lỗi nếu có, không crash toàn bộ pipeline |

## Routing policy

```
START
  │
  ▼
[Supervisor] ──────────────────────────────┐
  │                                        │
  ├─ research_notes is None  ──► [Researcher] ──► back to Supervisor
  │
  ├─ analysis_notes is None  ──► [Analyst]    ──► back to Supervisor
  │
  ├─ final_answer is None    ──► [Writer]     ──► back to Supervisor
  │
  └─ all filled OR iteration >= max_iterations ──► DONE
```

## Guardrails

- **Max iterations**: `MAX_ITERATIONS=6` (env), Supervisor enforce → route "done" nếu vượt quá.
- **Timeout**: `TIMEOUT_SECONDS=60` (env), cấu hình trong Settings — các LLM call có thể timeout.
- **Retry**: `tenacity` retry 3 lần với exponential backoff (2s → 4s → 8s) trong `LLMClient`.
- **Fallback**: Nếu Wikipedia 0 kết quả → `ResearcherAgent` ghi fallback note, pipeline tiếp tục.
- **Validation**: `Pydantic` validate toàn bộ `ResearchState` và `ResearchQuery` (min_length, ge/le).

## Benchmark plan

| Query | Metric | Expected outcome |
|---|---|---|
| "What is GraphRAG and how does it work?" | Latency | Multi-agent ~15s vs single-agent ~12s (overhead có thể chấp nhận) |
| "What is GraphRAG and how does it work?" | Cost (USD) | Multi-agent ~$0.0006 vs single-agent ~$0.0000 (1 call) |
| "What is GraphRAG and how does it work?" | Quality (0-10, LLM-as-judge) | Cả hai ≥ 8.0 — multi-agent có cấu trúc rõ hơn |
| "What is GraphRAG and how does it work?" | Citation coverage | 0% (Wikipedia không có bài GraphRAG) → fallback note ghi nhận |
| "What is GraphRAG and how does it work?" | Failure rate | 0% — pipeline hoàn thành nhờ fallback khi search miss |

## Exit ticket

**1. Case nào nên dùng multi-agent? Vì sao?**
Nên dùng khi task có nhiều bước rõ ràng cần skill khác nhau (tìm kiếm, phân tích, viết),
khi cần traceability để debug, hoặc khi muốn retry/fallback độc lập từng bước mà không restart
toàn bộ pipeline.

**2. Case nào không nên dùng multi-agent? Vì sao?**
Không nên dùng cho câu hỏi đơn giản trả lời được trong 1 LLM call — multi-agent thêm latency
(overhead routing + nhiều API call) và cost mà không tăng chất lượng đáng kể.
