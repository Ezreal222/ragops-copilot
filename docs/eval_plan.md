# Eval 方案设计（W5 蓝图）

> RAGOps Copilot · 第 5 周交付物 = **eval 指标表（前后对比）**。
> 本文是 D2–D6 的施工蓝图：定指标 → 升级 eval set → 跑 baseline → ablation。
> 一句话方法论：**定指标 → 标 eval set → 量 baseline → 每次只改一个变量 → 重测对比 → 决策。**

---

## 0. 为什么这周最值钱

W4 我们把检索"跑通"了，也量了 `recall@1/3/5`。但有两个洞还没补：

1. **gold 标注太脆**。W4 的 gold 钉在概览 `::0` chunk 上，导致 chunk-level 和 doc-level 召回差一大截（见 README「Results」）——这个 gap 大半是**标注假象**，不是检索器真实能力。
2. **只评了检索，没评生成**。"答案忠不忠于上下文（抗幻觉）""答得切不切题"完全没量化。

这周就是把项目从"能跑"升级成"有可信度量"：补上**生成侧指标**，并把 gold 标注**从 chunk 级升级到文档级 + 多 gold**，让数字可信。

---

## 1. 指标选型（评什么、用什么工具）

分两层评，缺一层都会"盲"：组件级定位问题出在哪，端到端反映用户体验。

| 维度 | 指标 | 评什么 | 工具 | 何时引入 |
|---|---|---|---|---|
| **检索** | `recall@1/3/5/10` | 该召回的 gold 有没有进 top-k（覆盖率） | 自己的 `eval/eval_retrieval.py`（已有） | W4 已有 |
| **检索** | **Context Recall** | 回答所需的信息有没有被召回到上下文里（需 reference） | RAGAS | D3 |
| **检索** | **Context Precision** | 召回的上下文相不相关、相关的排没排在前面 | RAGAS | D3 |
| **生成** | **Faithfulness** | 答案是否**忠于检索到的上下文**（每个 claim 能否被 context 支持）→ 直接量化**抗幻觉** | RAGAS（+ D4 自写 LLM-as-judge 复现原理） | D3 / D4 |
| **生成** | **Answer Relevancy** | 答案是否**切题**（答到点上，没跑偏/没注水） | RAGAS | D3 |
| **运维** | 成本 / 延迟 | 每次 ablation 顺手记 token 成本与 query 延迟 | 手动 / 脚本计时 | 全周 |

**记忆法**：precision / recall 管**检索**，faithfulness / answer relevancy 管**生成**。

**faithfulness 直觉**：把答案拆成若干 claim，逐条判断能否被检索到的 context 支持，**支持比例 = faithfulness**。

**LLM-as-judge 的局限**（为什么 D4 还要自己写一遍）：用 LLM 当裁判**有成本、有偏差、不稳定**。缓解：固定 prompt + `temperature=0` + 多次取平均；并保留人工抽检。D4 自写一版是为了理解打分原理，不把 RAGAS 当黑箱。

---

## 2. eval set 升级（接 W4 伏笔）

W4 的 schema（仍保留，向后兼容）：

```jsonc
{"q": "...", "gold_chunk_ids": ["docs/.../file.md::5", ...]}
```

**升级后 schema**（D2 标注目标，目标 **30–50 题**）：

```jsonc
{
  "q": "What is PagedAttention and what problem does it solve?",
  "gold_doc_ids":   ["docs/design/paged_attention.md"],        // 文档级：答案在哪个/哪些文档里（多 gold）
  "gold_chunk_ids": ["docs/design/paged_attention.md::0"],     // 可选：保留 chunk 级做对照
  "reference": "PagedAttention is vLLM's attention algorithm that ... (一两句话参考答案)"  // RAGAS context recall / answer 类指标要用
}
```

三处改动及理由：

1. **新增 `gold_doc_ids`（文档级、多 gold）**：W4 把 gold 钉死在单个 `::0` chunk 上，使 chunk-level recall 被低估。改成"答案落在哪些**文档**里"，更贴近真实检索目标，也消除"分块边界/重叠"带来的标注噪声。`gold_chunk_ids` 保留做细粒度对照。
2. **新增 `reference`（参考答案，一两句话）**：RAGAS 的 context recall 和答案类指标需要一个 ground-truth 参考。**由 Yang 手写**——这是 eval set 的金标准，不交给模型生成。
3. **扩到 30–50 题**：覆盖更多文档、更多问题类型（概念 / how-to / 配置 / 报错），让数字不被小样本噪声主导。

> 分工：**harness（脚本、schema、loader）我来搭；gold 标注和 reference 答案 Yang 手写。**
> D2 会提供一个 `eval/find_chunks.py` 式的辅助脚本，帮 Yang 把候选文档/chunk 列出来，但最终 gold 由人定。

---

## 3. 本周 ablation 计划（每次只动一个变量）

ablation = 控制变量法：固定其它一切，只改一个旋钮，重测全套指标，看动了哪个数字。所有旋钮都集中在 `src/config.py`，便于复现。

| Ablation | 改的变量（`config.py`） | 固定不动 | 看哪些指标 | 排期 |
|---|---|---|---|---|
| **① chunking** | `ChunkConfig.chunk_size` / `chunk_overlap`（如 800/100 → 512/64、1200/150） | embedding、index、retrieval、prompt | recall@k + context recall/precision（注意：换分块要**重新 ingest 重建索引**） | **D5** |
| **② 检索 / rerank** | `RetrievalConfig.top_k`、`rerank_top_n`、`use_reranker` 开/关 | 分块、embedding、index | recall@k + context precision + faithfulness + 延迟/成本 | **D6** |

**呼应 W4 结论**：W4 发现 doc-level recall@5 已饱和（=1.00），reranker 在召回上没有提升空间、只能重排序——"召回已饱和时 reranker 不划算"。D6 在升级后的 eval set 上**复测**这个结论是否依然成立（更大/未饱和的标注下，rerank 可能重新变得有价值）。

---

## 4. D2–D6 排期

| Day | 产出 | 依赖 |
|---|---|---|
| **D2** | 标 eval set：升级 schema，扩到 30–50 题，补 `gold_doc_ids` + `reference`（Yang 手写，我搭辅助脚本 + loader） | 本文 schema |
| **D3** | 接 RAGAS，跑出**生成侧 baseline**：context precision/recall + faithfulness + answer relevancy | D2 的 eval set |
| **D4** | 自写 **LLM-as-judge**（复现 faithfulness 打分原理，理解 RAGAS 黑箱） | D3 |
| **D5** | **chunking ablation**（①）：size/overlap 对 eval 的影响 | D2–D3 |
| **D6** | **检索 / rerank ablation**（②）：top-N / k / 用不用 rerank，**复测 W4 「rerank 不划算」结论** | D2–D3 |

**完成判定（本周）**：产出一张 **eval 指标表（前后对比）**——baseline vs 各 ablation，覆盖检索 + 生成两层，并附成本/延迟。

---

## 5. 复现性约定

- 所有可调旋钮集中在 `src/config.py`（chunk / embed / index / retrieval / rerank / llm），ablation = 改一处。
- eval set 固定在 `eval/eval_set.jsonl`，版本随仓库走。
- 换分块 → 必须重建 OpenSearch 索引（ingest 幂等、可重跑）。
- LLM 打分固定 `temperature=0`，必要时多次取平均，记录用的是哪个 provider/model（当前 DeepSeek `deepseek-v4-pro`）。
