# Provident V1 中文公开 Benchmark

核心组合为 **CRUD-RAG + RGB 中文 + C-MTEB/DuRetrieval、T2Reranking**。这里不新增问题、标准答案或 Golden Case，不做人工标注，不改 Retriever、Reranker、Verifier 和 Agent 工作流。

这是现有 `app.eval` 下的公开数据适配层，沿用 Provident 的模型客户端、生成入口、Cross-Encoder、使用量账本与阶段计时。原始数据、转换快照和执行结果均保存在 `data/benchmarks/`，不进入业务表。

运行后生成的审计、准备检查、模型结果和 Trace 保存在本地 `reports/` 与 `../data/benchmarks/`，不随源码发布。它们可能包含文档全文、机器路径、远端运行信息和数据库备份；公开结果前需单独脱敏。准备检查不能当作质量基线。仓库保留适配器源码、数据版本注册表与执行配置，不附带具体问答样本或单测正文。

## 数据与版本

`registry.json` 固定官方 Git/Hugging Face revision、URL、下载范围、标签来源、许可证及限制。DuRetrieval 的 corpus/queries 与 qrels 分别锁定版本，官方评测 split 为 `dev`；`queries` 是源文件 split。RGB 没有发布 train/test 划分，保持 `official_split=null`。CRUD 的 `crud_split` 是论文实验子集，不能称为官方 test split。

每个 normalized case 保留原始行、原始 ID（存在时）、来源行定位、题面、上下文、参考答案、相关性标签、对话字段、任务和模式。缺失字段保留 `null`；不伪造答案、文档 ID 或支持标签。CRUD 新闻 ID 可能跨任务重复，使用官方任务名及行号定位样本，同时保留源 ID。T2Reranking 没有样本 ID，使用固定源文件中的行号定位。

普通下载逐文件验证 SHA256；HF qrels/T2 还对照官方 LFS 大小和 SHA256，并检查 Parquet 首尾标记。失败有明确错误，原始字节不修补。`--retry-invalid` 将已确认不完整的下载保存在 `.failed-<sha256>` 后再尝试下载，保留失败记录。不会覆盖无法与既有清单对上的原始文件。

T2Reranking `dev` 文件为 120,293,598 bytes。慢网络可显式获取前 20 行所需的 Parquet 字节范围，保存每个原始区间及其校验值。此方式保留官方 revision、完整文件预期哈希、ETag、行数和来源行范围，并明确记录 `full_file_sha256_verified=false`。它只验证读取区间与 Parquet 解析，不能声称验证了整个 LFS 文件或获得完整官方成绩。区间读取失败时不会自动下载全文件。

## 环境

离线结构检查与单元测试使用 Python 3.11+ 标准库。Parquet、中文分词和真实生成的可选依赖分开安装，不会安装 torch/CUDA：

```powershell
python -m pip install -r benchmarks/requirements.txt
python -m pip install -r benchmarks/requirements-generation.txt
python scripts/test_benchmark.py
python scripts/check_benchmark_runtime.py
```

Cross-Encoder 必须使用已有的后端运行时及本地模型快照。本工具不会自动下载模型。运行环境检查只构造使用虚拟密钥的客户端，不发起生成请求；会把业务数据库地址强制覆盖为内存 SQLite。真实 worker 同样隔离数据库，并通过 `capture_calls` 将 LLM 使用量留在评测输出。

## 可复现下载与转换

```powershell
python scripts/run_benchmark.py fetch rgb
python scripts/run_benchmark.py fetch crud --max-file-bytes 30000000
python scripts/run_benchmark.py fetch cmteb_du
python scripts/run_benchmark.py fetch cmteb_du_qrels
python scripts/run_benchmark.py fetch cmteb_protocol

# 二选一；不会自动下载 DuRetrieval 全语料或 CRUD 新闻库。
python scripts/run_benchmark.py fetch cmteb_t2 --max-file-bytes 125000000
python scripts/run_benchmark.py fetch-sample cmteb_t2 --limit 20 --max-bytes 32000000

python scripts/prepare_benchmark.py
```

`prepare_benchmark.py` 不联网、不调用模型。它转换 RGB 五种上下文设置、CRUD 六组任务和两个 C-MTEB 子集各前 20 条，记录每项的成功或阻碍，并生成 `controlled_plan.json`。重复转换生成同一内容哈希；每次运行生成独立审计记录。

也可以独立转换：

```powershell
python scripts/run_benchmark.py convert rgb --file data/zh_refine.json --noise-rate 0.6 --passage-num 5 --seed 2333 --limit 20
python scripts/run_benchmark.py convert crud --file data/crud_split/split_merged.json --task questanswer_1doc --limit 20
python scripts/run_benchmark.py convert cmteb_du --file data/queries-00000-of-00001-7c7edb40be6b560c.parquet --limit 20
python scripts/run_benchmark.py validate <normalized-snapshot.json>
```

## 执行与授权

`validate` 只校验数据和快照哈希。`dry-run` 默认准备输入、Trace、预算和报告，状态为 `prepared_not_executed`，输出为 `null`。只有 `dry-run --execute --authorize-models` 才进行至多 20 条真实试跑。`evaluate` 必须显式授权，并指定模式和样本范围。没有隐式全量执行。

```powershell
python scripts/run_benchmark.py dry-run <snapshot.json> --mode generation_with_context --limit 10 --config benchmarks/controlled.example.json

# 仅在获得明确授权、检查配置和运行环境后执行：
python scripts/run_benchmark.py evaluate <snapshot.json> --mode generation_with_context --limit 10 --config benchmarks/controlled.example.json --authorize-models
python scripts/run_benchmark_suite.py benchmarks/controlled_plan.json --authorize-models

python scripts/run_benchmark.py compare <left-run-directory> <right-run-directory>
```

示例配置限制总请求、总 LLM Token、输出 Token、按配置单价计算的费用、每条超时与 Cross-Encoder 对数。模型与 SDK 均禁止重试。失败或超时后，服务商可能已接收请求，因此保留最坏预算预留，并停止后续服务商请求。suite 在各运行间共享预算，不能分别享用同一份上限。

为覆盖 Qwen 的思考 Token 预算，执行器仅在评测进程中把 SDK 的 `max_tokens` 替换为包含思考与回复的 `max_completion_tokens`，总上限由 `max_total_output_tokens` 指定；保留 Provident 原 Prompt、模型及默认思考模式。这个请求级预算变化随配置和 Trace 留痕，不与此前的预算协议混为同一实验。未重写生产模型适配层。

示例固定了 2026-10-29 核查的百炼北京地域 Qwen3.8-Max 非缓存单价：输入 12 元/百万 Token、输出 36 元/百万 Token，来源见配置中的官方价格 URL。折扣、缓存与实际账单未知，执行前仍须确认该价格是否适用。Token 预留基于输入 UTF-8 字节及模板余量，响应后再核对 usage；它不是声称精确计算了供应商 tokenizer。报告区分实际返回 Token、预算预留、按配置单价估算成本、失败请求未知用量及实际账单未知。本地 CE 的电力和硬件成本不按零成本推定，只是不产生本工具统计的 LLM API 费用。

不支持的模式记为 `not_supported`。部分失败、预算停止或未支持的真实执行返回非零退出码并保留报告。程序结束不表示评测通过。

## 任务与指标边界

- RGB：固定官方上下文，选段和种子对齐官方 `processdata`，答案匹配对齐外层 AND、内层别名 OR 的 `checkanswer`。使用 Provident 原生生成 Prompt，因此不是完整官方实验协议。参考答案子串匹配、官方关键词拒答代理和 Provident 拒答模板检测分开记录；不把子串命中称为语义正确性。固定上下文不测端到端检索。
- CRUD：保留 `questanswer_1doc/2docs/3docs`、`event_summary`、`continuing_writing`、`hallu_modified`。使用官方发布的源新闻作为固定上下文与官方任务 Prompt，复用 Provident 模型适配器。这是有证据的模型任务测试，**不是 Agent `digest/synthesis` 或完整 RAG 的成绩**。续写通过同一官方新闻 ID 关联原文；不把参考续写本身单独填成证据。原始参考含服务商失败字符串时不给有效分数。
- CRUD 文本指标：Jieba 分词、官方默认不含 brevity penalty 的 BLEU、ROUGE-L F1。作为公式对齐的文本重合指标，不能替代语义正确性。没有导入其 Milvus，没有启用会产生新问答和额外模型请求的 RAGQuestEval。
- DuRetrieval：问题及官方 qrels 已可转换。BM25、pgvector、Hybrid 和 Reranker 的完整语料检索需先准备隔离数据库、保留官方 passage ID 的索引与 ID 映射；当前 worker 明确不支持，没有用候选子集伪装全库检索。
- T2Reranking：完整保留每条选中问题的正负候选，排序器只读取问题与候选文本，不读取标签。复用 `rerank_scores`。计算 Recall@5/10、MRR@10、nDCG@10、MAP，MAP 对同分使用阈值 AP。只测官方候选内排序，不能称为全库 Recall。局部查询结果不能称为完整 C-MTEB 总分。
- 缺少输出级支持标签时，Unsupported Precision/Recall/F1、引用语义 Precision/Coverage 与多轮正确率均为 `not_evaluable`。T2 相关性标签不转成 Verifier 蕴含标签。没有 LLM Judge 时不输出伪语义正确率。

## 结果与 Badcase

每次运行保存 `manifest.json`、`snapshot.json`、`results.json`、`summary.json`、`report.md`、`badcases.json` 和 `traces/<id>.json`。保留数据及代码哈希、dirty 状态、输入、实际输出、上下文、Prompt、有效模型配置、阶段计时、usage、成本口径、错误原因和分母。

Trace ID 属于独立的 `benchmark_local` 命名空间。`application_trace_id=null` 表示没有写入业务 `query_traces`；不会伪造可在业务界面查到的 Trace。执行失败、引用答案匹配失败、错误拒答、检索未命中等仅标记 `candidate_badcase`，不晋升为人工确认 Golden Case。没有真正运行模型，就没有真实模型失败案例。

比较要求完全相同的数据快照、样本顺序、模式、语料和索引快照及相同的实际执行样本集合。准备记录和空结果不能产生对比收益；每项指标同时显示实际可评分样本数和选中总数。

## 来源与许可证

- [CRUD-RAG 官方仓库](https://github.com/IAAR-Shanghai/CRUD_RAG)：中文 README 标有 Apache-2.0，但固定版本未提供独立 LICENSE；新闻内容权利需另核实，不能推定全部可商业再分发。参考答案部分来自模型。
- [RGB 官方仓库及许可证](https://github.com/chen700564/RGB/blob/65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615/LICENSE.txt)：CC-BY-NC-SA-4.0；商业用途需要正式许可。原始样本需由使用者自行获取，并遵守对应许可；源码中不附带问答正文。
- [DuRetrieval](https://huggingface.co/datasets/C-MTEB/DuRetrieval)、[qrels](https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels)、[T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking)：dataset card 未明确许可证；遵循原数据源使用条款，不自动视为所有标签已由人工核实。
- [C-MTEB 官方协议](https://github.com/FlagOpen/FlagEmbedding/tree/897baa67c0ffddbfccf4e6776ff55310d67809d4/research/C_MTEB)：DuRetrieval 主指标为 nDCG@10，T2Reranking 主指标为 MAP。数据版本、范围和实现差异均须随分数披露。
