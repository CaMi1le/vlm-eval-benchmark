# vlm-eval-benchmark

[![CPU checks](https://github.com/CaMi1le/vlm-eval-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/CaMi1le/vlm-eval-benchmark/actions/workflows/ci.yml)

[在线交互式雷达图](https://cami1le.github.io/vlm-eval-benchmark/) · [实测数据与条件](examples/minicpm47/README.md)

可复现的视觉语言模型评测：**自动运行 → 统一打分 → 回答与原图核查 → 交互式雷达图**。
支持本地 Transformers / vLLM 和 OpenAI-compatible HTTP 接口。评分和图表离线运行，无需 LLM judge。

- 冻结题目、图片、视频帧与 SHA256；六个数据集各 240 题，共 **1,440 题**。
- 同一解析器提取最终答案；失败、空回答、未解析答案保留在分母，格式差异另列。
- 记录逐请求延迟、输出长度、运行参数和本地 GPU 使用；无图诊断单独统计。
- HTML 雷达图支持模型开关、轴说明、数值表、SVG 与 JSON 导出；无需 CDN 或在线服务。
- 内置 CPU 演示与离线回归测试；不包含模型权重、真实数据集媒体或访问凭据。

这是固定分层子集工具，**不等于各数据集完整官方榜单**。运行时、硬件、精度、视觉压缩与输出预算会影响成绩和速度。参见 [评测口径](docs/protocol.md)。

## 五分钟演示（不下载数据、不调用模型）

Python 3.10+，建议独立虚拟环境：

```bash
git clone https://github.com/CaMi1le/vlm-eval-benchmark.git
cd vlm-eval-benchmark
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
vlmbench demo --output demo-output
```

打开 `demo-output/report/review.html` 查看逐题回答，打开
`demo-output/report/dashboard/dashboard.html` 查看雷达图。
演示的 144 张图、问题、回答和耗时全部是合成数据，模型名称带 `demo-scripted`，不作为能力成绩。

## 准备固定 1,440 题

```bash
python -m pip install -e '.[datasets]'
vlmbench fetch --config configs/datasets.json --cache cache/datasets
vlmbench prepare --config configs/datasets.json --cache cache/datasets --output data/public-vlm-v1
vlmbench fetch --config configs/datasets-v2.json --cache cache/datasets
vlmbench prepare-expanded --config configs/datasets-v2.json --cache cache/datasets \
  --base-suite data/public-vlm-v1 --output data/public-vlm-v2
vlmbench verify --suite data/public-vlm-v2
```

首次下载包含视频压缩包，需预留空间和时间。源数据固定 revision，下载时核对远端文件清单；准备完成后保存逐文件 SHA256、采样与媒体信息。v2 保留 v1 的 144 题，再分层扩展到 1,440 题。不同视频解码版本可能影响帧字节；比较模型时共用同一份冻结目录。数据按原发布方条款使用，源链接见配置文件；仓库许可证不覆盖源数据和模型。

## 自动运行

**HTTP 接口**：将 `configs/api.example.json` 复制到 `configs/api.local.json`，填写 endpoint、key、model。该本地文件已被 Git 忽略。

```bash
vlmbench api-run --suite data/public-vlm-v2 --credentials configs/api.local.json \
  --output runs/api-model --concurrency 1 --timeout 300
```

包含一次预热、1,440 个有图请求和 6 个无图诊断。API 请求使用 temperature=0、max_tokens=128；服务端实际推理预算和 thinking 行为须另行记录。失败返回非零状态，部分结果与恢复清单保留。

**本地 GPU**：参照 [运行环境](docs/runtime.md) 准备独立模型环境，复制 `configs/models.example.json` 为 `configs/models.local.json`，填写路径和 GPU 索引。

```bash
vlmbench run --suite data/public-vlm-v2 --models configs/models.local.json \
  --model minicpm47-awq --workspace runs/local-awq --run-id full --seconds 900
```

在 Linux GPU 主机执行。控制器等待所选 GPU 空闲，保存输入与配置快照；每个正式请求超时后释放本轮进程并继续剩余请求。结果位于 `runs/local-awq/runs/full/minicpm47-awq/result`。先加 `--view smoke` 并使用另一新 workspace 可做 24 个有图请求与 6 个无图诊断；正式评分须使用 core 全集。

## 自动打分与交互式报告

把模型结果目录传给统一入口：

```bash
vlmbench report --suite data/public-vlm-v2 \
  --runs runs/api-model runs/local-awq/runs/full/minicpm47-awq/result \
  --output reports/comparison
```

产出 `scores/comparison.json`、逐题 CSV、评分规则、`review.html` 原图浏览器与 `dashboard/dashboard.html` 雷达图。HTML 内嵌数据，不调用第三方网站。报告可能包含源数据媒体和运行机器信息，分享前按自身数据授权范围检查。

只打分或只画图：

```bash
vlmbench score --suite data/public-vlm-v2 --runs runs/api-model --output reports/scores
vlmbench radar --scores reports/scores/comparison.json --output reports/radar
vlmbench radar --scores examples/minicpm47/comparison.json --output reports/minicpm47
```

雷达图的六条能力轴是数据集正确率，第七轴为 `100 × τ / (τ + 单题中位秒数)`，默认 τ=1 秒；越快越外，保留原始秒数，不以面积生成总排名。图表拒绝混合不同冻结输入、分母或评分策略。

## MiniCPM-V 4.7 实测示例

[示例说明与汇总](examples/minicpm47/README.md) 保留此前 1,440 题实测的聚合指标：BF16 **1,153 / 1,440（80.07%）**；官方 INT4 AWQ **1,142 / 1,440（79.31%）**。这两次使用不同运行框架和 GPU 数，不能据此得出纯量化损失或同硬件加速倍数。原始回答、媒体、主机名与本地路径未随示例发布。

## 开发与结构

```bash
python -m pip install -e '.[datasets,dev]'
python -m pytest -q
python -m build
```

`src/vlmbench/` 提供运行、解析、评分和图表；`configs/` 保存固定数据来源与可编辑模型示例；`tests/` 覆盖输入完整性、故障分母、解析、单位换算、API 脱敏、超时处理与图表一致性。CI 使用合成数据和 loopback HTTP fixture，不访问模型 API、不使用 GPU。

[数据格式](docs/data-format.md) · [运行环境](docs/runtime.md) · [贡献说明](CONTRIBUTING.md) · [Apache-2.0](LICENSE) · [第三方说明](NOTICE)
