# Darjeeling 介绍文章迭代记录

## 写作约束

- 目标读者：第一次听说 Darjeeling、不了解仓库内部概念的人。
- 读者问题：这个项目到底做什么，为什么要这么做，怎么避免为了省钱而增加错误。
- 证据来源：项目 README、架构总览、目标定义、候选评测、发布运行时、运行反馈和实现状态文档。
- 成功标准：读者在网页第一页内能复述为一句话。

## 计数口径

- 字数：只统计正文中的中文、英文和数字字符，不统计 Markdown 标记、标点和空白。
- 术语种类数：统计正文中出现过的项目专有或半专有技术名词的不同种类。每轮使用同一份术语表。

## 第一版

Darjeeling 是一个面向任务优化的运行框架。它从用户审阅过的 Target Definition 开始，读取输入输出 schema、contract.py、reference.py、data.yaml 和运行要求，生成 contract hash，并用这个哈希贯穿 Snapshot、Candidate、Report、Release、Trace、Audit 和 Telemetry Evidence。Core 不解释 NLU、业务字段、数据集标签或应用语义，只把这些内容当作目标自带的不透明数据。

系统的基本目标是在强参考模型前部署便宜的本地 Artifact。L1、L2、L3 Artifact 使用统一的 accept/abstain Worker Protocol：当 Artifact 有足够信心时返回完整输出，否则 abstain；Core 再继续路由到后续层或 L4 fallback。Cache 只是运行时优化，不被计入本地 Coverage。系统只追求在 Precision 不下降、wrong accept 受控、Generalization 有证据的情况下提高 Coverage，并用 cost ledger 判断每千次请求的节省和回本请求量。

编译路径由 Compile Orchestration 启动。它把 RecompileRequest、当前 Release、TargetDefinition、TargetRuntimeContract、预算、调度策略和历史报告组合起来，调用 Snapshot And Reference 生成冻结的 Snapshot。Snapshot 会固定 train、validation、test 边界，处理 Reference Qualification，并确保 Agent 只能看到 train view。随后 Agent Workspace 创建隔离 attempt，让一个 target adaptation agent 编写 scaffolding 和 L1/L2/L3 runtime source。Agent 可以写搜索脚本、训练器、规则表或代码生成器，但不能看到隐藏验证/测试行，也不能拿到注册表凭证或发布权限。

Agent 提交 Candidate 后，Artifact Worker 校验包、manifest、资源限制、网络禁用和协议兼容性，并冻结 Artifact 字节。Candidate Evaluation 重新运行官方评测，生成 validation/test/final Report，比较当前 Release，检查 Precision、Coverage、Generalization、延迟、成本、安全、分片稳定性和 holdout consumption。Agent 自报的指标只作为本地线索，不作为发布依据。只有通过核心评测并得到 approval 的 Candidate 才能变成不可变 Release。

运行时，Release Runtime 按固定 Release 服务请求。它先查缓存，再按 routing 调用启用的 L1/L2/L3 Artifact；Artifact accept 时必须产出完整合法输出，abstain、timeout、invalid output 或 protocol error 都会安全退回后续层或 L4。Runtime 写入经过脱敏和哈希的 Trace，Audit Monitoring 可以抽样或按风险重新调用 reference，Telemetry Evidence And Recompile 只把隐私策略允许的观测变成 ApprovedTelemetryEvidence，再经 TelemetryDataSource 进入下一轮 Snapshot。Trace 和 AuditRecord 永远不是训练数据。

当前仓库实现的是文件系统和内存版的核心架构，覆盖 target check、snapshot/reference、agent workspace、artifact worker、candidate evaluation、release runtime、runtime trace metrics、audit monitoring、telemetry recompile 和 compile orchestration。它适合开发和设计验证；持久存储、耐久队列、跨平台资源限制和真实外部 reference broker 仍是生产硬化工作。

## 迭代 1：合并内部模块，改成读者问题

删去大部分字段名和模块名，保留“先本地、没把握就交给强模型”的主线。把 L1/L2/L3、Candidate、Report、Release 等概念合并为“本地程序、测试、运行版本”。

## 迭代 2：提前一句话复述

把核心复述放到标题后第二段，让第一页先完成理解，再解释用户要提供什么、系统如何工作。

## 迭代 3：收束限制

保留当前实现状态和生产硬化限制，但不展开内部清单，只说明这是开发版以及未来要补的生产能力。

## 最终计数

| 版本 | 字数 | 术语种类数 | 相对第一版 |
| --- | ---: | ---: | ---: |
| 第一版 | 1645 | 68 | 100% / 100% |
| 最终稿 | 615 | 6 | 37.4% / 8.8% |

达标情况：

- 字数目标：最终稿 615 <= 第一版一半 822.5。
- 术语目标：最终稿 6 <= 第一版三分之一 22.7。

最终稿保留的术语种类：Darjeeling、强模型、本地程序、成本、兜底、开发版。

## 第二次修正：按用户反馈重定核心主张

用户指出上一版把重点放在“省钱且安全”上，忽略了 Darjeeling 更核心的解释：系统由 L1 到 L4 组成，这个结构来自时延、正确率和覆盖率的天然分层；核心 idea 是让系统动态地从 L4 学习，并把 L4 的能力外化到前三层。

修正动作：

- 开头 300 字内直接说明 L1-L4、三项指标和能力外化。
- 把“省钱”降为分层结果，不再作为主线。
- 明确 L4 同时是兜底层和老师，L1-L3 是逐步沉淀出的本地能力。
- 保留拒答、评测、任务边界和开发版状态，但放到主张之后。

校验结果：前 300 个正文字符已包含 L1、L2、L3、L4、时延、正确率、覆盖率、外化、前三层。修正后全文正文计数为 670 个中文/英文/数字字符。

## 第三次修正：减少重复并解释四层本体

用户指出第二次修正仍有重复，并且没有讲清楚四个层次分别是什么。新版改成“定义段 + 四层角色 + 学习循环”的结构：

- L1：最快的局部答案，查表、规则、模板或确定性程序，只答极小但几乎确定的问题。
- L2：轻量模型或较复杂程序，覆盖更多常见变体。
- L3：更强的本地模型或组合程序，处理需要推理但仍可本地完成的请求。
- L4：最强参考模型，覆盖完整任务，负责兜底，也提供学习信号。

删去重复的“越前越快、越后越强”表述，只保留一次指标解释。校验结果：前 300 个正文字符已包含 L1、L2、L3、L4、查表、轻量模型、本地模型、参考模型、时延、正确率、覆盖率和外化。修正后全文正文计数为 517 个中文/英文/数字字符。

## 第四次修正：加入图示和方法说明

用户要求利用网页发布 skill 的 Mermaid 和 LaTeX 支持，加入架构图、数据流图、核心方法示意图及说明。这部分不受前面 300 字、字数和术语种类限制，但仍保持面向第一次了解项目的人。

新增内容：

- 架构图：展示请求如何按 L1、L2、L3、L4 顺序路由，以及 L1-L3 的拒答机制。
- 数据流图：展示在线运行摘要、人工反馈、隐私过滤、冻结数据、代码代理、隐藏评测和发布之间的关系。
- 核心方法示意图：展示从 L4 完整处理任务，到提取稳定局部能力，再外化成 L1-L3 的循环。
- LaTeX 简化公式：用正确率、覆盖率和时延说明候选本地能力上线的直观门槛，同时注明实际系统还会检查失败退回、切片稳定和漂移。
