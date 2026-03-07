# GAIA2

一、它的 benchmark 到底是什么：ARE 平台 + Gaia2 基准集
这个仓库是 Meta Agents Research Environments（ARE），一个用来“执行/回放/验证”代理任务的动态仿真平台；ARE 的定位强调动态环境（事件会随时间推进发生）、多步任务（通常 10+ 步、持续数分钟）、以及更接近真实助理工作流的多应用协作。 ￼

Gaia2 是构建在 ARE 上的 agent benchmark。它相对早期 GAIA（偏 web 搜索/检索）扩展到“读写式、多应用工作流、时间约束、噪声与失败、以及多代理协作”等维度。官方 tutorial 明确把 Gaia2 定义为评测 general agent capabilities 的综合套件

二、Gaia2 的组成：Universes、Apps、Scenarios、Capabilities

1）规模与结构

官方 Gaia2 评测指南给出：

- 10 个 Universes：每个 universe 是一个“预置的用户世界/背景”，包含该用户的邮件、联系人、日历、文件等。
- 11 个核心应用（Apps）：包括 AgentUserInterface（与代理交互的入口）+ 多种消息应用（MessagingApp、ChatsApp、EmailClient 等）+ 工具类应用（Calendar、Contacts、RentAFlat、Shopping、Cab、City、FileSystem）。
- 800 个动态 Scenarios：每个 scenario 是一个带时间推进的仿真任务，包含事件（events）、任务目标、以及验证逻辑。

另外，文档里还提到 Gaia2-mini：作为小规模基准子集，用于更快地迭代验证（文档口径是 160 条，且均匀覆盖若干能力维度）。

2）它评测哪些能力（Capabilities）

官方 Gaia2 指南把核心能力分为 7 类，并且说明最终分数“等权”聚合：

- Execution：多步、可改变状态的操作序列（计划与执行一致性）。
- Search：跨来源信息检索/聚合（在仿真世界的多应用数据中找答案）。
- Adaptability：环境变化/新信息出现后，能否调整计划并完成目标。
- Time：时间约束与调度（如等待若干分钟无回复就走默认分支）。
- Ambiguity：识别不可解/矛盾/不明确指令并向用户澄清。
- Agent2Agent：与“应用方代理”协作（不是直接调用 API，而是通过对话/协商来让应用代理代办）。
- Noise：在 API 不稳定、随机失败、环境扰动下的鲁棒性。

其中 Execution/Search/Adaptability/Time/Ambiguity 是标准评测主轴；Agent2Agent/Noise 在文档中被描述为 augmentation capabilities（在 Gaia2-mini 上评）。

三、数据与场景格式：Scenario JSON / Trace 到底长什么样

ARE 把“场景定义 + 执行轨迹”统一成一个标准化 JSON（Pydantic 模型定义），文档称其核心根结构为 ExportedTrace。

1）ExportedTrace 根字段（关键点）

文档给出的结构（核心字段）：

- metadata：场景元信息（scenario_id、seed、duration、start_time、hints、tags 等）。
- apps：该场景可用应用列表；每个 app 包含 name、class_name、app_state（或序列化状态）。
- events：计划事件（含 event_type、class_name 等）。
- completed_events：执行后已完成事件记录。
- world_logs：世界/代理日志（可选）。
- version、context、augmentation 等辅助信息。

2）Apps 的含义

apps 不是“工具列表的文字描述”，而是可执行的仿真应用实例（Email、Calendar、FileSystem 等）及其初始状态。也就是说，你的 agent 不是在真实 Gmail/Calendar 上操作，而是在“可验证的仿真世界状态机”里操作；这也是它能做精确验证与可复现实验的基础。

3）Oracle events（用于验证/开发）

Gaia2 的 validation split 提供 oracle events（可理解为“参考解的工具调用事件图”），用于 judge 做在线/离线验证与即时反馈；而 test 集对外是私有的。

四、怎么跑 benchmark：CLI、配置、三段式 pipeline、并发与多次运行

1）安装/运行入口

仓库 README 给出三类命令：

- are-run 跑单个 scenario
- are-benchmark 做评测（run/judge/gaia2-run）
- are-gui 打开交互式 GUI

并推荐用 uvx --from meta-agents-research-environments ... 免安装直接运行命令。

2）are-benchmark 三个主命令分别干什么

Benchmarking 指南写得很明确：

- run：执行场景、采集 traces、并（在 validation 上）触发在线 judge。
- judge：对已有 traces 做离线验证（对照 oracle/ground truth）。
- gaia2-run：一键跑 Gaia2 榜单所需的全流程（自动跑多个 configs + 多阶段 + 固定多次运行）。

3）关键参数（你真正会用到的）

Benchmarking 文档给了完整参数表，核心包括：

- 数据来源：--dataset（本地 JSON/JSONL）或 --hf-dataset / --hf-config / --hf-split / --hf-revision（Hugging Face 数据集）。
- 运行控制：--limit、--scenario_timeout、--max_concurrent_scenarios、--executor_type（thread/process）、--enable_caching。
- 输出：--output_dir、--trace_dump_format（hf/lite/both）、--hf_upload、--hf_public。
- 模型：--model、--provider（LiteLLM providers 列表很长）、--endpoint（本地/自建 OpenAI-compatible endpoint）。
- Augmentation：--noise，以及 --a2a_app_prop（>0 则启用 Agent2Agent 模式，按比例把 apps 换成“应用代理”）。

五、怎么判分：Judge 系统的机制、事件匹配、因果约束、时间容差
ARE 的验证不是只看最终文本答案，而是以“参考工具调用事件图（oracle events）”为核心对齐对象，做事件级验证。Judge 文档里把主 judge 实现描述为 GraphPerEventJudge，流程包括： ￼
1）Preliminary checks：先核对工具调用计数（tool call counts）是否与 oracle 一致（或符合约束）。
2）Event matching：尝试把每个 oracle event 匹配到 agent 的某个事件（通常是工具调用及其参数）。
3）Causality verification：检查依赖拓扑顺序，防止 agent “先做后因”。
4）Tool validation：针对具体 tool 做参数级/语义级校验（部分场景需要语义等价而非严格字符串匹配）。

时间相关的验证还有明确的“比较类型”和“容差窗”：EQUAL/LESS_THAN/GREATER_THAN，并配置 pre/post tolerance（例如允许在 oracle 时间点前 5 秒、后 20 秒等窗口内算通过）。 ￼

此外，Benchmarking 文档强调 judge 系统的 LLM 与被测 agent 的 LLM 配置是独立的：judge LLM 用于语义验证/软匹配，而硬规则验证不依赖 LLM。

## System Input

- general_instructions
    - 这里定义的是“身份”和“价值观”。例如模型叫什么名字、属于什么环境、要保持有帮助、无害、诚实、优先保证准确性等。这一部分不涉及具体工具或流程，而是规定行为边界与基本原则，相当于“行为哲学层”。
- 第二层是执行协议（agent_instructions）。
    - 这是最关键的结构控制层。它严格定义：
    - 必须按 Thought → Action → Observation 的循环执行
    - Thought 里只能写推理，不得包含工具调用
    - Action 必须是严格 JSON，且一次只调用一个工具
    - 不能生成 Observation
    - JSON 结构必须精确
    - 布尔值、空参数、结束标记等都有硬性规范
- 第三层是环境说明（environment_instructions）。这一部分说明模型所处的是一个“虚拟个人工作空间”，并给出环境特征，例如：
    - 环境是动态的
    - 用户拥有完全控制权
    - 可以访问多个应用
    - 代用户写作时必须 impersonate
    - 然后是一个非常长的工具列表。这不是简单的枚举，而是标准化的 API 描述，每个工具包含：
        - 所属 app
        - 功能描述
        - 输入参数 schema（类型、默认值、是否必须）
        - 返回值类型
        - 这一块本质上是一个“接口文档嵌入模型上下文”的设计。它让模型在上下文中直接拥有工具规范，从而能生成结构正确的 JSON 调用。
- 第四层是执行规则（FUNDAMENTAL RULES FOR TASK EXECUTION）。这里规定了高层任务策略，例如：
    - 只能在完成或失败时和用户通信
    - 不允许进度更新
    - 必须完全执行
    - 模糊部分要先执行明确部分
    - 优先用工具获取信息
- 第五层是通知机制（Notification policy）。这部分定义：
    - 哪些系统事件会触发通知
    - 哪些工具调用会生成通知
    - wait_for_notification 的行为
    - 这是一个“事件驱动模型”设计，使 agent 可以在异步环境中运行。
    - 最后是当前时间上下文。例如：Today’s date in ‘YYYY-MM-DD HH’ format is 2024-10-15 07,这属于运行时上下文注入，让模型有一个确定的时间基准。