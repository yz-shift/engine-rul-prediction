# Codex 评审第一轮

> 评审者：本地 Codex CLI，只读沙箱。下面先列处理结论，后面附 Codex 原文。
> Codex 没能在自己的沙箱里运行 pytest（虚拟环境启动失败），所以它的结论来自读代码和结果文件的只读计算。

## 处理结论

| # | 意见 | 严重度 | 结论 | 怎么处理 |
|---|---|---|---|---|
| 1 | 官方测试结果影响过验证方案（改主指标、固定截顶），不能再称完全独立 | 严重 | 采纳 | README 改为说明测试集是“开发过程中看过的基准”（T-013）；这个历史无法用重跑消除 |
| 2 | 嵌套 CV 只包住了超参数搜索，特征集和权重是在外面选的 | 严重 | 采纳 | `select_pipeline` 一次选特征集、参数和权重；`nested_cross_validate` 在外层每折重跑整个流程 |
| 3 | 只看临近故障 RMSE 不够；“不是泛化问题”“一定偏向”说得太满 | 中 | 采纳 | 新增 `alerts.py`，报告首次预警余量、漏报、健康误报；分层误差放进 notebook 和 README（T-013） |
| 4 | 超参数先四舍五入再被读回训练，结果对应的不是那组配置 | 中 | 采纳 | `results.py` 的 `EXACT_KEYS` 不对模型配置取整 |
| 5 | `train_models` 的 CV 忽略自定义窗口和参数 | 中 | 采纳 | 统一走 `cross_validate_experiment` |
| 6 | 阈值 40 的依据来自旧模型 | 中 | 采纳 | 改为每个子集用嵌套预测重算预警表；界面默认值在 T-013 复核 |
| 7 | 多工况子集的 z 分数被标成物理单位 | 低 | 采纳 | 界面改标题，不写单位 |
| 8 | notebook 和 README 过时 | 中 | 采纳，放到 T-013 | T-013 修复 |
| 9 | 权重规则实现成“先取最低分再检查”，好候选会被漏掉 | 中 | 采纳 | `choose_late_weight` 改为先筛选再选最低分 |

## 修复后的结果

| 子集 | 默认 CV | 嵌套 CV | 测试 RMSE（全部 / RUL≤125） | 官方分数 | 权重 |
|---|---|---|---|---|---|
| FD001 | 14.04 | 11.32 | 13.92 / 12.83 | 273.6 | 5 |
| FD002 | 16.46 | 13.07 | 25.82 / 12.02 | 6756.1 | 5 |
| FD003 | 14.16 | 12.15 | 14.33 / 11.59 | 318.1 | 5 |
| FD004 | 16.55 | 13.76 | 26.75 / 13.09 | 5090.9 | 5 |

嵌套 CV 比默认配置 5/5 折更好。相比对称损失，权重 5 让 FD002、FD004 的官方分数变差（4845 到 6756，4015 到 5091），这是已知代价。

---

## Codex 原文

**发现了两项影响评估可信度的严重问题，以及几项可核实的代码和工程问题。** 核心的按发动机分组、折内归一化和因果窗口实现基本正确；主要风险出在完整选模流程、实验之间的配置传递，以及结果能支持多强的结论。

本次全程只读，没有修改文件或重跑完整实验。尝试运行 pytest 时，虚拟环境 Python 在启动阶段报出 `Unable to create process using ...Python312\python.exe`，因此**本次未执行测试，也不能确认当前测试全部通过**。以下结论来自代码调用链、已有结果文件，以及对原始数据和 JSON 的只读计算。

先说明核实通过的部分：

- 未发现同一台发动机同时进入一折的训练和验证；工况聚类、归一化、方差筛选及标准化都在训练折上拟合。
- 窗口按发动机计算，只使用当前及过去读数；官方测试答案没有进入预测特征。
- 最新 T-010、T-011 日志的主要数字与 JSON 一致。用保存的逐折结果重新相减，四个子集确实都是搜索方案 **5/5 折优于默认方案**。
- `nested_cross_validate()` 内部的外层、内层数据边界是正确的。下面指出的是它**没有覆盖完整选择流程**，不是说函数内部直接混入了外层验证行。

**严重**

**1. 官方测试结果已经反馈到验证方案和搜索范围，不能再称为完全独立的最终测试。**

**位置：**[PROGRESS.md:16](D:/projects/engine-rul-prediction/PROGRESS.md:16)、[PROGRESS.md:51](D:/projects/engine-rul-prediction/PROGRESS.md:51)、[evaluate.py:36–42](D:/projects/engine-rul-prediction/src/rul/evaluate.py:36)、[tuning.py:49–64](D:/projects/engine-rul-prediction/src/rul/tuning.py:49)、[README.md:15](D:/projects/engine-rul-prediction/README.md:15)。

**问题与核实：**实验日志明确记录了两次查看测试结果后的调整：先因 CV 与官方测试差距很大，把主验证指标改为只统计 `RUL≤125`；后来因搜索出的 `cap=100` 在官方测试上变差，把搜索范围改为固定 `cap=125`。现行代码确认这两项调整就是当前的选择协议。

这里能确认的是**日志记录了测试反馈，代码采用了调整后的协议**；没有发现代码直接将官方测试样本送进 `.fit()`。但“只改验证指标，没有改模型”不能消除影响，因为后续模型正是按这个指标选出来的。

**后果：**README 的“官方测试集只用来报告”不完整。现有测试分数仍是有效的基准测量，但不能作为整个开发过程从未接触过的独立泛化证明，也不能定量判断测试反馈带来了多少偏差。

**建议：**明确把这些成绩标为开发过程中已查看过的基准结果。冻结验证目标、候选空间和决策规则后，再用未参与任何选择的新发动机或独立数据检验；重新打乱已有 CV 折不能抹去历史反馈。

**2. 嵌套 CV 只覆盖中间的超参数搜索，最终模型的成绩仍复用了选模数据。**

**位置：**[scripts/tune_models.py:40–54](D:/projects/engine-rul-prediction/scripts/tune_models.py:40)、[experiments.py:155–160](D:/projects/engine-rul-prediction/src/rul/experiments.py:155)、[scripts/tune_asymmetric.py:31–55](D:/projects/engine-rul-prediction/scripts/tune_asymmetric.py:31)、[scripts/tune_asymmetric.py:75–88](D:/projects/engine-rul-prediction/scripts/tune_asymmetric.py:75)、[app/streamlit_app.py:178–182](D:/projects/engine-rul-prediction/app/streamlit_app.py:178)。

**问题与核实：**当前流程是：

1. 在全部训练发动机的 CV 上选择 `default.features`。
2. 固定这个已选特征集，进行嵌套超参数搜索。
3. 用全部训练数据 CV 选出的超参数，在同样五折上比较不对称权重。
4. 把获胜权重的那一行成绩直接写成 `final.cv`，并在界面展示。

例如 FD001 的 [tuning.json:21](D:/projects/engine-rul-prediction/results/FD001/tuning.json:21) 是嵌套 CV **11.9437**；[final.json:54](D:/projects/engine-rul-prediction/results/FD001/final.json:54) 的 **11.2839** 则直接等于选权重表中 `weight=5` 的成绩。后者没有经过涵盖最终权重选择的外层评估。上游特征选择也已经看过外层发动机的验证表现。

**后果：**不能把最终 CV 改善解释成“嵌套 CV 已独立确认最终模型的提升”。即使权重选择规则事先写好，从多个候选中选优仍会产生选择偏差。

**建议：**每个外层训练集中完成特征方案、超参数、损失权重的全部选择，再评估外层验证发动机。暂时应把 `final.cv` 标为“用于选择的 CV，可能偏乐观”，并明确已有嵌套结果的适用范围。现有 [test_tuning.py:73–90](D:/projects/engine-rul-prediction/tests/test_tuning.py:73) 只检查局部函数的数据边界，需要增加覆盖完整选择流程的测试。

**一般**

**3. 主指标没有覆盖完整维修代价；“筛到同一区间即可直接比较”和“不是泛化问题”的判断过强。**

**位置：**[evaluate.py:36–42](D:/projects/engine-rul-prediction/src/rul/evaluate.py:36)、[evaluate.py:59–68](D:/projects/engine-rul-prediction/src/rul/evaluate.py:59)、[evaluate.py:78–90](D:/projects/engine-rul-prediction/src/rul/evaluate.py:78)、[asymmetric.py:87–94](D:/projects/engine-rul-prediction/src/rul/asymmetric.py:87)、[PROGRESS.md:23](D:/projects/engine-rul-prediction/PROGRESS.md:23)。

**问题与核实：**CV 对完整验证轨迹的所有 `RUL=0…125` 周期评分，官方测试则每台只取一个截断末点。读取原始数据确认，四个子集的训练最短寿命都超过 125，因此 CV 每台固定贡献 126 点，RUL 均值为 **62.5**。但 FD001 测试近故障区间的 89 台发动机，RUL 均值为 **68.20**；其中 `RUL≤30` 占 **28.09%**，CV 对应比例是 **24.60%**。筛选相同区间并没有统一采样分布或观测历史长度。

同时，选权重时使用的 RMSE 和 NASA 分数都只覆盖 `RUL≤125`，健康阶段的过早报警代价没有进入采用规则。已有结果确实体现了这种取舍：

| 子集 | 调参后对称模型测试 NASA 分数 | 最终不对称模型测试 NASA 分数 | 变化 |
|---|---:|---:|---:|
| FD002 | 4845.0258 | 7078.2873 | 恶化 46.09% |
| FD004 | 4015.2691 | 5013.8020 | 恶化 24.87% |

日志如实披露了这些数字，因此这不是结果算错；问题是**近故障误差改善不足以证明整体维修策略改善**。

**后果：**`PROGRESS.md:23` 的“不是泛化问题，是口径不同”排除了尚未排除的可能性。模型可以在主指标上更好，同时在其他寿命区间付出明显代价。

**建议：**把结论改为“口径差异解释了整体差距的一部分”，按 RUL 区间及观测长度分层报告。若目标是维修决策，还应评估完整轨迹上的漏报、健康期误报、首次报警提前量及成本，而非仅依赖条件 RMSE。也不应把“当前实验中低截顶获胜”写成“该指标一定偏向更低截顶”的普遍结论。

**4. 保存结果时把超参数也舍入，后续实验使用的配置已发生变化。**

**位置：**[results.py:11–23](D:/projects/engine-rul-prediction/src/rul/results.py:11)、[tuning.py:58–63](D:/projects/engine-rul-prediction/src/rul/tuning.py:58)、[scripts/tune_models.py:63–74](D:/projects/engine-rul-prediction/scripts/tune_models.py:63)、[scripts/tune_asymmetric.py:33–36](D:/projects/engine-rul-prediction/scripts/tune_asymmetric.py:33)。

**问题与核实：**`rounded()` 递归舍入整个结果字典，学习率、采样比例和正则系数也被保留为四位小数。T-010 用原始精度参数训练和评分，写 JSON 时修改了参数；T-011 又读回这些舍入值训练。

已有结果也显示，搜索阶段与下一阶段理论上对应的 `weight=1` 基线并不相同：

| 子集 | 搜索最佳 CV RMSE | 下一阶段 weight=1 CV RMSE |
|---|---:|---:|
| FD001 | 11.9437 | 11.9187 |
| FD002 | 13.9045 | 13.8763 |
| FD003 | 12.7020 | 12.6182 |
| FD004 | 14.6176 | 14.6712 |

**参数被修改是代码可确定的事实；本次无法运行隔离实验，不能把表中全部差异都归因于舍入。**

**后果：**`tuning.json` 保存的配置不是产生其成绩时的精确配置。T-010 与 T-011 的比较同时涉及参数精度变化和损失变化，无法严格归因于损失函数。

**建议：**机器读取的配置保留完整浮点精度，仅对展示指标舍入。补充“配置保存再加载完全相等”的测试，以及使用实际搜索参数验证普通 LightGBM 与 `weight=1` 等价的测试。

**5. `train_models.run()` 的 CV 忽略自定义窗口和模型参数，最终测试却应用了它们。**

**位置：**[scripts/train_models.py:21–40](D:/projects/engine-rul-prediction/scripts/train_models.py:21)、[experiments.py:129–134](D:/projects/engine-rul-prediction/src/rul/experiments.py:129)。

**问题与核实：**CV 调用 `build_model(experiment.model, SEED)` 和 `build_featurizer(experiment.features, experiment.subset)`，遗漏 `experiment.model_params` 和 `experiment.window`；随后 `fit_final(experiment, train)` 却正确传递了它们。

因此，传入 `window=50、num_leaves=7` 时，CV 会使用默认的 `window=30、num_leaves=15`，测试使用指定配置。这由调用参数即可核实。

**后果：**复用此函数评估非默认配置时，同一个结果文件会混合两个不同模型的 CV 和测试成绩。**当前批量实验列表恰好使用默认配置，没有证据表明现存默认实验结果已受此问题影响。**

**建议：**统一从 `Experiment` 构建模型和特征器，让 CV、最终训练共享配置转换逻辑。增加一条携带非默认窗口和参数的集成测试，确认两条路径一致。这也能解决当前多处重复构建流水线造成的实际配置漂移。

**6. 阈值 40 的证据来自旧 FD001 模型，却被沿用到新模型和四个子集。**

**位置：**[app/streamlit_app.py:18–19](D:/projects/engine-rul-prediction/app/streamlit_app.py:18)、[app/streamlit_app.py:163–169](D:/projects/engine-rul-prediction/app/streamlit_app.py:163)、[02_modeling.ipynb:856](D:/projects/engine-rul-prediction/notebooks/02_modeling.ipynb:856)、[02_modeling.ipynb:867–874](D:/projects/engine-rul-prediction/notebooks/02_modeling.ipynb:867)。

**问题与核实：**应用仍引用旧 notebook 的“阈值 40 时首次报警最晚还剩 26 周期”。该输出来自旧 FD001 默认模型的 100 台发动机；当前四个子集的最终模型均采用窗口 50、权重 5。notebook 的 OOF 代码也没有传递新的 `model_params`。

**后果：**旧的报警提前量证据与当前演示模型不对应，不能据此推断新模型或 FD002–FD004 也有相同表现。这里没有证据证明阈值 40 一定不好，问题是验证尚未跟上模型变化。

**建议：**按各子集最终完整流程重新生成外层 OOF 预测，核实首次报警提前量、漏报和健康期误报。在此之前可以保留 40 作为演示值，但应明确它尚未针对当前模型重新验证。

**7. 传感器图把归一化后的 z 分数标成了物理量。**

**位置：**[app/streamlit_app.py:47–48](D:/projects/engine-rul-prediction/app/streamlit_app.py:47)、[app/streamlit_app.py:109–116](D:/projects/engine-rul-prediction/app/streamlit_app.py:109)、[app/streamlit_app.py:205–212](D:/projects/engine-rul-prediction/app/streamlit_app.py:205)、[features.py:53–60](D:/projects/engine-rul-prediction/src/rul/features.py:53)。

**问题与核实：**图表使用经过 `ConditionNormalizer` 转换的 `table`，标题却保留 `°R`、`psia`、`rpm`。FD003 使用单工况归一化，但提示条件要求工况数大于 1，因此连 z 分数说明也没有。

按原始 FD003 数据重算，第一条测试 `sensor_3` 为 **1581.93 °R**，训练均值约 **1588.079175**、样本标准差约 **6.810418**，传给图表的值约为 **−0.902907**；标题仍是 HPC 出口温度 `°R`。

**后果：**不影响 RUL 指标，但展示的物理含义错误，会直接误导演示和面试中的传感器解释。

**建议：**保留原始数据用于传感器展示，或者所有 `condition_*` 图表都标为无量纲 z 分数、去掉物理单位。补充 FD003 图表标签测试。

**建议**

**8. 修复当前复现入口：notebook 已存在确定的接口断裂，README 也未对应最终模型。**

**位置：**[02_modeling.ipynb:270](D:/projects/engine-rul-prediction/notebooks/02_modeling.ipynb:270)、[02_modeling.ipynb:283](D:/projects/engine-rul-prediction/notebooks/02_modeling.ipynb:283)、[README.md:99–102](D:/projects/engine-rul-prediction/README.md:99)、[README.md:139](D:/projects/engine-rul-prediction/README.md:139)。

**问题与核实：**notebook 首个代码单元导入当前已不存在的 `prepare_table`，随后读取已迁移的根目录 `results/summary.csv`；静态检查即可确定当前不能从头运行。README 仍称“没有做超参数搜索”，其“重跑全部实验”命令也只覆盖默认实验，不会执行两个调参脚本。

**后果：**别人按仓库说明复现时，无法得到当前最终模型及其分析。这是功能性复现缺口，不是文档风格问题。

**建议：**在已安排的 T-013 中更新 notebook 接口、结果路径和完整复现顺序，并将旧默认模型结果与当前最终模型结果明确对应。考虑到任务清单已经承认这项未完成工作，我将其列为收尾建议，而非新的严重缺陷。

**9. 不对称权重规则可以改成“先满足约束，再选择最佳分数”。**

**位置：**[asymmetric.py:87–94](D:/projects/engine-rul-prediction/src/rul/asymmetric.py:87)、[test_asymmetric.py:71–81](D:/projects/engine-rul-prediction/tests/test_asymmetric.py:71)。

**问题与核实：**当前先挑分数最低者，再检查 RMSE 门槛。例如基线为 `(score=1000, RMSE=20)`，权重 2 为 `(850,20.5)`，权重 5 为 `(700,22)`：代码会因权重 5 的 RMSE 不合格而退回基线，忽略满足全部门槛的权重 2。

**后果：**可能放弃一个合格改进方案。该实现符合文件开头写下的规则，因此不是实现违背规格；目前四个子集的权重 5 均通过门槛，现有选择未受影响。

**建议：**若实际目标是带 RMSE 约束的最优分数，先过滤满足两个门槛的候选，再选择分数最低者；补充上述多候选边界测试。

如果我是面试官，最可能追问三个薄弱点：**① 为什么没有把测试数据放进训练，仍可能污染测试独立性，改变验证指标如何影响后续选模；② 你的嵌套 CV 究竟覆盖了哪些选择，如何为最终“特征＋参数＋损失权重”流程提供独立评估；③ 当近故障 RMSE 改善、总体 NASA 分数却恶化时，你凭什么判断维修策略更好，如何量化漏报、过早维修和提前量的实际成本。**
