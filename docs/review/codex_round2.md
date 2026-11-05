# Codex 评审第二轮

> 评审对象是提交 76d3d4e。Codex 的结论：当前代码没有严重问题，有 1 项中等、3 项低严重度问题。
> Codex 仍然无法在它的沙箱里运行 pytest，也没有重新训练，所以它只核对了代码和已保存的数字。

## 处理结论

| # | 意见 | 严重度 | 结论 | 怎么处理 |
|---|---|---|---|---|
| 1 | 回退到权重 1 时，返回普通 LightGBM，记录的却是包装器的 CV 成绩 | 中 | 采纳 | `Selection.selected_cv` 记录真正返回的模型自己的成绩；规则的比较基线仍是包装器权重 1，因为这样所有权重的随机流相同；补了测试 |
| 2 | `configurations_cv` 里的候选参数仍被四舍五入 | 低 | 采纳 | 搜索记录改为把参数放在受保护的 `model_params` 下 |
| 3 | 嵌套 CV 测试没有检查外层最终拟合；配置测试一次改多项 | 低 | 采纳 | 新增外层 `fit_final` 的监视测试，并手工改成泄漏版本，确认测试会失败；窗口和参数分开断言。没有另外测 `train_models.run()` 入口：它只是一行调用 `cross_validate_experiment` |
| 4 | 注释说测试集筛到 RUL≤125 后“可以直接比较”，说得太满 | 低 | 采纳 | 改了 `evaluate.py` 的说明：RUL 范围相同，但分布不同，只能作参考对照；`score_per_100` 和测试总分也不能直接比较 |

## 需要更正的一处 Codex 说法

Codex 说“当前四个子集及保存的所有外折均选择权重 5”。这不准确：FD004 的第 3 个外折回退到了普通 LightGBM（`results/FD004/tuning.json` 的 `chosen_per_fold`）。

这不影响嵌套 CV 的数字，因为外层成绩一直来自真正返回的模型。第 1 条问题只影响 `selection_cv_optimistic` 的记录，而最终选择四个子集都是权重 5，没有用到回退分支。

## 另外记下的结论边界（写进 README，T-013）

- 嵌套 CV 估计的是选择流程，不是已经固定的最终模型；内层是 3 折，全量选择是 5 折
- “5/5 折更好”只是这组划分上的描述，没有做显著性检验
- 不对称损失不是所有指标都变好：FD002 官方分数变差 39%，FD004 变差 27%；FD004 阈值 40 下过早预警和健康期误报也略有增加
- 官方测试集是开发过程中看过的基准，不是独立证明

---

## Codex 原文

**当前代码中未发现严重问题。** 第一轮最关键的“嵌套 CV 没有覆盖完整选择流程”已经落实修复；本轮仍发现一项中等问题，以及配置记录、测试覆盖和指标解释方面的低严重度问题。历史上使用官方测试结果调整方案的事实仍然限制结论，重跑不能恢复测试集的独立性。

评审对象是当前提交 `76d3d4e`。我读取了 `src/rul/`、`scripts/`、`app/`、`tests/`，并交叉核对了四个子集的 `final.json`、`tuning.json` 及相关默认实验结果，全程没有修改文件。

本轮尝试运行 pytest，禁用了字节码和 pytest 缓存写入，但虚拟环境 Python 在启动阶段失败：

```text
Unable to create process using '"C:\Users\hoeyz\AppData\Local\Programs\Python\Python312\python.exe" ...'
```

因此，**本轮未执行测试，不能独立确认日志中的“113 个测试全过”**。也没有重新训练完整实验；下文明确区分代码与文件一致性核查、保存数字的重算，以及尚未独立复现的训练结果。

1. **【中】回退到权重 1 时，实际返回的模型与记录的选择 CV 成绩不对应。**

   **位置：**[tuning.py:165](/D:/projects/engine-rul-prediction/src/rul/tuning.py:165)、[tuning.py:172](/D:/projects/engine-rul-prediction/src/rul/tuning.py:172)、[tune_models.py:67](/D:/projects/engine-rul-prediction/scripts/tune_models.py:67)、[tune_models.py:88](/D:/projects/engine-rul-prediction/scripts/tune_models.py:88)。

   **问题：**权重比较表里的所有候选，包括 `late_weight=1`，都用 `LateWeightedLightGBM` 自定义目标包装器进行评估。但当规则决定不采用不对称损失时，代码返回普通 `lightgbm` 的 `symmetric` 实验；脚本仍把包装器的权重 1 成绩写进 `selection_cv_optimistic`。

   这两个实现不能按当前配置视为完全相同。现有结果已经给出反例：FD004 同一组参数、同一 CV 划分下：

   | 实现 | 近故障 CV RMSE | score/100 |
   |---|---:|---:|
   | 普通 LightGBM | 14.6176 | 478.8094 |
   | 自定义目标，权重 1 | 14.6713 | 487.5284 |

   证据见 [FD004 tuning.json:258](/D:/projects/engine-rul-prediction/results/FD004/tuning.json:258) 的 `cap=125` 对照与 [同文件:220](/D:/projects/engine-rul-prediction/results/FD004/tuning.json:220) 的权重 1 记录。

   **后果：**一旦没有不对称候选通过门槛，结果文件会把一种模型的选择 CV 和另一种模型的最终测试成绩放在一起；规则使用的对照也不是实际回退模型。

   **影响范围：当前四个子集及保存的所有外折均选择权重 5，因此没有证据说明现存四份 `final.json` 已受这个回退分支影响。**

   **建议：**权重 1 时也返回实际评估过的包装器；或者明确使用普通 LightGBM 作为回退模型，并记录它自己的成绩。补一个强制回退的流程测试，核对保存的 CV 与实际返回模型一致。现有 [test_tuning.py:113](/D:/projects/engine-rul-prediction/tests/test_tuning.py:113) 只断言回到普通模型，没有检查成绩对应关系。

2. **【低】第一轮的参数精度问题只修复了一部分：完整搜索表仍保存了舍入后的候选参数。**

   **位置：**[results.py:12](/D:/projects/engine-rul-prediction/src/rul/results.py:12)、[results.py:17](/D:/projects/engine-rul-prediction/src/rul/results.py:17)、[tuning.py:146](/D:/projects/engine-rul-prediction/src/rul/tuning.py:146)。

   **问题：**`EXACT_KEYS` 保护了 `model_params` 和 `chosen_per_fold`，但 `configurations_cv` 把参数直接展开到每行顶层，仍会被递归舍入。

   例如 FD001 获选配置的学习率，在 [tuning.json:107](/D:/projects/engine-rul-prediction/results/FD001/tuning.json:107) 是：

   ```text
   0.01713718503144242
   ```

   同一配置在完整搜索表 [tuning.json:772](/D:/projects/engine-rul-prediction/results/FD001/tuning.json:772) 中却是 `0.0171`；`subsample`、`colsample_bytree` 和 `reg_lambda` 也有相同问题。

   **后果：**直接按搜索表里的参数重放候选时，使用的配置不再是产生该行成绩的精确配置，削弱搜索记录的可审计性。

   **当前最终模型的 `model_params`、对称模型配置和外折获选配置已经保留完整精度。不能再沿用第一轮结论，声称当前最终训练仍在读取舍入参数。**

   **建议：**把搜索表中的配置也放入受保护的 `model_params`，或分开保存精确配置与展示指标。现有 [test_alerts_and_results.py:51](/D:/projects/engine-rul-prediction/tests/test_alerts_and_results.py:51) 只测受保护键，没有覆盖真实的 `configurations_cv` 结构。

3. **【低】关键回归测试仍不能充分发现“外层最终拟合泄漏”和“只丢失一部分配置”的回归。**

   **位置：**[test_tuning.py:119](/D:/projects/engine-rul-prediction/tests/test_tuning.py:119)、[test_alerts_and_results.py:97](/D:/projects/engine-rul-prediction/tests/test_alerts_and_results.py:97)。

   **问题与证据：**

   - 嵌套 CV 测试只记录 `select_pipeline()` 收到了哪些发动机，没有检查外层 `fit_final()` 实际拟合了哪些发动机。它能发现“把全部数据交给选择器”，却没有断言阻止“选择正确，但最终外层模型又用全部数据拟合”。当前 [tuning.py:201](/D:/projects/engine-rul-prediction/src/rul/tuning.py:201) 正确使用 `train`；这里报告的是测试缺口，**不是现存泄漏**。
   - 自定义配置测试同时改变窗口和多项模型参数，最终只断言自定义成绩与默认成绩不同。假如仅丢失窗口，其他模型参数仍可能使成绩不同；反之亦然。因此这个断言不足以分别证明窗口与参数都传递正确。

   **后果：**即使这些测试通过，也不能据此宣称上述两类回归已经被完整防住。

   **建议：**逐折检查最终模型、特征器实际拟合的发动机集合，与预测集合无交集；对窗口和模型参数分别断言每折实际构造值，并覆盖 `train_models.run()` 的入口传递。

   这是对现有断言的静态核查；**本轮未实际执行变异测试**。

4. **【低】第一轮关于“筛到相同 RUL 区间即可直接比较”的过强解释，仍留在核心代码中。**

   **位置：**[evaluate.py:124](/D:/projects/engine-rul-prediction/src/rul/evaluate.py:124)，另见 [evaluate.py:44](/D:/projects/engine-rul-prediction/src/rul/evaluate.py:44)。

   **问题：**说明仍写，测试集筛到 `RUL≤125` 后与 CV “口径一致，两者才能直接比较”。实际实现是：

   - CV：每台完整验证轨迹的全部 `RUL=0…125` 周期。
   - 官方测试：每台发动机截断轨迹的最后一个周期，再筛选 `RUL≤125`。

   本轮只读重算原始数据确认：四子集训练发动机都足够长，所以 CV 每台贡献 126 点，RUL 均值固定为 **62.5**。FD001 测试近故障区间则是 **89 台、每台一点，RUL 均值约 68.20**；`RUL≤30` 的比例分别为 **24.60%** 和 **28.09%**。筛选相同范围没有统一 RUL 分布或观测历史长度。

   此外，`score_per_100` 是近故障区间的归一化分数，而 `test.score` 是全部测试发动机的总分，也不能直接作为同一种统计量比较。

   **后果：**指标本身没有因此算错，但这段解释仍可能让作者在面试中错误地认为已经排除了分布差异或泛化差异。

   **建议：**改为“在相同 RUL 区间报告不同采样方案下的误差，可作参考对照，但不是同分布估计”。比较 NASA 分数时，还需明确样本范围和归一化方式。

逐条回看第一轮，处理落实情况如下：

| 第一轮意见 | 本轮核实结论 | 实际依据 |
|---|---|---|
| **#1 官方测试反馈影响过方案** | **已承认，历史限制仍然存在。** 不能标成通过重跑“消除”。 | [evaluate.py:15](/D:/projects/engine-rul-prediction/src/rul/evaluate.py:15) 和 [tuning.py:11](/D:/projects/engine-rul-prediction/src/rul/tuning.py:11) 已明确记录这一限制。README 更新按你的要求不重复报告。 |
| **#2 嵌套 CV 未覆盖完整选择** | **核心修复已落实。** | [tuning.py:196](/D:/projects/engine-rul-prediction/src/rul/tuning.py:196) 在每个外层训练折内调用完整 `select_pipeline()`，随后才拟合并预测外层验证发动机；特征方案、参数和权重都在边界内选择。 |
| **#3 维修评价不完整、解释过强** | **部分落实。** 新预警指标真实存在；上述第 4 条解释残留仍在。 | [alerts.py:24](/D:/projects/engine-rul-prediction/src/rul/alerts.py:24) 实现首警、未报警、过晚和健康期报警统计，且写入结果文件。分层展示属于后续任务。 |
| **#4 参数保存时丢精度** | **最终训练配置已修，完整搜索记录未修全。** | `final.model_params`、`symmetric.model_params` 和 `chosen_per_fold` 保精度；详见本轮第 2 条。 |
| **#5 CV 忽略窗口和模型参数** | **当前实现已修复。** | [evaluate.py:95](/D:/projects/engine-rul-prediction/src/rul/evaluate.py:95) 与 [experiments.py:135](/D:/projects/engine-rul-prediction/src/rul/experiments.py:135) 共用 `Experiment.make_model()`／`make_featurizer()`；[train_models.py:22](/D:/projects/engine-rul-prediction/scripts/train_models.py:22) 已切换到该入口。 |
| **#6 阈值 40 引用旧模型证据** | **新证据已生成，旧依据已移除。** | [tune_models.py:61](/D:/projects/engine-rul-prediction/scripts/tune_models.py:61) 取得嵌套外层预测，`:90` 生成 `alerts_nested`；[app:18](/D:/projects/engine-rul-prediction/app/streamlit_app.py:18) 已不再引用旧“26 周期”结论。阈值最优性仍未得到证明。 |
| **#7 z 分数误标物理单位** | **已修复。** | [app:110](/D:/projects/engine-rul-prediction/app/streamlit_app.py:110) 为归一化图标注 z-score；[app:209](/D:/projects/engine-rul-prediction/app/streamlit_app.py:209) 至 `:214` 对单工况显示原始数据，解决了 FD003 的错配。 |
| **#8 notebook／README 过时** | **按你的要求排除，不再次报告。** | 已安排 T-013。 |
| **#9 先取最低分再检查约束** | **已修复，且补了对应反例测试。** | [asymmetric.py:94](/D:/projects/engine-rul-prediction/src/rul/asymmetric.py:94) 先筛选合格候选；[test_asymmetric.py:76](/D:/projects/engine-rul-prediction/tests/test_asymmetric.py:76) 覆盖“最低分不合格、次优合格”的场景。 |

结果文件方面，以下核对通过：

- 四个子集各有 **42 条配置记录**，对应两个特征方案乘以“默认配置＋20 个随机候选”，以及五个外折配置。
- `final.nested_cv` 与 `tuning.nested_cv` 的摘要一致。
- 从保存的逐折 RMSE 重算均值、样本标准差和相对默认模型的配对差，均与文件一致。
- 四个子集确实都是 **5/5 外折的近故障 RMSE 低于所保存的默认模型**。
- 最终权重 5 均满足当前规则：相对包装器权重 1，近故障分数至少改善 10%，且近故障 RMSE 没有超出允许恶化范围。
- 未发现同一发动机在一折中同时进入训练和验证；工况聚类、归一化、方差筛选和标准化均位于训练边界内。窗口按发动机计算，只使用当前及历史读数。官方测试答案没有进入预测特征。

当前保存的主要结果如下，第一轮处理文档的新结果摘要与这些文件相符：

| 子集 | 默认 CV 近故障 RMSE | 嵌套 CV 近故障 RMSE | 官方测试 RMSE：全部／近故障 | 官方测试总分 |
|---|---:|---:|---:|---:|
| [FD001](/D:/projects/engine-rul-prediction/results/FD001/final.json:20) | 14.0391 | 11.3210 | 13.9210／12.8341 | 273.6215 |
| [FD002](/D:/projects/engine-rul-prediction/results/FD002/final.json:20) | 16.4587 | 13.0681 | 25.8175／12.0206 | 6756.1023 |
| [FD003](/D:/projects/engine-rul-prediction/results/FD003/final.json:20) | 14.1616 | 12.1472 | 14.3250／11.5931 | 318.0530 |
| [FD004](/D:/projects/engine-rul-prediction/results/FD004/final.json:20) | 16.5461 | 13.7565 | 26.7502／13.0854 | 5090.9014 |

FD002 的嵌套 CV 与选择 CV 数字完全相同，**本身不是泄漏证据**：保存的五个外折恰好都选中相同配置，也与全量选择的配置一致，最终外层评估使用相同划分，因此这个相同值有合理解释。

**未独立复现的部分：**由于本轮没有重新训练、生成逐样本预测，官方测试分数和预警汇总的数值尚未从原始数据端到端重算。上述“核对通过”指调用链、配置、文件间关系和已保存数字的算术一致性。

关于结果能支持多强的结论，我认为可以表述为：

> 在 NASA C-MAPSS 四个子集上，采用按发动机分组、折内预处理，以及覆盖特征、超参数和损失权重选择的嵌套交叉验证；在当前近故障评价目标和固定划分下，完整选择流程的外层 RMSE 均优于保存的默认配置。

这仍需保留几个具体边界：

- **嵌套 CV 估计的是选择流程，不是已经固定的最终模型本身。** 当前外层内部选择使用三折、seed 43；全量最终选择使用五折、seed 42。没有因此形成外层泄漏，但不宜称为逐项完全相同的选择协议。
- **“5/5 折更好”是这组划分的描述性证据，不是已经完成统计显著性检验。**
- **不对称损失的取舍并非所有指标都改善。** 相对当前保存的普通对称模型，FD002 官方总分从 **4845.03 升至 6756.10，恶化 39.44%**；FD004 从 **4015.27 升至 5090.90，恶化 26.79%**。近故障目标改善与全寿命误差代价同时存在。
- **预警表增加了有用证据，但还没有证明总体维修成本更低。** 例如 FD004 阈值 40，在同组、但受选择偏差影响的 CV 对照中，首次报警时 `RUL>100` 的发动机从 **2 台增至 4 台**，健康周期报警率从 **0.22% 增至 0.25%**；见 [tuning.json:143](/D:/projects/engine-rul-prediction/results/FD004/tuning.json:143) 与 [tuning.json:191](/D:/projects/engine-rul-prediction/results/FD004/tuning.json:191)。当前规则没有把这些代价纳入采用条件，也没有实际维修成本模型。
- **官方测试集应继续定位为开发过程中看过的基准。** 本轮对嵌套边界的修复是实质改进，但不能消除过去测试反馈对指标、截顶方案和研究方向的影响。
