# Codex 评审第三轮

> 评审对象是提交 f8f5e78。Codex 的结论：第二轮 4 条修复都已落实，没有发现新问题；它也确认了对它上一轮说法的更正是对的。
> Codex 没能在沙箱里运行 pytest，也没有重新训练。测试和重跑由 Claude 这边完成，记录在 PROGRESS.md。

本轮没有需要处理的意见，迭代到此结束。

---

## Codex 原文

**没有发现新的、已经核实的问题。** 对最新提交 `f8f5e78` 的复审表明，第二轮四项修复已落实，未发现修复引入的新问题。

四项修复的核实结果：

| 修复 | 核实结果 |
|---|---|
| 回退模型与记录成绩不一致 | 回退时重新计算普通 LightGBM 自己的 CV；采用不对称损失时取对应权重的成绩。脚本统一写入 `selection.selected_cv`，模型与记录已对应。见 [tuning.py:175](/D:/projects/engine-rul-prediction/src/rul/tuning.py:175)、[tune_models.py:87](/D:/projects/engine-rul-prediction/scripts/tune_models.py:87)。 |
| 搜索记录中的参数被舍入 | 候选参数已移入受保护的 `model_params`，递归序列化会保留参数精度。未发现仓库运行时代码依赖旧的扁平结构。见 [tuning.py:50](/D:/projects/engine-rul-prediction/src/rul/tuning.py:50)、[results.py:19](/D:/projects/engine-rul-prediction/src/rul/results.py:19)。 |
| 两处回归测试缺口 | 新断言能拦住“外层最终拟合误用全量数据”；窗口和模型参数现在分别改变、分别比较，解决了原先同时改变多项导致的断言混杂。见 [test_tuning.py:130](/D:/projects/engine-rul-prediction/tests/test_tuning.py:130)、[test_alerts_and_results.py:120](/D:/projects/engine-rul-prediction/tests/test_alerts_and_results.py:120)。 |
| CV 与测试指标可比性解释过强 | 注释已明确相同 RUL 范围不代表同分布，并区分近故障区间每 100 点分数与全部测试发动机总分；与实际计算一致。见 [evaluate.py:44](/D:/projects/engine-rul-prediction/src/rul/evaluate.py:44)、[evaluate.py:127](/D:/projects/engine-rul-prediction/src/rul/evaluate.py:127)。 |

**第二轮文档对上一轮说法的更正正确。** [FD004/tuning.json:59](/D:/projects/engine-rul-prediction/results/FD004/tuning.json:59) 的第三个外折确实是 `model="lightgbm"`、`late_weight=1.0`；五折权重为 **`[5, 5, 1, 5, 5]`**。上一轮“所有外折均选择权重 5”的说法错误，应撤回。四个子集的全量最终选择仍都是权重 5。

“该记录错误不影响已有嵌套 CV 数字”的更正也成立：修复前后的外层流程都实际拟合返回的模型，再对留出发动机预测并计算指标，没有拿权重表中的 CV 成绩代替外层评估。见 [tuning.py:212](/D:/projects/engine-rul-prediction/src/rul/tuning.py:212)。

本轮全程只读，没有修改文件或重新训练。尝试以禁用字节码及 pytest 缓存的方式运行相关测试，但虚拟环境 Python 在启动阶段报 `Unable to create process`，所以以上结论来自代码、断言逻辑和已有结果核对，**不代表我独立确认了“116 个测试全过”**。
