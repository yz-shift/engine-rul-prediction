# PROGRESS — engine-rul-prediction

> 进度日志。新的写在最上面；只追加，不删改旧记录。

## 2026-11-05 23:51 · T-010 · 完成
- 做了：新增 tuning.py（随机搜索 20 组、嵌套 CV：外层 5 折、内层 3 折）和 scripts/tune_models.py；搜索窗口长度和 7 个 LightGBM 参数，截顶固定 125；另外报告不同截顶的 CV 取舍。嵌套 CV 相比默认：FD001 14.04 到 11.94，FD002 16.46 到 13.90，FD003 14.16 到 12.75，FD004 16.55 到 14.62，都是 5/5 折更好。官方测试 RMSE：FD001 14.57 到 13.20，FD002 26.51 到 24.42，FD003 15.39 到 14.62，FD004 26.60 到 25.43
- 改了：src/rul/tuning.py、scripts/tune_models.py、tests/test_tuning.py、results/*/tuning.json、TASKS.md
- 验证：pytest 全过，有测试确认外层留出的发动机不参与选参数；结果写入 results/*/tuning.json
- 问题：第一版把截顶也放进搜索，所有子集都选中截顶 100，CV 大幅变好（FD001 11.69），官方测试 RMSE 却变差（FD001 16.98、FD002 31.55）。原因是选参指标只统计 RUL <= 125，看不到 125 以上的误差，一定偏向低截顶。这是看到测试结果后才发现的，已如实记录；修正是截顶固定 125（方案已确认的值），不再由这个指标决定。局限：几乎所有折都选中窗口 50，这是搜索范围的上限，更长窗口可能更好，没有测试
- 下一步：T-011 不对称损失

## 2026-11-05 22:55 · T-009 · 完成
- 做了：重构特征流程，改为在每一折内部只用训练发动机拟合（Featurizer）；新增 ConditionNormalizer（KMeans 聚类工况后按工况 z 分数归一化）；实验扩展到四个子集，结果移到 results/<子集>/；界面支持切换子集和 ?subset= 链接；测试集结果增加和 CV 同口径的 rmse_near_failure。最优模型（默认参数）CV 和测试 RMSE：FD001 14.04、14.57；FD002 16.46、26.51；FD003 14.16、15.39；FD004 16.55、26.60
- 改了：src/rul/features.py、experiments.py、evaluate.py、data.py、models.py、results.py、scripts/train_models.py、app/streamlit_app.py、tests/、results/、TASKS.md
- 验证：重构前后 FD001 的结果文件 diff 为 0 行（回归测试）；pytest 全过；有测试确认特征器只在训练折上拟合、归一化只学训练数据
- 问题：FD002、FD004 的 CV（16.5）和测试 RMSE（26.5）差很多，排查后不是泛化问题，是口径不同：同样只算 RUL <= 125 时测试 RMSE 是 13.35、14.45。这两个子集 22–27% 的测试发动机 RUL 大于 125（最高 195），截顶 125 的模型平均低估 40 多个周期。这是截顶在多工况子集上的真实代价。notebook 02 还用旧接口，在 T-013 一起更新
- 下一步：T-010 超参数搜索

## 2026-11-05 21:18 · T-008 · 完成
- 做了：写 README（简介、截图、结果表、提升来源、数据发现、预警线、技术栈、运行方法、设计取舍、踩过的坑、局限、数据出处和许可证）；修正 download_data.py 里无法核实的来源注释
- 改了：README.md、scripts/download_data.py、TASKS.md、PROGRESS.md
- 验证：README 的结果数字逐项对照 results/summary.csv 和 notebook 输出；用当前代码的干净副本和全新虚拟环境按 README 跑通：安装、下载数据（13 个文件）、pytest 73 个全过、重跑全部实验后 results/ 差异 0 行、Streamlit 启动健康检查返回 200
- 问题：未验证：传感器名称表和评分函数没有对照 Saxena 2008 原文，NASA 的页面和 PDF 用抓取和内置浏览器都打不开，只核对了转引论文，README 已写明。数据集许可证未能确认：data.gov 记录没有明确条款，所以仓库不分发数据，README 已写明。之前说下载地址是“NASA 页面链接的”也没有核实，已改掉注释。代码本身的许可证还没选，README 也还是中文，等我决定。第一次干净副本验证因 tar 把 C: 当成远程主机而没有实际执行，修好后重跑才通过
- 下一步：全部开发任务完成。git push 需要我同意；开新会话做 T-999 面试演练

## 2026-11-05 21:14 · T-007 · 完成
- 做了：新增 Streamlit 界面（选测试发动机、预警线滑块、传感器多选、预测和真实 RUL 曲线、维修提示、指标），默认阈值 40 来自训练集交叉验证外预测；预测逻辑放在 src/rul/inference.py；支持 ?engine=34 链接；AGENTS.md 第 2 节填入运行命令；安装 streamlit
- 改了：app/streamlit_app.py、src/rul/inference.py、tests/test_inference.py、docs/app_alert.png、docs/app_healthy.png、AGENTS.md、.gitignore、pyproject.toml、requirements-lock.txt、TASKS.md
- 验证：一条命令启动；内置浏览器实测：1 号发动机显示运行正常（预测 113、真实 112），选 34 号转为维修提示（预测 8、真实 7，第 152 周期首次触发），滑块调到 10 后首次触发变为第 203 周期，?engine=34 能直接打开，控制台无错误；AppTest 自动化界面测试和 pytest 全过（73 个）；无头 Edge 生成截图
- 问题：预警有时会反复：34 号发动机第 152 周期触发后，第 180 周期前后预测短暂回到预警线之上。界面用“需要留意”状态提示这种情况，没有做平滑。此外更正 T-006 日志时间（写成 21:12，实际 21:07）
- 下一步：T-008 README

## 2026-11-05 21:07 · T-006 · 完成
- 做了：新增 features.py（窗口 30 的均值、标准差、斜率，只看过去、不跨发动机）；models.py 加随机森林和 LightGBM（参数事先定好，没有搜索）；跑 14 组实验（含 Nc/NRc 和全部列两组特征对照）；写 notebooks/02_modeling.ipynb（对比表、配对比较、种子波动、测试集误差、预警提前量、特征重要性）。按 CV 选出 LightGBM + 窗口 + 截顶 125：CV 14.04 ± 0.82，测试 RMSE 14.57、分数 308.7；基线是 21.90、1314
- 改了：src/rul/features.py、src/rul/models.py、src/rul/experiments.py、scripts/train_models.py、tests/test_features.py、tests/test_experiments.py、notebooks/02_modeling.ipynb、results/、pyproject.toml、requirements-lock.txt、TASKS.md
- 验证：pytest 全部通过（含真实数据上窗口特征恰好 56 列）；实验连续跑两遍，results/ 差异 0 行；notebook 从头跑通；图表和结论数字已对照输出
- 问题：(1) 恒定传感器的斜率因浮点误差变成 1e-14 级噪声，骗过了方差筛选，已通过先减首个读数修复并加测试。修复后 LightGBM 的 CV 从 13.83 变为 14.04，也就是说修复没有提升分数，反而略差，但这些列确实是噪声，所以保留修复。(2) 随机森林多线程预测有 1e-13 级抖动，结果文件保留 4 位小数后逐字节一致。第一次加四舍五入的脚本修改没有生效（后台运行没看到报错），是用 diff 发现后才补上的。(3) 随机森林测试集略好于 LightGBM（14.44 对 14.57），按规则仍按 CV 选 LightGBM，两者实际打平
- 下一步：T-007 Streamlit

## 2026-11-05 20:48 · T-005 · 完成
- 做了：新增 models.py（Pipeline：方差筛选 + 标准化 + 线性回归）、evaluate.py（交叉验证、官方测试评估）、experiments.py（实验配置，训练脚本、notebook、界面共用）、scripts/train_models.py（指标写入 results/）。两种标签的基线结果：截顶 125 为 CV 临近故障 RMSE 21.18 ± 1.73、官方测试 RMSE 21.90、分数 1314；不截顶为 CV 33.98 ± 4.76、官方测试 RMSE 31.90、分数 13221
- 改了：src/rul/models.py、src/rul/evaluate.py、src/rul/experiments.py、scripts/train_models.py、tests/test_evaluate.py、results/、TASKS.md
- 验证：pytest 36 个全过；CV 指标和官方测试结果写入 results/*.json 与 summary.csv；CV 与官方测试 RMSE 接近（21.18 对 21.90）
- 问题：第一版 CV 按测试集轨迹长度随机截断验证发动机，得到 CV RMSE 39、分数约 28 万，和测试集差很多。用训练发动机查明：模拟 RUL 最高到 313，25% 大于 125，单点罚分可达 70 万，分数被极端点主导。改为只统计真实 RUL <= 125 的周期。这次改动是看到测试结果后做的，但只改了验证指标，没有改模型或参数。另外，全周期 RMSE 上不截顶反而更好（44.11 对 51.76），因为截顶模型不会预测超过 125，这是“临近故障”指标和全周期指标的取舍，需要在 README 里说明
- 下一步：T-006 随机森林 + LightGBM + 滑动窗口特征

## 2026-11-05 20:43 · T-004 · 完成
- 做了：data.py 增加 add_rul（支持截顶）和 engine_folds（按发动机的 GroupKFold，种子固定）；新增 metrics.py（rmse、官方 nasa_score）；01_eda 改用共用的 add_rul，去掉重复逻辑；安装 scikit-learn（scipy 随之装上）
- 改了：src/rul/data.py、src/rul/metrics.py、tests/test_metrics.py、tests/test_labels_and_folds.py、notebooks/01_eda.ipynb、pyproject.toml、requirements-lock.txt、TASKS.md
- 验证：pytest 29 个全过，覆盖切分无交集、每台发动机只验证一次、种子可复现、RUL 倒数到 0、截顶、评分函数手算样例（2(e-1)、e^2-1、e^(20/13)-1）；01_eda 重新从头跑通
- 问题：无
- 下一步：T-005 线性回归基线

## 2026-11-05 20:39 · T-003 · 完成
- 做了：写 notebooks/01_eda.ipynb（传感器含义表、恒定特征、每台发动机内部的相关、趋势小图、Nc/NRc 方向分析、退化指标和截顶依据、结论表）；经我同意装 matplotlib、jupyterlab；填好 AGENTS.md 第 1、2 节（已在 a088e21 提交）
- 改了：notebooks/01_eda.ipynb、pyproject.toml、requirements-lock.txt、TASKS.md
- 验证：jupyter nbconvert --execute 从头跑通，无报错、无未执行单元格；结论里的数字对照了 notebook 输出；5 张图逐张看过；pytest 13 个全过。只用训练集，测试集只看轨迹长度，没读 RUL 答案
- 问题：传感器名称表核对的是两篇转引论文（arXiv 2605.02507、2503.07854），未读到 Saxena 2008 原文，写 README 前要再查。pandas 的 Spearman 依赖 scipy，改用先排名再算 Pearson，没有新增依赖
- 下一步：T-004 标签、按发动机切分、评分函数（需要 scikit-learn，装之前先问我）

## 2026-11-05 20:28 · T-002 · 完成
- 做了：搭项目骨架（src 布局、pyproject、.venv）；写读取模块（load_trajectories、load_rul，读入时校验列数和缺失值）；写下载脚本（白名单提取、支持嵌套 zip、防路径穿越）；经我同意安装 pandas、pytest 并下载数据到 data/raw/
- 改了：.gitignore、pyproject.toml、requirements-lock.txt、src/rul/__init__.py、src/rul/data.py、scripts/download_data.py、tests/test_data.py、tests/test_download.py、TASKS.md
- 验证：pytest 13 个全部通过，其中 2 个读真实数据。FD001 训练集 (20631, 26)、100 台，测试集 (13096, 26)、100 台，RUL 100 行。列数和台数与官方 readme 一致；readme 没写行数，行数用 wc -l 核对原始文件。git check-ignore 确认 data/ 和 .venv 不进 git
- 问题：官方 readme 的 FD004 台数疑似写反（已记入待定，不影响 FD001）；AGENTS.md 第 1、2 节的入口和命令还没填，改它要先问我
- 下一步：T-003，需要安装 matplotlib、jupyter 等依赖，装之前先问我

## 2026-11-02 · T-001 · 完成
- 做了：我确认方案全部按推荐执行：同意从 NASA 下载 C-MAPSS（约 12 MB，不进 git）；梯度提升用 LightGBM；不加 LSTM；RUL 截顶 125 并和不截顶对比。排在项目 3 之后。已拆成具体任务
- 改了：TASKS.md
- 验证：无代码
- 问题：无
- 下一步：T-002

## 2026-10-31 · T-001 · 进行中
- 做了：写好 PLAN.md（架构、技术选型、目录结构、分步计划、待确认问题）
- 改了：PLAN.md（新增）、TASKS.md、AGENTS.md（填了项目说明和特有规则）
- 验证：无代码。本机检查：没有 docker、gcc、make、WSL
- 问题：见 PLAN.md 第 6 节
- 下一步：等我回复 PLAN.md 第 6 节的 4 个问题

## 2026-10-31 00:43 · 初始化 · 完成
- 做了：用模板创建项目，生成 AGENTS.md / TASKS.md / PROGRESS.md
- 改了：无代码
- 验证：无
- 问题：无
- 下一步：填 AGENTS.md 第 1、2 节，然后做 T-001
