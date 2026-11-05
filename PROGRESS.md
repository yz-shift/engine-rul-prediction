# PROGRESS — engine-rul-prediction

> 进度日志。新的写在最上面；只追加，不删改旧记录。

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
