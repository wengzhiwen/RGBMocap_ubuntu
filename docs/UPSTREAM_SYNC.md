# 上游同步经验与操作流程

记录日期：2026-11-11。本文记录 2026-11-10 至 11 日从
`toyxyz/toyxyz_RGBMocap` 同步新特性的实际经验，并为后续同步提供操作依据。
安装、运行时与缓存规则仍以 [Ubuntu 指南](UBUNTU.md#upstream-sync) 为准。

## 1. 同步原则

- 保留源项目的功能、GUI 工作流和原有入口，尽量保持上游代码与行为；Ubuntu、ARM64 和 CLI 能力作为扩展。
- CLI 增强不等于产品 CLI 化，不得据此将 GUI 启动器重定向到 CLI。
- 使用独立同步分支执行 merge，保留双方历史，便于审查平台适配和缓存身份；不要用整体覆盖或全选 ours/theirs 解决冲突。
- 分开报告“代码已合并”“已推送”“已进入 main”“已部署”“已在目标平台验证”。
- 不通过删除模型、项目、缓存、运行时或修改已有推理结果解决合并问题。

## 2. 本次同步范围及状态

| 项目 | 本次记录 |
| --- | --- |
| 上游 | `https://github.com/toyxyz/toyxyz_RGBMocap.git` |
| 原基线 | `ef05127a0dbddb3f3ac672a0722dddeae42ea651` |
| 本次同步目标 | `1b0384367fc6b63932c82f1c9c5cb4243a91de2e`，抓取时的 `upstream/main` |
| 同步分支 | `codex/sync-2026-11-10` |
| 合并提交 | `9bad935` |
| 范围 | 9 个上游提交 |

截至本文记录时，上游新特性已进入同步分支并推送，尚未合入 `main`。
启动器修正已分别提交到同步分支和 `main`，不代表上游特性已进入 `main`。
后续状态以 Git 分支和验证记录为准，不将这张历史表视为持续更新的部署清单。

| 上游变化 | 同步时确认的边界 |
| --- | --- |
| Blender 骨骼串流、烘焙和可安装插件 | 代码已同步；本次未做 Linux 串流实测 |
| Maya 兼容 FBX 导出 | 代码已同步；本次未做 Linux 导出或 Maya 导入验收 |
| 原点放置、对称休息姿态导出 | 代码已同步；本次未做实际产物验收 |
| GEM-X Native77 | 上游运行时显式拒绝非 Windows 平台；保留该边界 |
| SAM 3.1 人物跟踪 | 依赖上游 Windows 安装流程及 ExMR Python 路径；Linux 尚不可用 |
| FlowHMR 下载进度回调修复 | 保留上游修改及其缓存别名 |
| 安装恢复、验证复用、更新启动器修复 | 保留上游 Windows 安装与更新逻辑；本次未做 Windows 安装实测 |

## 3. 合并前先明确差异

先阅读 `AGENTS.md`、`docs/AGENT_GUIDE.md` 和 `docs/UBUNTU.md`。
检查工作区；已有用户修改需要单独保存和审查，不能混入同步冲突处理中。

```bash
git status --short
git remote -v
# 仅在 upstream 不存在时添加：
git remote add upstream https://github.com/toyxyz/toyxyz_RGBMocap.git
git fetch upstream

# 从上次同步记录填写基线，并固定本次目标，避免同步中途引用漂移。
sync_base=1b0384367fc6b63932c82f1c9c5cb4243a91de2e
sync_target=$(git rev-parse upstream/main)
git log --oneline "$sync_base..$sync_target"
git diff --stat "$sync_base..$sync_target"
git diff --name-only "$sync_base..$sync_target"
git diff --name-only "$sync_base..HEAD"

git switch -c codex/sync-YYYY-MM-DD
git merge "$sync_target"
```

创建分支前确认当前分支确实是预期的本项目基线；已有同步分支不要重复创建。
重点对照双方都修改的文件，以及运行时、receipt 哈希来源、依赖锁、平台启动器。
本次双方修改交集有 9 个文件，实际冲突为 `AGENTS.md`、`README.md`、
`mocap/resource_cache.py`。自动合并成功仍须审查行为。

## 4. 本次冲突如何解决

### 缓存别名是累加冲突

`resource_cache.py` 中，本分支增加了 `PLATFORM_ONLY`，上游增加了
`DOWNLOAD_PROGRESS_ONLY`。两者用途不同，应同时保留，在 `code_hash` 中
分别查询。当前查询顺序为平台、下载进度、日志、策略别名，最终回退到实际哈希；
这不是将任意旧哈希递归映射到新代码。

本次 `worker.py`、`exmr_backend.py`、`flow_hamer.py` 在上游没有变化，
原平台别名仍有效；上游下载进度别名对应的源码也没有被本分支再次改写。
不能因为测试失败就将新哈希映射回旧身份：数值算法、推理输入或设置发生改变时，
必须让缓存正常失效。

### 文档保留本分支定位，补充上游能力

README 保留 Ubuntu 快速开始与 CLI 模板介绍，补充新特性及 Linux 状态。
AGENTS 合并双方约束，既保留 Linux/ARM64 指引，也保留 GEM-X 的上游限制。
不要将上游 Windows 安装章节直接插入 Ubuntu 命令代码块。

### 自动合并的文件也要检查

`video.py` 同时保留了 Linux 可执行文件查找、进程标志兼容处理，以及上游新增的
可选 `include_seek_offset` 参数。平台适配不能丢，上游增量行为也不能丢。

对于 GUI 顶层导入的新模块，区分“导入时就失败”和“用户启动功能时报告不可用”。
本次 GEM-X 的非 Windows 限制、SAM 3.1 的运行时检查保留上游实现。
发现 `.exe` 或 `CREATE_NO_WINDOW` 时要追踪调用路径，不能仅按字符串批量替换。

## 5. 字节、依赖与测试身份

- 仓库 `.gitattributes` 使用 `* -text`，文件字节参与哈希。保持每个文件原有 CRLF/LF，不运行全仓库换行规范化。本次 `resource_cache.py` 保持全 CRLF。
- `runtime_registry.py` 和 `installer/runtime-lock.json` 本次未发生上游变化，无需为同步强行重写锁。今后若修改 registry，按 Ubuntu 指南重算 `registry_sha256` 和锁的 `id`。
- 保留独立模型运行时和固定版本；ARM64 变体不要改写 Windows/amd64 的 `PROFILES`。
- 本次更新了测试 `BASE_SHA` 和同步指南基线；其他摘要只有对应来源实际改变并审查后才更新。
- `test_cli_exports` 原来检查 FBX worker 内联的临时文件命名公式；上游改用 `blend_support.partial()`。测试应继续检查同样的暂存命名约定，跟进真实重构，而不是退回上游实现或直接删除断言。
- `SOURCE_MANIFEST.json` 随上游更新，不为消除 fork 差异自行重生成上游发布快照。

## 6. 启动器大小写冲突：本次最重要的教训

原来同时跟踪 `RGBMoCap.sh`（GUI）和 `rgbmocap.sh`（CLI）。在当前 Mac 的
大小写不敏感文件系统上，两条路径实际指向同一文件：修改 CLI 会表现成 GUI
文件也被覆盖。最初误判为用户希望 CLI 化，导致提交了错误的 GUI 重定向。
用户明确要求保留 GUI，只增强 CLI，随后已修正。

最终保留以下入口：

| 入口 | 用途 |
| --- | --- |
| `RGBMoCap.sh` | 原有 Linux GUI 入口，保留原启动行为 |
| `rgbmocap-gui.sh` | 额外的明确命名 GUI 入口 |
| `rgbmocap-cli.sh` | 增强 CLI，替代旧的 `rgbmocap.sh` 名称 |
| `RGBMoCap.cmd` | 上游 Windows GUI 入口，保持上游行为 |

后续不要重新引入仅大小写不同的两条跟踪路径。修改入口时同时检查 README、
安装提示、CLI 的 `argparse.prog`、操作指南、测试和本地项目技能引用。
历史验证记录中的旧命令保留为当时的事实，不批量改写成仿佛当时使用了新命令。

本次还出现过 `git add` 返回成功、暂存区却不符合预期，以及切换旧分支后文件
暂时缺失的现象。已确认大小写路径冲突存在，不能据此断言 Homebrew Git 本身有 bug。
排查时对照 `git diff`、`git diff --cached`、`git ls-files --stage` 和文件 inode；
提交后用 `git show HEAD:<path>` 核对实际提交内容。`git update-index` 只能辅助
处理索引，不能从根本上解决两个文件名在文件系统上冲突的问题。

## 7. 验证要区分回归与已有失败

本次在同一 Mac 上通过独立临时 worktree 对比合并前 `main` 和同步分支：

- 最初全套 188 项中，同步分支比合并前多一个临时文件命名测试失败；跟进上游辅助函数重构后，两边失败集合一致。
- 当时仍有 5 个失败、7 个错误、20 个跳过，包含非 Linux 环境、依赖/进程测试条件和 GUI 入口问题；“集合一致”只说明未新增该套测试可见的回归，不代表全套通过。
- `tools/check_source.py` 当时扫描 2661 个文件，报告零错误；69 个变更 Python 文件的语法检查通过。
- 完成启动器修正后，`Launchers` 的 4 项检查在 `main` 和同步分支均通过；CLI 改名阶段另运行启动器与 CLI 导出测试共 47 项，6 项跳过，其余通过。
- 后续入口修正后没有重新完整运行 188 项，不能沿用早先失败数描述修正后的全套结果。
- 本次未运行 Linux GPU doctor、真实 FlowHMR + HaMeR 视频、缓存复用验收，也未验证实际 GUI 播放或新导出功能。

后续同步使用以下入口执行检查，并记录实际平台和结果：

```bash
python3 -m unittest discover -s tools/tests -v
python3 tools/check_source.py
./rgbmocap-cli.sh doctor --gpu
```

在目标 Ubuntu 主机上使用授权的测试视频和明确的项目目录，完成真实
FlowHMR + HaMeR 处理，再用相同参数重复运行确认阶段缓存复用。
若改动涉及导出、播放或串流，还须做相应实际功能验收。
按 [同步指南](UBUNTU.md#upstream-sync)，完成目标平台验证后才将同步分支合入 main。

## 8. 提交与交付检查

提交前检查冲突标记、`git diff --check`、暂存差异和新增文件；不要把
third_party 许可证中的分隔线误认为合并冲突。提交后检查工作区和提交树，
本次曾发现三处已编辑的文档/测试尚未进入最初的合并提交，随后已补入。

合并提交保留上游来源、目标 SHA、冲突决策、缓存身份处理和验证限制。
入口兼容性修复可以作为独立提交同步到 main，但这与将整个上游同步分支
合入 main 是两件事。不要为了推送而覆盖远程已有的新提交；本次新分支的
合并提交曾补充遗漏文件并改写，后续优先用追加提交，保留可审查记录。
