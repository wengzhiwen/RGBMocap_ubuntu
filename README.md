# RGB MoCap · Ubuntu

[toyxyz/RGB MoCap](https://github.com/toyxyz/toyxyz_RGBMocap) 的 Ubuntu 移植与增强分支。
原项目是仅面向 Windows 的原生应用（CMD 安装器、`runtime_*` Python 发行版、Windows 进程与内存接口）；
本分支在不修改上游数值与工作流的前提下，把它移植到 Ubuntu（x86-64 与 ARM64 / NVIDIA DGX Spark），
并在多 GPU 批量处理、命令行流水线和安装分发上做了自己的增强。上游的全部模型、GUI、导出与串流代码保持合并同步
（当前基线 `76612b52`，2026-11-20），Windows 行为不受本分支改动影响。

通用模型与 GUI 操作见[模型指南](docs/models/README.md)和[用户指南](docs/USER_GUIDE.md)；本页只讲本分支的移植与优化。

## 相对原项目的移植工作

### 独立运行时体系（替代 Windows CMD 安装流）

- `tools/setup_ubuntu.py` 用哈希固定的 wheel 安装替代 `install.cmd`：共享/GUI（`.venv`）、
  FlowHMR（`.venv-flow`）、ExMR（`.venv-exmr`）、HandFlow（`.venv-hand`）、HTD（`.venv-htd`）
  各自独立、版本互不干扰；`--plan` 支持干跑检查，可分步重试失败的阶段。
- `mocap/runtime_registry.py` 增加平台路由：Linux 下各阶段进入自己的 `.venv*` 解释器，
  Windows 的 `PROFILES`、receipt 内容与 amd64 字节身份保持不变。
- 进程与内存接口移植：峰值内存用 `ru_maxrss` 替代 `peak_wset`，进程取消经 psutil
  递归终止子进程，Windows 创建标志在 Linux 回落为 0。
- 原生动作修正模块（KimodoMotionCorrection）在安装时于本机编译，ARM64 使用 NEON 适配；
  amd64 与 aarch64 的修正结果经过逐值对比验证一致。

### ARM64 / NVIDIA DGX Spark（GB10）

- `installer/linux-aarch64.json` 为 ARM64 单独固定 Torch / torchvision（CUDA 13 轮子）与 Qt 版本，
  安装前用 uv 实际校验每个 pin 的轮子可获取性，不改写 Windows/amd64 配置。
- 修复 ARM64 上 Python 3.10 的 Qt 绘图引用损坏导致的 GUI 崩溃，GB10 通过 60 秒原生网格播放复检。
- 提供 **arm64 Ubuntu 24.04 在线安装 deb 包**：`sudo apt install ./rgbmocap_<version>_arm64.deb`
  后从应用菜单或终端 `rgbmocap` 启动，首启由用户身份完成 uv 与环境安装（不以 root 下载、不装后台服务），
  支持 `rgbmocap --setup` / `--doctor` / `--doctor --gpu`。见 [deb 安装说明](docs/DEB.md)。

### 诊断、修复与启动加固

- `./rgbmocap-cli.sh doctor --gpu`：除依赖与文件检查外，在共享/Flow/ExMR 每个环境里运行真实
  CUDA matmul 与 NMS 运算，并检查 Qt 绘图绑定、FFmpeg、Blender 与原生修正模块。
- Linux GUI 启动检查加固，可按缺失项做定向依赖修复，而不是整包重装。

### 兼容性策略：黑名单 + 上游同步 + 缓存身份

- **Linux 黑名单**：确定无法在 Linux 运行的模型与功能在界面隐藏——模型为 PEAR、MAMMA、
  PromptHMR、GEM-X 与 SAM 3.1 跟踪（MSVC/CUDA 原生构建或 Windows 共享内存依赖）；
  功能为 Track Face 人脸捕获（`runtime_exmr/python.exe` + Windows wheel 锁）与
  Unreal Live Link 串流桥（Win64 DLL）。引用这些模型的既有项目文件仍可打开，
  已有的人脸数据查看、中性脸校准与 TXT / Live Link Face CSV 导出保持跨平台。
- **可审查的上游合并流**：以 merge（非 rebase）定期并入上游 `main`，冲突决策、缓存身份与验证
  限制记录在[同步指南](docs/UPSTREAM_SYNC.md)。
- **缓存身份保持**：receipt 哈希涉及的文件在本分支改动时，经 `PLATFORM_ONLY` 别名仍指向基线身份，
  因此同步上游后已保存的阶段缓存继续命中——2026-11-20 合并后用 451 帧真实项目重跑验证，
  HaMeR 阶段报告 "Verified hand cache reused"，全流程约 14 秒完成。
- 项目可在 amd64 与 aarch64 之间迁移：receipt 记录架构变体，跨架构自动重算对应阶段而不是复用错误结果。

## 相对原项目的增强与优化

### FlowHMR 命令行流水线（headless）

- `rgbmocap-cli.sh flow`：视频准备 → 自动单人跟踪 → 相机估计 → 动作重建 → 可选 HaMeR 手指增强 →
  接触修正 → 预览与 FBX/BVH 导出，一条命令完成；支持 `--dry-run`、`--json`、单阶段调用、
  裁剪范围、资源模式与修正参数。
- `rgbmocap-cli.sh export`：对已准备好的结果导出 FBX / BVH / 便携 Blender 项目，
  支持 Biped BVH、前置 T-pose、保存的相机动画等选项。
- CLI 与 GUI 调用相同 worker、使用相同项目文件与阶段缓存：命令行生成的项目可直接在 GUI 中打开，
  资源保护、停滞检测与进程组取消行为一致。
- **项目配置模板**：把调好参数的 `project.json` 作为新视频的处理模板
  （`--template`），模板决定 FlowHMR 与修正参数，命令行仅控制新视频、输出目录与裁剪。

### 多 GPU 并行与批量处理（GUI）

- **批量视频处理**：一次选择多个视频，自动完成 Find People（每视频自动勾选前 N 人）、
  重建与导出，产出逐视频的导出文件与进度/错误报告；首个视频提供共享的模型设置与裁剪。
- **多 GPU 并行**：勾选两块以上 GPU 后，父进程为每块 GPU 启动一个无头子进程，
  一卡一视频同时推理；并行批次同样支持预览视频导出（并行重建完成后由主窗口顺序渲染）。
- 批量导出对话框沿用上游导出选项（含 2026-11-20 新增的 take 范围、A/T 参考姿态、UE5 Mannequin 等），
  Linux 上隐藏不可用项。

### 性能测量与实验性并行

- [PRO 4000 / GB10 性能测量记录](docs/BENCHMARK_RESULTS_UBUNTU.md)：同一项目在两类硬件上的
  阶段耗时、显存与内存实测，作为资源模式选择的依据。
- 实验性张量并行 / 流水线并行适配器（TP2/PP2）与 CUDA peer 数据完整性诊断已随仓库发布，
  但**未接入生产推理**，仅用于受控基准实验。

## Ubuntu 快速开始

需要 64 位 Ubuntu（x86-64 或 aarch64）、兼容的 NVIDIA GPU / 驱动，以及可用的 `uv`。
GUI 还需要 OpenGL 4.3。系统依赖与 `uv` 查找方式见[完整安装指南](docs/UBUNTU.md#prerequisites)。

```bash
git clone https://github.com/wengzhiwen/RGBMocap_ubuntu.git
cd RGBMocap_ubuntu

sudo apt install ffmpeg libglu1-mesa libegl1 libgl1 libopengl0 libxcb-cursor0 cmake build-essential git

python3 tools/setup_ubuntu.py --plan
python3 tools/setup_ubuntu.py --stage-assets --build-correction
./rgbmocap-cli.sh doctor --gpu
./RGBMoCap.sh
```

ARM64 也可直接安装 [deb 在线安装包](releases/ubuntu-arm64/)（含校验和，说明见 [docs/DEB.md](docs/DEB.md)）。
模型权重与人体数据需另行通过 **Model Manager**
下载或导入，安装脚本不包含这些文件；部分数据需要注册并接受提供方许可。

## 命令行处理与导出

```bash
# 预览执行计划
./rgbmocap-cli.sh flow --video clip.mp4 --hamer --correct --dry-run

# 单人重建、手指增强与动作修正
./rgbmocap-cli.sh flow --video clip.mp4 --hamer --correct

# 用已完成的模型项目导出
./rgbmocap-cli.sh export --project /path/to/model_project --kind corrected_initial --output motion.fbx
./rgbmocap-cli.sh export --project /path/to/model_project --kind corrected_initial --output motion.bvh

# 用旧项目的 project.json 作为新视频的处理模板
./rgbmocap-cli.sh flow --video new_clip.mp4 --template /path/to/old/project.json --dry-run
```

`flow` 当前自动跟踪一人；多人选择仍使用 GUI。`export` 导出已准备好的结果。
Blender 打包需要 Blender 4.5，源动作 FBX / BVH 导出不需要 Blender。
模板规则见[模板使用说明](docs/UBUNTU.md#reuse-a-projects-processing-settings)；
模板功能已通过 CPU 配置与路由测试，尚未单独完成真实视频 GPU 端到端验证。

## 上游同步新增（2026-11-20，自 1b03843a）

- **UE5 Mannequin FBX 导出**：可将动作重定向到 UE5 Manny / Quinn 官方骨架并导出 FBX，
  可选附带官方蒙皮网格；含绑定姿态与脚部间距保持（见[用户指南](docs/USER_GUIDE.md#ue5-mannequin-fbx)）。
  代码为跨平台 Python，本分支 Linux 端尚未实测。
- **Take 选择导出**：导出对话框可选择当前 take、全部 take 或单独 take，
  并说明每个不可用 take 需要的操作；导出诊断同步改进。
- **导出与预览增强**：源 FBX / BVH 导出新增 A / T 参考姿态；预览视频导出（Export Preview）
  可包含源视频音频；Blender 串流端口可修改；Blender 打包合并导出。
- **空脚部接触修复**：角色重定向在脚部接触数据为空时不再失败（含回归测试）。
- **Track Face 人脸跟踪与导出**：MediaPipe CPU 人脸跟踪、表情 One Euro 平滑、
  中性脸校准和 ARKit 系数 / Unreal Live Link Face CSV 导出。
  上游运行时固定使用 `runtime_exmr/python.exe` 与 Windows x64 wheel 锁
  （`installer/face-windows.lock`），Linux 端不可用，已加入功能黑名单。
- **Unreal Live Link / MetaHuman 串流**：身体骨骼与表情经 Live Link 串流到 Unreal；
  桥接模块仅在 Windows 上构建（`Setup_UnrealStream.cmd`，Win64 DLL 校验），
  Linux 端不可用，串流目标下拉框已隐藏该选项；Blender 串流目标不受影响。
- **SAM 3.1 CPU 卸载与 MAMMA 相机预览恢复**：沿用既有模型黑名单，代码随上游合并，Windows 行为不变。
- 更多通用说明见[用户指南](docs/USER_GUIDE.md)与[Unreal 串流指南](docs/UNREAL_LIVE_LINK.md)（Windows）。

## 上游同步新增（2026-11-10，自 ef05127a）

- **导出选项**：源动作 FBX / BVH 导出新增每人物原点放置与对称休息姿态选项，
  并新增 Maya 兼容 FBX 导出（见 [用户指南](docs/USER_GUIDE.md)）。
- **Blender 骨骼串流**：GUI 播放或拖动时间轴时，可将所选人物的骨骼实时串流到 Blender 并烘焙成 Action；
  插件位于 [`blender_addon/`](blender_addon/rgbmocap_bone_stream.zip?raw=1)，Linux 端尚未验证。
- **FlowHMR 下载修复**：递归模型下载的进度回调修复（不影响推理输入）。
- **GEM-X Native77 与 SAM 3.1 人物跟踪**：上游仅提供 Windows 安装
  （`Setup_GEMX.cmd` / `Setup_SAM31.cmd`），本分支 Linux 端暂不可用，详见
  [GEM-X](docs/models/GEMX.md) 与 [SAM 3.1](docs/models/SAM31.md) 指南。
- **Windows 安装器改进**：更新自替换修复与运行时校验复用，不影响 Ubuntu 安装流程。
- 更多通用模型与 GUI 功能说明见[模型指南](docs/models/README.md)与[用户指南](docs/USER_GUIDE.md)。

## Linux 支持与验证范围

| 能力 | 当前状态 |
| --- | --- |
| FlowHMR + HaMeR + 相机估计 + 修正 + 源动作 FBX | 已有 x86-64 与 GB10 实机验证记录 |
| Ubuntu GUI | 已验证项目加载与播放；GB10 已通过 ARM64 Qt 修复后的 60 秒原生网格播放检查 |
| ExMR、GVHMR 及相关附加模型 | 已接入独立环境，尚未完成 Linux 端到端验证 |
| 多 GPU 并行批量、CLI、项目模板、deb 安装 | 本分支功能，已在开发机验证 |
| MAMMA、PromptHMR、PEAR 实时预览 | 当前不支持 Linux |
| ARM64 Blender 打包与角色重定向 | 需要自行提供兼容的 Blender 4.5 |
| Maya FBX、原点放置 / 对称休息姿态导出、Blender 骨骼串流、UE5 Mannequin FBX、take 选择导出、A/T 参考姿态、预览音频 | 代码已随上游合入，尚未在 Linux 验证 |
| GEM-X Native77、SAM 3.1 人物跟踪、Track Face 人脸跟踪、Unreal Live Link / MetaHuman 串流 | 上游仅支持 Windows，Linux 暂不可用（界面已隐藏） |

验证结论仅对应记录中的硬件、输入与设置。详细结果见 [Ubuntu 验证](docs/UBUNTU_VALIDATION.md)、
[Spark 验证](docs/SPARK_VALIDATION.md)和 [GB10 GUI 修复与播放验证](docs/validation/gb10-gui-crash-2026-11-07.md)。

## 更多资料

- [Ubuntu 安装、CLI、架构差异与上游同步](docs/UBUNTU.md)
- [上游同步经验与操作流程](docs/UPSTREAM_SYNC.md)
- [arm64 deb 安装说明](docs/DEB.md)
- [PRO 4000 / GB10 性能测量与实验性双卡结果](docs/BENCHMARK_RESULTS_UBUNTU.md)（双卡实验未接入生产推理）
- [Agent 操作指南](docs/AGENT_GUIDE.md)
- [第三方组件与模型许可](THIRD_PARTY_NOTICES.md)

原项目代码与文档采用 [MIT License](LICENSE)；第三方组件和模型文件遵循各自许可。
