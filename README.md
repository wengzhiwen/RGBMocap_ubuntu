# RGB MoCap · Ubuntu

基于 [toyxyz/RGB MoCap](https://github.com/toyxyz/toyxyz_RGBMocap) 的 Ubuntu 分支，
重点提供 **Ubuntu 原生运行、FlowHMR 命令行工作流和 NVIDIA DGX Spark / ARM64 适配**。
本页只介绍本分支新增能力；通用模型与 GUI 操作见[模型指南](docs/models/README.md)和[用户指南](docs/USER_GUIDE.md)。

## 本分支特色

- **Ubuntu 原生安装与启动**：用 `tools/setup_ubuntu.py` 和 Shell 启动器替代 Windows CMD 流程，
  为不同模型保留独立、固定版本的 `.venv*` 环境。
- **FlowHMR CLI**：从视频准备、自动单人跟踪、相机估计到动作重建、可选 HaMeR 手指增强和接触修正，
  支持 `--dry-run`、JSON 输出及单阶段调用。
- **GUI / CLI 共用项目**：调用相同 worker、使用相同项目文件和阶段缓存，命令行生成的项目可直接在 GUI 中打开。
  CLI 同样保留资源保护、停滞检测和进程组取消。
- **项目配置模板**：用已有 `project.json` 的详细配置处理新视频，模板决定处理参数，避免与命令行参数混用。
- **命令行导出**：导出 FBX、BVH 或便携 Blender 项目；支持 Biped BVH、前置 T-pose 和保存的相机动画等选项。
- **DGX Spark / ARM64**：自动选择独立的 Torch / torchvision CUDA 13 配置，支持原生动作修正的 NEON 编译适配；
  ARM64 GUI 使用单独的 Qt 版本，并在安装和诊断时实际检查绘图绑定。

## Ubuntu 快速开始

需要 64 位 Ubuntu（x86-64 或 aarch64）、兼容的 NVIDIA GPU / 驱动，以及可用的 `uv`。
GUI 还需要 OpenGL 4.3。系统依赖与 `uv` 查找方式见[完整安装指南](docs/UBUNTU.md#prerequisites)。

```bash
git clone https://github.com/wengzhiwen/RGBMocap_ubuntu.git
cd RGBMocap_ubuntu

sudo apt install ffmpeg libglu1-mesa libegl1 libgl1 libopengl0 libxcb-cursor0 cmake build-essential git

python3 tools/setup_ubuntu.py --plan
python3 tools/setup_ubuntu.py --stage-assets --build-correction
./rgbmocap.sh doctor --gpu
./rgbmocap-gui.sh
```

模型权重与人体数据需另行通过 **Model Manager** 下载或导入，安装脚本不包含这些文件。
部分数据需要注册并接受提供方许可。详见 [Ubuntu 安装与运行](docs/UBUNTU.md)。

## 命令行处理与导出

```bash
# 预览执行计划
./rgbmocap.sh flow --video clip.mp4 --hamer --correct --dry-run

# 单人重建、手指增强与动作修正
./rgbmocap.sh flow --video clip.mp4 --hamer --correct

# 使用上一步输出的模型项目路径
./rgbmocap.sh export --project /path/to/model_project --kind corrected_initial --output motion.fbx
./rgbmocap.sh export --project /path/to/model_project --kind corrected_initial --output motion.bvh
```

`flow` 当前自动跟踪一人；多人选择仍使用 GUI。`export` 导出已准备好的结果。
Blender 打包需要 Blender 4.5，源动作 FBX / BVH 导出不需要 Blender。
更多命令见 [CLI 文档](docs/UBUNTU.md#command-line)或 `./rgbmocap.sh flow --help`。

## 新增：用 project.json 复用处理配置

先在 GUI 项目中调好 FlowHMR 和动作修正设置，再将其作为新视频的模板：

```bash
./rgbmocap.sh flow \
  --video new_clip.mp4 \
  --template /path/to/old/project.json \
  --project /path/to/new_project \
  --dry-run
```

确认计划后去掉 `--dry-run` 执行。

- 接受工作区根目录或 FlowHMR 模型目录中的 `project.json`。
- 继承 FlowHMR 选项、是否启用修正和修正配置；`--hamer` / `--no-hamer`、相机模式等处理参数会被忽略，并在终端提示。
- 新视频、输出目录、`--start` / `--end` 裁剪范围、资源模式及日志选项仍由本次命令控制。
- 模板只读，不复制旧视频、人物、手工接触时间或结果缓存。目标目录须为新目录或空目录；省略 `--project` 会生成独立目录。

模板功能已通过 CPU 配置与路由测试，尚未单独完成真实视频 GPU 端到端验证。
完整规则见[模板使用说明](docs/UBUNTU.md#reuse-a-projects-processing-settings)。

## Linux 支持与验证范围

| 能力 | 当前状态 |
| --- | --- |
| FlowHMR + HaMeR + 相机估计 + 修正 + 源动作 FBX | 已有 x86-64 与 GB10 实机验证记录 |
| Ubuntu GUI | 已验证项目加载与播放；GB10 已通过 ARM64 Qt 修复后的 60 秒原生网格播放检查 |
| ExMR、GVHMR 及相关附加模型 | 已接入独立环境，尚未完成 Linux 端到端验证 |
| MAMMA、PromptHMR、PEAR 实时预览 | 当前不支持 Linux |
| ARM64 Blender 打包与角色重定向 | 需要自行提供兼容的 Blender 4.5 |

验证结论仅对应记录中的硬件、输入与设置。详细结果见 [Ubuntu 验证](docs/UBUNTU_VALIDATION.md)、
[Spark 验证](docs/SPARK_VALIDATION.md)和 [GB10 GUI 修复与播放验证](docs/validation/gb10-gui-crash-2026-11-07.md)。

## 更多资料

- [Ubuntu 安装、CLI、架构差异与上游同步](docs/UBUNTU.md)
- [PRO 4000 / GB10 性能测量与实验性双卡结果](docs/BENCHMARK_RESULTS_UBUNTU.md)（双卡实验未接入生产推理）
- [Agent 操作指南](docs/AGENT_GUIDE.md)
- [第三方组件与模型许可](THIRD_PARTY_NOTICES.md)

原项目代码与文档采用 [MIT License](LICENSE)；第三方组件和模型文件遵循各自许可。
