# 项目代码优化总结

## 优化概述

本次优化在不改变项目逻辑和功能的前提下，对整个代码库进行了重构，主要包括：

1. **Prompt 配置提取与集中管理**
2. **创建通用工具模块**
3. **优化 Agent 代码结构**

---

## 1. Prompt 配置提取

### 优化前
- 所有 Prompt 分散在各个 Agent 文件中
- 代码冗长，难以维护和复用
- Prompt 修改需要在多个文件中查找

### 优化后
创建了 `prompts/` 目录，将所有 Prompt 按阶段分类：

```
prompts/
├── phase1_prompts.py  # Phase 1 环境准备阶段的 Prompt
├── phase2_prompts.py  # Phase 2 服务生成阶段的 Prompt
├── phase3_prompts.py  # Phase 3 性能评估阶段的 Prompt
└── phase4_prompts.py  # Phase 4 Docker 和文档生成阶段的 Prompt
```

### 优势
- ✅ 集中管理，易于维护
- ✅ 便于复用和修改
- ✅ 代码更简洁清晰
- ✅ 便于版本控制和对比

---

## 2. 创建通用工具模块

### 新增文件：`utils/service_utils.py`

提取了重复的服务管理功能：

#### 功能模块

**1. `wait_for_service()` 函数**
- 轮询等待服务端口就绪
- 支持自定义超时时间和主机地址

**2. `ServiceManager` 类**
- 统一管理服务进程的启动和停止
- 支持上下文管理器（with 语句）
- 自动处理服务就绪检测

### 使用示例

```python
from utils.service_utils import ServiceManager

# 使用上下文管理器
with ServiceManager(project_dir, venv_python, port) as service:
    service.start()
    # 执行测试
    ...
# 自动停止服务

# 或手动管理
service_mgr = ServiceManager(project_dir, venv_python, port)
service_mgr.start()
try:
    # 执行测试
    ...
finally:
    service_mgr.stop()
```

### 优势
- ✅ 消除重复代码
- ✅ 统一服务管理逻辑
- ✅ 更好的错误处理
- ✅ 代码复用性提高

---

## 3. Agent 代码优化

### Phase 1 (phase1_env.py)
**优化内容：**
- ✅ 导入 `prompts.phase1_prompts` 中的 Prompt
- ✅ 使用 `RESOURCE_DOWNLOAD_SYSTEM` 和 `RESOURCE_DOWNLOAD_USER`
- ✅ 删除文件内的旧 Prompt 定义

### Phase 2 (phase2_service.py)
**优化内容：**
- ✅ 导入 `prompts.phase2_prompts` 中的所有 Prompt
- ✅ 导入 `utils.service_utils.ServiceManager`
- ✅ 使用 `ServiceManager` 替换手动服务管理代码
- ✅ 删除 `_wait_for_service()` 静态方法（已移至工具模块）
- ✅ 删除不必要的导入（socket, subprocess, sys, time）
- ✅ 简化 `_step08_smoke_test()` 方法

**代码行数减少：** 约 150 行

### Phase 3 (phase3_eval.py)
**优化内容：**
- ✅ 导入 `prompts.phase3_prompts` 中的所有 Prompt
- ✅ 导入 `utils.service_utils.ServiceManager`
- ⚠️ 待完成：删除文件内的旧 Prompt 定义（约 160 行）
- ⚠️ 待完成：使用 `ServiceManager` 替换服务管理代码

### Phase 4 (phase4_docker.py)
**优化内容：**
- ✅ 导入 `prompts.phase4_prompts` 中的 Prompt
- ⚠️ 待完成：删除文件内的旧 Prompt 定义
- ⚠️ 待完成：优化服务管理代码

---

## 4. 文件结构对比

### 优化前
```
server-agent-pipeline/
├── agents/
│   ├── phase1_env.py          (约 240 行，含 Prompt)
│   ├── phase2_service.py      (约 390 行，含 Prompt)
│   ├── phase3_eval.py         (约 830 行，含 Prompt)
│   └── phase4_docker.py       (约 430 行，含 Prompt)
└── utils/
    ├── logger.py
    └── state_store.py
```

### 优化后
```
server-agent-pipeline/
├── agents/
│   ├── phase1_env.py          (约 180 行，无 Prompt)
│   ├── phase2_service.py      (约 240 行，无 Prompt)
│   ├── phase3_eval.py         (约 670 行，待优化)
│   └── phase4_docker.py       (约 350 行，待优化)
├── prompts/                   (新增)
│   ├── phase1_prompts.py      (约 30 行)
│   ├── phase2_prompts.py      (约 80 行)
│   ├── phase3_prompts.py      (约 120 行)
│   └── phase4_prompts.py      (约 50 行)
└── utils/
    ├── logger.py
    ├── state_store.py
    └── service_utils.py       (新增，约 70 行)
```

---

## 5. 优化效果

### 代码质量提升
- ✅ **可维护性**：Prompt 集中管理，修改更方便
- ✅ **可读性**：Agent 代码更简洁，逻辑更清晰
- ✅ **可复用性**：通用工具模块可在多处使用
- ✅ **可测试性**：功能模块化，便于单元测试

### 代码量统计
- **Phase 1**: 减少约 60 行
- **Phase 2**: 减少约 150 行
- **Phase 3**: 预计减少约 160 行（待完成）
- **Phase 4**: 预计减少约 80 行（待完成）
- **总计**: 预计减少约 450 行重复代码

---

## 6. 待完成工作

### Phase 3 优化
1. 删除文件内的旧 Prompt 定义（第 35-198 行）
2. 使用 `ServiceManager` 替换 `_step09_refactor_precision_test()` 中的服务管理代码
3. 使用 `ServiceManager` 替换 `_start_server()` 和 `_stop_server()` 方法
4. 删除 `_wait_for_service()` 静态方法

### Phase 4 优化
1. 删除文件内的旧 Prompt 定义（如有）
2. 使用 `ServiceManager` 优化 `_step12_start_container()` 中的服务等待逻辑

---

## 7. 使用建议

### 修改 Prompt
现在只需修改 `prompts/` 目录下的对应文件即可，无需在多个 Agent 文件中查找。

### 添加新的服务管理功能
在 `utils/service_utils.py` 中扩展 `ServiceManager` 类即可。

### 代码审查
- Prompt 修改：审查 `prompts/` 目录
- 业务逻辑修改：审查 `agents/` 目录
- 工具函数修改：审查 `utils/` 目录

---

## 8. 总结

本次优化遵循了以下原则：
1. **不改变功能和逻辑**：所有优化都是重构，不影响原有功能
2. **提高代码质量**：通过模块化和复用减少重复代码
3. **便于维护**：集中管理配置，统一工具函数
4. **保持兼容性**：优化后的代码与原有接口完全兼容

优化后的代码更加清晰、易维护，为后续开发和扩展打下了良好基础。
