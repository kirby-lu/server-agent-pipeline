"""Phase 3 性能评估阶段的 Prompt 配置"""

PRECISION_REFACTOR_SYSTEM = """你是 MLOps 工程师，专注于将本地精度测试迁移为服务测试。
输出完整的 Python 测试脚本，不要任何额外解释。"""

PRECISION_REFACTOR_USER = """请作为 Python 后端开发与测试专家，协助我基于现有脚本进行重构，并生成一份新的精度验证脚本。
### 一、重构目标
将 `val_precision` 脚本中原有的**本地模型推理链路**，替换为基于 **RESTful API 的远程调用方式**，使其专注于服务精度验证。

### 二、重构要求
#### 1. 模块裁剪
参考 `server_refactor` 服务脚本，移除 `val_precision` 中与服务端重复的处理逻辑，包括但不限于：
- 模型加载（`init_model`）
- 本地预处理
- 本地推理
- 推理后处理

#### 2. 服务集成
- 使用 Python `requests` 模块调用远程推理接口
- 服务地址（`server_url`）：`{server_url}`
- 请求结构（`request_json`）：`{request_json}`
- 响应结构（`response_json`）：`{response_json}`

#### 3. 逻辑保留
必须完整保留以下原有逻辑：
- 数据集的循环读取流程
- 最终精度指标的计算逻辑（Precision / Recall / mAP 等）

#### 4. 输出精简
- 删除新脚本中所有冗余的中间过程打印（`print`）语句
- **仅保留最终精度指标的打印与返回**

### 三、参考脚本
| 脚本名称 | 内容 |
|---|---|
| `server_refactor` 服务脚本 | `{server_refactor}` |
| `val_precision` 精度验证脚本 | `{val_precision}` |

### 四、交付要求
请输出完整的重构后脚本，并在关键改动处附加注释，说明替换或删除的原因。"""

REGENERATE_USER = """请对以下代码进行审查和修复：
代码审查请求：
之前生成的代码存在执行错误，请仔细检查代码逻辑，并根据错误信息重新生成正确的版本。
相关代码：{val_precision}
错误信息：{error_info}
要求：
- 分析错误原因，定位问题所在
- 修复代码中的问题
- 重新生成完整、可运行的代码
- 确保代码质量，避免类似错误再次发生
请提供修复后的完整代码。"""

EXTRACT_PRECISION_SYSTEM = """你是一个专业的 MLOps 工程师。
你的任务是分析所提供内容，提取所有的关于精度的信息。
输出严格的 JSON 格式，不要有任何额外文字。"""

EXTRACT_PRECISION_USER = """分析以下{content}的内容，提取所有的关于精度的信息。

输出 JSON 格式，下面提供了样例，如果还有其他精度名称和数据，请在列表中进行追加：
{{
  "precision_info": [
    {{
      "精度名称": "精度结果"
    }}
  ],
  "notes": "其他注意事项"
}}

如果没有需要下载的资源，返回 {{"precision_info": []}}"""

PERF_REQUEST_GEN_SYSTEM = """你是一位 MLOps 压测工程师。
你的任务是分析数据集目录结构和请求模板，生成一批真实可用的压测请求体。
输出严格的 JSON 数组，不要有任何额外文字、注释或 Markdown 代码块。"""

PERF_REQUEST_GEN_USER = """## 任务
根据以下信息，生成 {sample_count} 条真实的 HTTP 压测请求体（JSON 数组）。

## 数据集目录结构
```
{dataset_structure}
```

## 请求模板（request.json）
```json
{request_template}
```

## 服务推理代码（server_refactor.py 中的 pre_process 函数，用于理解字段含义）
```python
{pre_process_code}
```

## 生成要求
- 每条请求体的结构必须与请求模板完全一致（字段名、层级不变）
- resourceUrl 等文件路径字段，填写数据集目录中真实存在的文件相对路径
- 从数据集中均匀采样，不要重复使用同一个文件
- 若数据集文件数量少于 {sample_count}，则允许适量重复，但顺序需打乱
- 只输出 JSON 数组，格式示例：
  [{{"requestId": "perf-001", "body": {{"resourceUrl": "data/img1.jpg"}}}}, ...]"""

DATASET_ANALYSIS_SYSTEM = """你是一位专业的数据集分析工程师。
你的任务是分析给定数据集目录下的文件信息，输出结构化的数据集描述。
输出严格的 JSON 格式，不要有任何额外文字。"""

DATASET_ANALYSIS_USER = """请分析以下数据集采样信息，生成一份结构化的数据集说明，用于填写性能测试报告。

## 数据集文件采样列表
```
{file_samples}
```

## 文件统计信息
{file_stats}

## 分析要求
根据文件后缀和统计信息判断数据集类型，并按对应规则分析：

- 图像数据集（jpg/jpeg/png/bmp/tiff 等）：
  分析内容：文件总数、总大小、平均文件大小、分辨率分布（若可采样）

- 视频数据集（mp4/avi/mov/mkv 等）：
  分析内容：文件总数、总大小、平均文件大小、分辨率、平均时长

- 音频数据集（wav/mp3/flac/ogg 等）：
  分析内容：文件总数、总大小、平均文件大小、平均时长

- 文本/其他数据集：
  分析内容：文件总数、总大小、格式说明

输出 JSON 格式：
{{
    "dataset_type": "图像|视频|音频|文本|混合|未知",
    "total_files": 0,
    "total_size_mb": 0.0,
    "avg_file_size_mb": 0.0,
    "format_distribution": {{"jpg": 10, "png": 5}},
    "extra_info": {{
        "说明字段": "对应值（如分辨率、时长等，视数据类型而定）"
    }},
    "summary": "一段自然语言描述，2-3句话概括数据集特征"
}}"""
