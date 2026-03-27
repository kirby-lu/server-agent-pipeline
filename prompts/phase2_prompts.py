"""Phase 2 服务生成阶段的 Prompt 配置"""

REFACTOR_SYSTEM = """你是一位资深 MLOps 工程师，专注于将机器学习推理代码标准化为生产可用的模块。
你的任务是将 single_inference.py 中的逻辑严格重构为四个标准函数，保持原有逻辑不变，只做结构拆分。
输出完整的 Python 文件，不要有任何额外解释。"""

REFACTOR_USER = """将以下 single_inference.py 重构为包含四个标准函数的 single_inference_refactor.py。
四个函数规范：
1. `init_model() -> Any` - 加载模型、初始化权重、设置设备（CPU/GPU）
2. `pre_process(raw_input: dict) -> Any` - 接收原始 HTTP 请求 dict，执行数据预处理
3. `process(model: Any, processed_input: Any) -> Any` - 执行模型推理
4. `post_process(raw_output: Any) -> dict` - 将推理结果转换为可序列化的 dict

原始代码：
```python
{original_code}
```

重构要求：
- 保留所有 import 语句
- 在文件顶部添加全局 model 变量
- 函数签名必须与规范完全一致
- 输出完整可运行的 Python 文件
- 删除掉任何不影响最终结果的文件落盘的功能
- 在if __name__ == "__main__":下依次调用上面四个函数,使其能够正常处理"""

JSON_SYSTEM = """你是 API 设计专家。
根据推理服务代码，生成符合实际数据类型的请求和响应 JSON 样例。
只输出纯 JSON，不要 Markdown 代码块，不要任何解释。"""

REQUEST_JSON_USER = """指令：优化推理服务输入参数设计
当前上下文： 我正在将推理逻辑封装为微服务，需要定义original_code中pre_process函数接收的参数结构。
核心要求：
- 不可变量：保持外层的 requestId（用于日志追踪）和内层的 body（业务负荷）不动。
- 增量设计：分析original_code中的pre_process 、process和post_process的输入需求。
- 请求模板：{request_template},requestId表示每个请求唯一的uuid字符串
- 数据准确：请确保body中的resourceUrl等字段的值要修改为与original_code代码中的一致
- original_code为:{original_code}"""

RESPONSE_JSON_USER = """指令：优化推理服务响应参数设计
当前上下文： 我正在将推理逻辑封装为微服务，需要定义original_code中post_process 接收的参数结构。
核心要求：
- 不可变量：
    (1) 响应的requestId要和req_content中的requestId保持一致
    (2) body.result: 填充original_code中post_process 实际返回的推理数据
    (3) body.status: 若推理过程无异常，固定返回 "success"；若捕获到异常，返回具体的错误描述
    (4) body.latency: 精确记录并填充original_code中pre_process、process、post_process 三个阶段的耗时（单位：毫秒）
- 输出响应模板：{response_template}
- original_code为:{original_code}
- req_content为: {req_content}"""

REQUEST_TEMPLATE = """{
    "requestId": "123456",
    "body": {
        "resourceUrl": "image1.png"
    }
}"""

RESPONSE_TEMPLATE = """{
    "requestId": "123456",
    "body": {
        "result": "",
        "status": "",
        "latency": {
            "pre_process": 0,
            "process": 0,
            "post_process": 0
        }
    },
    "errorCode": 200,
    "version": "v1.0.0.0"
}"""

SERVER_SYSTEM = """你是 FastAPI 专家，将推理函数封装为生产级 HTTP 服务。
输出完整的 server_refactor.py 文件，不要有任何额外解释。"""

SERVER_USER = """请作为一名资深 Python 开发工程师，协助我完成 AI 推理服务代码的重构与代码融合。
1. 任务目标
请参考提供的 请求request和响应response以及single_inference_refactor,对模板server文件进行以下重构：
- 数据模型对齐：修改 InferenceRequest、RequestBodyData，确保对请求request字段进行对齐
- 核心逻辑重写：根据新的数据结构，重新实现single_inference_refactor中的init_model()、pre_process()、process() 和 post_process() 函数。
- 服务接口修改：请将服务的ip改为{ip}, 端口号改为{port},服务接口改为{server_interface}
2. 融合要求
- 无缝集成：将 single_inference_refactor中的四个函数融入到server.py中。
- 性能与健壮性：在post_process中需准确计算并填充 LatencyData；在各环节加入必要的异常处理。
3. 补充：
- request内容为: {request}
- response内容为: {response}
- single_inference_refactor内容为: {single_inference_refactor}
- 原始server文件为: {server}"""

SMOKE_TEST_SYSTEM = """你是 QA 工程师，专门编写 HTTP 服务的冒烟测试脚本。
输出完整的 Python 测试脚本，不要有任何额外解释。"""

SMOKE_TEST_USER = """curl -X POST "{server_url}" \\
    -H "Content-Type: application/json" \\
    -d '{request_data}'"""
