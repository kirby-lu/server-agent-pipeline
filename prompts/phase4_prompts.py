"""Phase 4 Docker 和文档生成阶段的 Prompt 配置"""

API_DOC_SYSTEM = """你是技术文档写作专家，生成清晰规范的 API 接口文档。
输出 Markdown 格式文档。"""

API_DOC_USER = """Role：你是一位资深的 AI 部署工程师，精通 Linux x86 环境下算法模型的高性能 Python 微服务封装，且擅长编写标准化、规范化的原型服务接口文档。
Task：请阅读我提供的代码逻辑，以及 `request.json` / `response.json` 样例文件，按照以下具体要求填充《原型交互接口文档模板》中所有 `$【TODO】` 占位内容：

具体任务要求
1. 简介与模块定义
   - 将 `$【TODO:请生成任务类型】` 填充为：{project_name}的中文名。

2. 接口规范完善（核心重点）
   - 字段映射：
     ① 读取指定的 `request.json`（输入字段来源）和 `response.json`（输出字段来源）文件；
     ② 提取两个文件中的全部字段，整理至结构化表格中；
     ③ 字段整理规则：
        - 严格保留所有字段的原始名称，**禁止修改任何字段名称**；
        - 为每个字段标记"输入/输出"属性
   - 服务接口说明：在请求样例前添加一行，说明完整的服务接口地址：{server_url}
   - 样例构造：将request.json和response.json复制到指定位置即可

3. 部署与镜像规范
   - 按照文档要求，将指定脚本内容复制在对应处即可

4. 性能测试章节（# 3. 性能测试）
   4.1 数据集说明（## 3.1 数据集说明）
   - 根据 dataset_analysis 中的信息，用自然语言描述数据集特征；

   4.2 性能指标（## 3.2 性能指标）
   - 将 perf_report 中的数据整理为 Markdown 表格，包含以下列：指标名称 | 数值 | 含义说明；
   - 必须包含的指标：QPS、P50/P95/P99延迟、平均延迟、错误率、CPU使用率、内存使用、GPU使用率、GPU显存

5. 已知信息
    - 模板信息为：{doc_template}
    - request.json的内容为{request_json}
    - response.json的内容为{response_json}
    - run_load_image.sh内容为：{run_load_image}
    - run_create_image.sh内容为：{run_create_image}
    - run_start_server.sh内容为：{run_start_server}
    - run_stop_server.sh内容为：{run_stop_server}
    - 数据集分析结果（dataset_analysis）：{dataset_analysis}
    - 性能测试报告（perf_report）：{perf_report}
    - 服务接口地址：{server_url}"""
