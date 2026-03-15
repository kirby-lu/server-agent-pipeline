from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
from typing import Any  # 用于定义任意类型的响应结果
from single_inference_refactor import *

# 初始化 FastAPI 应用，统一命名规范，便于后续运维识别与管理
app = FastAPI(
    title="标准化推理服务接口",
    version="1.0",
    description="微服务封装标准化推理接口，支持单条数据推理，遵循统一请求/响应规范"
)

# 定义请求参数模型（严格遵循1.2.3.1请求格式，实现参数校验，规避无效输入导致的服务异常）
class RequestBodyData(BaseModel):
    resourceUrl: str  # 核心输入参数，资源地址（如图片路径、特征文件路径），必填项

class InferenceRequest(BaseModel):
    requestId: str  # 请求唯一ID，必填项，建议采用UUID格式
    body: RequestBodyData  # 请求体，包含推理所需的核心输入参数

class LatencyData(BaseModel):
    pre_process:float
    process:float
    post_process:float

# 定义响应参数模型（严格遵循1.2.3.2响应格式，统一输出规范）
class ResponseBodyData(BaseModel):
    result: Any = ""  # 推理结果，可接收任意类型（如字典、列表、字符串）
    status: str = ""  # 处理状态，success表示成功，失败时返回具体错误信息
    latency: float = 0.0  # 推理耗时（秒），用于性能评估

class InferenceResponse(BaseModel):
    requestId: str  # 与请求中的requestId一致，用于请求与响应关联
    body: ResponseBodyData  # 响应体，包含推理结果、处理状态、耗时
    errorCode: int = 200  # 错误码，200成功，400参数错误，500服务器异常
    version: str = "v1.0.0.0"  # 服务版本号，固定统一格式

# 服务启动时加载模型（仅加载一次，避免重复加载浪费资源，提升服务响应效率）
@app.on_event("startup")
def startup_event():
    init_model()
    print("模型加载完成，推理服务启动成功，可正常接收请求")

# 核心推理接口（统一接口路径、请求方式，便于标准化调用）
@app.post(
    "/infer",
    response_model=InferenceResponse,
    summary="推理服务核心接口",
    description="接收标准化请求参数，执行推理流程，返回格式化响应结果"
)
def infer(request: InferenceRequest):
    try:
        # 1. 参数校验：校验请求参数的合法性（重点校验必填字段）
        if not request.requestId:
            raise HTTPException(status_code=400, detail="请求参数不合法：requestId为必填项")
        if not request.body.resourceUrl:
            raise HTTPException(status_code=400, detail="请求参数不合法：resourceUrl为必填项")
        
        # 2. 数据预处理：提取请求体中的核心参数，调用预处理函数转换格式
        input_data = request.body.resourceUrl  # 提取资源地址作为输入数据
        pre_process_start = time.time()  
        processed_data = pre_process(input_data)
        pre_process_end = time.time()  
        
        # 3. 模型推理：调用推理函数，获取原始推理结果
        process_start = time.time()  
        raw_output = process(processed_data)
        process_end = time.time()  
        
        # 4. 数据后处理：对原始推理结果进行格式化，适配响应格式
        post_process_start = time.time()  
        final_result = post_process(raw_output)
        post_process_end = time.time()  
        
        # 计算推理耗时（保留4位小数，提升精度，用于后续性能评估）
        pre_process_latency = round(pre_process_end - pre_process_start, 4)
        process_latency = round(process_end - process_start, 4)
        post_process_latency = round(post_process_end - post_process_start, 4)
        
        return InferenceResponse(
            requestId=request.requestId,
            body=ResponseBodyData(
                result=final_result,
                status="success",
                latency=LatencyData(
                    pre_process=pre_process_latency,
                    process=process_latency,
                    post_process = post_process_latency)
            ),
            errorCode=200,
            version="v1.0.0.0"
        )
    except HTTPException as e:
        # 捕获参数错误，按响应规范返回错误信息
        return InferenceResponse(
            requestId=request.requestId if request.requestId else "unknown",
            body=ResponseBodyData(
                result="",
                status=e.detail,
                latency=0.0
            ),
            errorCode=e.status_code,
            version="v1.0.0.0"
        )
    except Exception as e:
        # 捕获其他异常（如模型调用失败、预处理异常），返回服务器异常信息
        return InferenceResponse(
            requestId=request.requestId if request.requestId else "unknown",
            body=ResponseBodyData(
                result="",
                status=f"推理服务执行失败：{str(e)}",
                latency=0.0
            ),
            errorCode=500,
            version="v1.0.0.0"
        )

# 本地调试启动入口（生产环境部署时需注释或删除，避免影响服务稳定性）
if __name__ == "__main__":
    import uvicorn
    # 启动服务，默认监听本地8090端口，开启自动重载（便于本地调试时实时生效修改）
    uvicorn.run("server:app", host="0.0.0.0", port=8090, reload=True)