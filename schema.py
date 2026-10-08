"""VLM 检查器的结构化输出定义（Pydantic v2）。

三个核心判断的语义（请与人工标注口径保持一致）：
- is_clean  : 该实例的 mask 是否"干净"——边界贴合物体轮廓、没有多余杂块/空洞。
- is_split  : 该实例是否被错误切成了多个互不连通的部分（同一物体多块）。
- is_sticky : 该实例是否与其他实例或背景"粘连"（接触处边界不分、混在一起）。
注意：is_clean=False 时，is_split 与 is_sticky 可以同时为 False
（例如 mask 形状不贴合、带杂块但既未分裂也未粘连）。

置信度 confidence 由 VLM 自评，用于后续实验分析，不参与准确率计分。
"""
from pydantic import BaseModel, Field


class InstanceCheck(BaseModel):
    instance_id: int = Field(description="渲染图中该实例的编号，必须与图上的编号一致")
    is_clean: bool = Field(description="mask 是否干净")
    is_split: bool = Field(description="是否被错误切分成多个部分")
    is_sticky: bool = Field(description="是否与其他实例/背景粘连")
    confidence: float = Field(ge=0.0, le=1.0, description="0~1 的自评置信度")
    reason: str = Field(description="一句话说明判断依据（中文，30 字以内）")


class SceneCheck(BaseModel):
    instances: list[InstanceCheck] = Field(
        description="渲染图中所有可见实例的检查结果列表"
    )
