from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

class GeographicScope(str, Enum):
    GLOBAL = "Global"
    REGIONAL = "Regional"
    COUNTRY = "Country"

class Region(str, Enum):
    GLOBAL = "Global"
    # Regional
    US_REGIONAL = "US_Regional"
    EUROPE = "Europe"
    CHINA = "China"
    JAPAN = "Japan"
    INDIA = "India"
    LATAM = "Latin_America"
    # Country
    THAILAND = "Thailand"
    USA = "USA"

class EconomicIndicator(str, Enum):
    MONETARY_POLICY = "Monetary_Policy"
    ECONOMIC_GROWTH = "Economic_Growth"
    INFLATION = "Inflation"
    GEOPOLITICS = "Geopolitics"

class EconomicState(str, Enum):
    GOLDILOCKS = "Goldilocks"
    REFLATION = "Reflation"
    STAGFLATION = "Stagflation"
    RECESSION = "Recession"
    UNKNOWN = "Unknown"

class TrendDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"

class IndicatorComponent(BaseModel):
    symbol_or_id: str = Field(description="Ticker หรือ FRED Series ID หรือคีย์หลักของดัชนี")
    name: str = Field(description="ชื่อเรียกตัวแปรหรือดัชนี")
    value: float = Field(description="ค่าล่าสุดของตัวแปร")
    unit: str = Field(description="หน่วยของค่า เช่น %, % YoY, Index, USD")
    date: str = Field(description="วันที่ของตัวเลขล่าสุดที่มีรายงาน")
    change_pct: Optional[float] = Field(default=None, description="เปอร์เซ็นต์การเปลี่ยนแปลงเมื่อเทียบกับรอบก่อนหน้า (ถ้ามี)")

class CellMetrics(BaseModel):
    score: float = Field(description="คะแนนที่คำนวณและ Normalize ได้ ช่วง -1.0 ถึง 1.0 (เช่น Growth: -1.0 = หดตัวแรง, 1.0 = โตแรง)")
    trend: TrendDirection = Field(default=TrendDirection.FLAT, description="แนวโน้มระยะสั้น (UP, DOWN, FLAT)")
    status_label: str = Field(description="คำอธิบายสถานะย่อย เช่น 'Hawkish', 'Expansion', 'High Inflation'")
    components: list[IndicatorComponent] = Field(default_factory=list, description="รายการดัชนีย่อยที่ใช้ประกอบในเซลล์นี้")
    updated_at: datetime = Field(default_factory=datetime.now)

class RegionStateEvaluation(BaseModel):
    region: Region = Field(description="ภูมิภาคหรือประเทศที่ถูกประเมิน")
    scope: GeographicScope = Field(description="ระดับขอบเขตของข้อมูล (Global, Regional, Country)")
    evaluated_state: EconomicState = Field(description="ผลการระบุสภาวะเศรษฐกิจ")
    confidence_score: float = Field(description="คะแนนความเชื่อมั่นต่อการจัดกลุ่ม 0.0 - 1.0")
    recommended_assets: list[str] = Field(description="ประเภทสินทรัพย์เด่นที่แนะนำ")
    rationale: str = Field(description="สรุปคำอธิบาย/เหตุผลสนับสนุน")
    cells: dict[EconomicIndicator, CellMetrics] = Field(description="ผลวิเคราะห์แต่ละ Indicator สำหรับพื้นที่นี้")
    updated_at: datetime = Field(default_factory=datetime.now)

class MacroEconomicMatrix(BaseModel):
    evaluated_at: datetime = Field(default_factory=datetime.now)
    evaluations: dict[Region, RegionStateEvaluation] = Field(description="ผลประเมินของแต่ละภูมิภาค/ประเทศ")
