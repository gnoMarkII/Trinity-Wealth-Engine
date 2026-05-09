from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "Company", "Executive", "Macro_Event", "Strategy",
    "Concept", "Asset", "Person", "Organization", "Market_Data",
]


class MemoryEntry(BaseModel):
    title: str = Field(description="ชื่อของ entity หรือหัวข้อ ใช้เป็นชื่อไฟล์ .md ควรสั้น กระชับ และไม่ซ้ำกัน เช่น 'PTT', 'Somchai_Jatusripitak', 'Fed_Rate_Hike_2025'")
    content: str = Field(description="เนื้อหาหลักของ memory entry ในรูปแบบ Markdown ควรเขียนเกี่ยวกับ entity นี้โดยตรง ไม่ปนข้อมูลของ entity อื่น")
    folder_path: str = Field(description="โฟลเดอร์ปลายทางใน Vault เช่น '30_Knowledge_Base/Stocks', '20_Portfolio_Management/Current_Holdings' เลือกให้ตรงหมวดหมู่ของ entity")
    tags: list[str] = Field(description="รายการ tag สำหรับจัดหมวดหมู่และค้นหา เช่น ['energy', 'SET100', 'dividend'] ควรมีอย่างน้อย 2 tags")
    entity_type: EntityType = Field(description="ประเภทของ entity ที่บันทึก เลือกจาก: Company, Executive, Macro_Event, Strategy, Concept, Asset, Person, Organization, Market_Data")
    aliases: list[str] = Field(default_factory=list, description="ชื่อเรียกอื่นๆ ของ entity เดียวกัน เช่น PTT อาจมี aliases ['ปตท.', 'PTT PCL', 'บมจ.ปตท.'] ช่วยให้ค้นหาได้หลากหลาย")
    linked_files: list[str] = Field(default_factory=list, description="ชื่อไฟล์ .md อื่นๆ ใน Vault ที่เกี่ยวข้องกับ entity นี้ (ไม่รวมนามสกุล) เช่น ['PTT', 'Aromatics_Sector'] จะสร้าง Wikilinks เชื่อมโยงอัตโนมัติ")
