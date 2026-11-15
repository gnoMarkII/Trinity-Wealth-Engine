"""Pure domain models & persona specifications for Trinity Guru Roundtable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SpeakerId(str, Enum):
    HERMES_MODERATOR = "hermes_moderator"
    VALUE_GURU = "value_guru"
    QUALITY_GURU = "quality_guru"
    GROWTH_GURU = "growth_guru"
    MACRO_TACTICAL_GURU = "macro_tactical_guru"
    SKEPTIC_GURU = "skeptic_guru"
    USER = "user"


class RoundtableStance(str, Enum):
    BUY = "BUY"
    OVERWEIGHT = "OVERWEIGHT"
    HOLD = "HOLD"
    UNDERWEIGHT = "UNDERWEIGHT"
    SELL = "SELL"
    REVIEW = "REVIEW"


class RoundStage(str, Enum):
    STAGE_INTRO = "stage_intro"
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CROSSFIRE = "crossfire"
    INTERJECTION_REPLY = "interjection_reply"
    SYNTHESIS = "synthesis"
    USER_INTERJECTION = "user_interjection"


class DebateIntensity(str, Enum):
    BALANCED = "balanced"
    HEATED = "heated"
    CROSSFIRE = "crossfire"


class DebateDepth(str, Enum):
    CONCISE = "concise"
    DEEP_YOUTUBE = "deep_youtube"


class ContextAttachmentState(str, Enum):
    NOT_ATTACHED_DO_NOT_ASSUME = "NOT_ATTACHED_DO_NOT_ASSUME"
    ATTACHED_EMPTY = "ATTACHED_EMPTY"
    ATTACHED_FLAT_BOOK = "ATTACHED_FLAT_BOOK"
    ATTACHED_HELD = "ATTACHED_HELD"
    ATTACHED_PRESENT = "ATTACHED_PRESENT"


class AudioStatus(str, Enum):
    READY = "ready"
    CACHED = "cached"
    FALLBACK_WEB_SPEECH = "fallback_web_speech"
    DISABLED = "disabled"


@dataclass(frozen=True)
class GeminiVoiceOption:
    voice_name: str
    gender_tone: str  # "male" | "female" | "neutral"
    label_th: str
    recommended_for: str
    default_pitch: float
    default_rate: float
    default_sample_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


GEMINI_VOICE_CATALOG: Tuple[GeminiVoiceOption, ...] = (
    GeminiVoiceOption(
        voice_name="Kore",
        gender_tone="female",
        label_th="สุขุม สมดุล เป็นมืออาชีพ — น้ำเสียงนิ่ง คมชัด น่าเชื่อถือ เหมาะกับการสรุปประเด็นและจัดระเบียบวงสนทนา",
        recommended_for="ผู้ดำเนินรายการ (Hermes), ประธานคณะกรรมการลงทุน (CIO)",
        default_pitch=1.0,
        default_rate=1.04,
        default_sample_text="สวัสดีค่ะ ดิฉันเฮอร์เมส วันนี้เราจะมาตกผลึกข้อมูลรอบด้านเพื่อหาทางเลือกที่คุ้มค่าที่สุดให้พอร์ตของคุณค่ะ",
    ),
    GeminiVoiceOption(
        voice_name="Charon",
        gender_tone="male",
        label_th="ทุ้มนุ่ม สุขุม อบอุ่นแบบผู้ใหญ่ — จังหวะพูดเนิบช้า มีน้ำหนัก เหมือนนักลงทุนอาวุโสที่ผ่านวิกฤตมาหลายรอบ",
        recommended_for="สายคุณค่า (Value Guru — โทนปู่ Warren Buffett / Benjamin Graham)",
        default_pitch=0.90,
        default_rate=0.96,
        default_sample_text="การลงทุนที่ดีต้องเริ่มจากส่วนเผื่อความปลอดภัยเสมอ ถ้าราคาแพงเกินมูลค่าที่แท้จริง เราก็แค่รออย่างใจเย็นครับ",
    ),
    GeminiVoiceOption(
        voice_name="Fenrir",
        gender_tone="male",
        label_th="หนักแน่น เด็ดขาด จริงจัง — เนื้อเสียงแน่น คม ชัดถ้อยชัดคำ สื่อถึงความพิถีพิถันและมาตรฐานสูง",
        recommended_for="สายคุณภาพและคูเมืองธุรกิจ (Quality Guru — โทน Charlie Munger / Terry Smith)",
        default_pitch=0.95,
        default_rate=1.00,
        default_sample_text="ผมขอซื้อธุรกิจที่ยอดเยี่ยมในราคาที่สมเหตุสมผล ดีกว่าซื้อธุรกิจธรรมดาในราคาถูกแต่ไม่มีคูเมืองป้องกันครับ",
    ),
    GeminiVoiceOption(
        voice_name="Puck",
        gender_tone="male",
        label_th="กระฉับกระเฉง คล่องแคล่ว มีพลัง — จังหวะพูดไว ตื่นตัว ชวนมองหาโอกาสและนวัตกรรมใหม่",
        recommended_for="สายเติบโต (Growth Guru — โทน Peter Lynch / Philip Fisher)",
        default_pitch=1.06,
        default_rate=1.06,
        default_sample_text="ถ้าบริษัทนี้กำลังขยายตลาดใหม่และกำไรโตก้าวกระโดด การขายหมูเร็วเกินไปอาจทำให้พลาดหุ้นสิบเด้งได้นะครับ",
    ),
    GeminiVoiceOption(
        voice_name="Aoede",
        gender_tone="female",
        label_th="นุ่มลึก มีมิติ สุขุมลุ่มลึก — น้ำเสียงมีจังหวะจะโคน ชวนคิดภาพใหญ่และวัฏจักรเศรษฐกิจ",
        recommended_for="สายมหภาคและกระแสเงิน (Macro Guru — โทน Howard Marks / Ray Dalio)",
        default_pitch=1.02,
        default_rate=1.02,
        default_sample_text="เราต้องรู้ก่อนว่าตอนนี้เข็มนาฬิกาวัฏจักรเศรษฐกิจและสภาพคล่องโลกกำลังอยู่ตรงไหน ก่อนจะตัดสินใจเพิ่มน้ำหนักลงทุนค่ะ",
    ),
    GeminiVoiceOption(
        voice_name="Orus",
        gender_tone="male",
        label_th="เข้ม ดุดัน เฉียบคม ตรงไปตรงมา — น้ำเสียงจริงจังเหมือนผู้ตรวจสอบบัญชีหรือนักวิเคราะห์ความเสี่ยง",
        recommended_for="สายซักค้านและจับผิดความเสี่ยง (Skeptic Guru — โทน Michael Burry / Jim Chanos)",
        default_pitch=0.88,
        default_rate=0.98,
        default_sample_text="เดี๋ยวก่อนครับ ก่อนจะมองแต่ภาพสวยหรู มาดูคุณภาพกระแสเงินสดและหนี้สินที่ซ่อนอยู่ในงบการเงินกันก่อนดีกว่า",
    ),
    GeminiVoiceOption(
        voice_name="Zephyr",
        gender_tone="female",
        label_th="สดใส ทันสมัย คล่องตัว เข้าใจง่าย — โทนเสียงสว่าง เป็นกันเอง เหมือนที่ปรึกษาการเงินรุ่นใหม่",
        recommended_for="สาย Growth / Tech Disruptor หรือผู้ดำเนินรายการสไตล์กันเอง",
        default_pitch=1.05,
        default_rate=1.05,
        default_sample_text="ลองมาเจาะลึกตัวเลขคาดการณ์รายได้และเทรนด์ผู้บริโภครอบนี้กันค่ะ มีจุดที่น่าจับตาหลายเรื่องเลย",
    ),
    GeminiVoiceOption(
        voice_name="Leda",
        gender_tone="female",
        label_th="นุ่มนวล ใจเย็น ละเอียดรอบคอบ — น้ำเสียงอ่อนโยนแต่มีเหตุผล เหมาะกับการอธิบายเรื่องซับซ้อนให้ฟังสบาย",
        recommended_for="สาย Quality Compounder / สายอนุรักษ์นิยม / Wealth Advisor",
        default_pitch=1.00,
        default_rate=0.98,
        default_sample_text="ค่อยๆ พิจารณาความสม่ำเสมอของผลตอบแทนต่อเงินลงทุนและวินัยของผู้บริหารในระยะยาวกันนะคะ",
    ),
    GeminiVoiceOption(
        voice_name="Callirrhoe",
        gender_tone="female",
        label_th="มั่นใจ ชัดเจน มีพลังโน้มน้าว — โทนเสียงผู้บริหารหญิงที่ตัดสินใจเฉียบขาด",
        recommended_for="สาย Macro Tactical / ประธานกรรมการความเสี่ยง (Risk Chair)",
        default_pitch=0.98,
        default_rate=1.02,
        default_sample_text="เมื่อสัญญาณโมเมนตัมและกระแสเงินทุนสถาบันเปลี่ยนทิศ เราต้องพร้อมปรับสัดส่วนพอร์ตอย่างมีวินัยค่ะ",
    ),
    GeminiVoiceOption(
        voice_name="Iapetus",
        gender_tone="male",
        label_th="สุภาพ เป็นกลาง สุขุมแบบนักวิชาการ — โทนเสียงเรียบง่าย สบายหู ฟังต่อเนื่องได้นานไม่ล้า",
        recommended_for="สาย Quantitative / นักเศรษฐศาสตร์ / Value สไตล์ Seth Klarman",
        default_pitch=0.96,
        default_rate=1.00,
        default_sample_text="หากประเมินจากอัตราคิดลดและความผันผวนของตลาดในปัจจุบัน กรอบราคาเป้าหมายควรยึดตามข้อมูลจริงชุดนี้ครับ",
    ),
    GeminiVoiceOption(
        voice_name="Enceladus",
        gender_tone="male",
        label_th="ทุ้มมีพลัง หนักแน่นแบบผู้นำ — เสียงก้องกังวาน ชัดเจน เหมาะกับการตั้งคำถามท้าทายสมมติฐาน",
        recommended_for="สาย Contrarian / Activist Investor / Tactical Trader",
        default_pitch=0.92,
        default_rate=1.02,
        default_sample_text="จังหวะที่ตลาดกลัวสุดขีดหรือมั่นใจสุดขีด มักเป็นจุดที่ความคาดหวังฉีกออกจากความจริงมากที่สุดครับ",
    ),
    GeminiVoiceOption(
        voice_name="Autonoe",
        gender_tone="female",
        label_th="คมชัด กระชับ ตรงประเด็น — น้ำเสียงนักวิเคราะห์งบการเงินที่เน้นตัวเลขและหลักฐาน",
        recommended_for="สาย Forensic / Skeptic หญิง / Quality Analyst",
        default_pitch=0.98,
        default_rate=1.00,
        default_sample_text="ตัวเลขกำไรสุทธิที่เห็นต้องนำมาเทียบกับกระแสเงินสดจากการดำเนินงานจริงทุกครั้งเพื่อตรวจจับสัญญาณเตือนค่ะ",
    ),
)

VOICE_BY_NAME: Dict[str, GeminiVoiceOption] = {v.voice_name: v for v in GEMINI_VOICE_CATALOG}
ALLOWED_VOICE_NAMES = frozenset(VOICE_BY_NAME.keys())


@dataclass(frozen=True)
class GuruPresetTemplate:
    preset_id: str
    speaker_id: SpeakerId
    label_th: str
    display_name: str
    short_name: str
    archetype: str
    speaking_style: str
    custom_directive: str
    recommended_voice: str
    emoji: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["speaker_id"] = self.speaker_id.value
        return d


GURU_PRESET_CATALOG: Tuple[GuruPresetTemplate, ...] = (
    # Hermes Moderator (3 presets)
    GuruPresetTemplate(
        preset_id="default",
        speaker_id=SpeakerId.HERMES_MODERATOR,
        label_th="Hermes Balanced CIO (ค่าเริ่มต้น)",
        display_name="Hermes (Moderator)",
        short_name="Hermes",
        archetype="Moderator & Portfolio Distiller",
        speaking_style="กลาง ชัดเจน ร้อยเรียงข้อถกเถียงของทุกฝ่ายให้ตกผลึกเทียบกับพอร์ตจริงของผู้ใช้",
        custom_directive="",
        recommended_voice="Kore",
        emoji="🪶",
    ),
    GuruPresetTemplate(
        preset_id="hermes_risk_chair",
        speaker_id=SpeakerId.HERMES_MODERATOR,
        label_th="Strict Risk Committee Chair",
        display_name="Hermes Risk Chair",
        short_name="Hermes",
        archetype="Strict Risk Committee Chair",
        speaking_style="เข้มงวด รอบคอบ ให้ความสำคัญกับ Drawdown, ความกระจุกตัว และ Kill-Switch ก่อนอนุมัติไม้",
        custom_directive="ก่อนสรุป BUY หรือ OVERWEIGHT ต้องตรวจสอบเพดานสัดส่วนพอร์ต ความเสี่ยง Drawdown และตั้ง Kill-Switch ให้วัดผลได้ชัดเจน",
        recommended_voice="Callirrhoe",
        emoji="⚖️",
    ),
    GuruPresetTemplate(
        preset_id="hermes_action_coach",
        speaker_id=SpeakerId.HERMES_MODERATOR,
        label_th="Action-Oriented Portfolio Coach",
        display_name="Hermes Action Coach",
        short_name="Hermes",
        archetype="Action-Oriented Portfolio Coach",
        speaking_style="กระชับ เป็นกันเอง แปลงความเห็นซับซ้อนเป็นขั้นตอนแบ่งไม้ที่ทำตามได้จริงทันที",
        custom_directive="เน้นสรุป action_steps แบบแบ่งไม้ชัดเจน พร้อมโซนราคาเข้าซื้อ/ลดไม้, Watchlist Trigger และสัดส่วนจริงของพอร์ต",
        recommended_voice="Zephyr",
        emoji="🧭",
    ),
    # Value Guru (3 presets)
    GuruPresetTemplate(
        preset_id="default",
        speaker_id=SpeakerId.VALUE_GURU,
        label_th="Buffett & Graham Value (ค่าเริ่มต้น)",
        display_name="Value & MoS Guru",
        short_name="Value",
        archetype="Buffett / Graham Value Investor",
        speaking_style="สุขุม รอบคอบ แบบนักลงทุนอาวุโส ชั่งน้ำหนักราคาเทียบมูลค่าที่แท้จริงเสมอ",
        custom_directive="",
        recommended_voice="Charon",
        emoji="🏛️",
    ),
    GuruPresetTemplate(
        preset_id="value_graham_deep",
        speaker_id=SpeakerId.VALUE_GURU,
        label_th="Benjamin Graham Deep Value",
        display_name="Deep Value (Graham)",
        short_name="Graham",
        archetype="Benjamin Graham Deep Value Contrarian",
        speaking_style="เน้นตัวเลขงบดุล สินทรัพย์สุทธิ P/E, P/BV และส่วนเผื่อความปลอดภัยเข้มงวดเป็นพิเศษ",
        custom_directive="ตั้งเกณฑ์ Margin of Safety >= 25-30%, ตรวจ Current Ratio >= 1.5x และท้วงทันทีหาก FCF Yield ต่ำกว่าผลตอบแทนพันธบัตร",
        recommended_voice="Iapetus",
        emoji="📐",
    ),
    GuruPresetTemplate(
        preset_id="value_klarman_margin",
        speaker_id=SpeakerId.VALUE_GURU,
        label_th="Seth Klarman Margin-First",
        display_name="Klarman Downside Guard",
        short_name="Klarman",
        archetype="Seth Klarman Asymmetric Downside Protector",
        speaking_style="ตั้งคำถามด้านความเสี่ยงขาลง (Downside Risk) ก่อนประเมินกำไรขาขึ้น อดทนรอจังหวะตลาดตกใจ",
        custom_directive="ประเมิน Downside Risk จาก Bear DCF/Sensitivity และ Historical MDD ก่อนเสมอ แล้วระบุแนวรับที่คุ้มค่าความเสี่ยง",
        recommended_voice="Enceladus",
        emoji="🏰",
    ),
    # Quality Guru (3 presets)
    GuruPresetTemplate(
        preset_id="default",
        speaker_id=SpeakerId.QUALITY_GURU,
        label_th="Munger & Terry Smith (ค่าเริ่มต้น)",
        display_name="Quality & Moat Guru",
        short_name="Quality",
        archetype="Munger / Terry Smith Compounder",
        speaking_style="เฉียบคม มองทะลุโครงสร้างอุตสาหกรรม ROIC, Pricing Power และความยั่งยืนของคูเมือง",
        custom_directive="",
        recommended_voice="Fenrir",
        emoji="💎",
    ),
    GuruPresetTemplate(
        preset_id="quality_akre_compounder",
        speaker_id=SpeakerId.QUALITY_GURU,
        label_th="Chuck Akre Compounding Machine",
        display_name="Compounder (Akre)",
        short_name="Akre",
        archetype="Chuck Akre 3-Legged Compounding Specialist",
        speaking_style="นุ่มลึก มองเกมยาว เน้นธุรกิจดี เงินสดทบต้น และช่องทาง Reinvestment ระยะ 5-10 ปี",
        custom_directive="ตรวจ Akre 3-Legged Stool: ROIC >= 15%, ความทนทานของ Moat และ Reinvestment Runway 5-10 ปีจาก CapEx/D&A และเทรนด์ 3 ปี",
        recommended_voice="Leda",
        emoji="🌱",
    ),
    GuruPresetTemplate(
        preset_id="quality_forensic_auditor",
        speaker_id=SpeakerId.QUALITY_GURU,
        label_th="Forensic Quality Auditor",
        display_name="Quality Auditor",
        short_name="Auditor",
        archetype="Forensic Accounting & F-Score Analyst",
        speaking_style="ตรงประเด็น อ้างอิง Piotroski F-Score, Sloan Accrual และสัดส่วน OCF ต่อ Net Income",
        custom_directive="เทียบ GAAP vs Non-GAAP Margin เพื่อจับ SBC, เช็ก OCF/Net Income >= 0.85x และ Sloan Accrual หรือ Dilution ใน F-Score",
        recommended_voice="Autonoe",
        emoji="🔬",
    ),
    # Growth Guru (3 presets)
    GuruPresetTemplate(
        preset_id="default",
        speaker_id=SpeakerId.GROWTH_GURU,
        label_th="Peter Lynch & Fisher Growth (ค่าเริ่มต้น)",
        display_name="Growth & Expectations Guru",
        short_name="Growth",
        archetype="Peter Lynch / Fisher Growth",
        speaking_style="มีพลัง มองหาอัตราโตที่ตลาดยังประเมินไม่ครบ เทียบ Implied Growth กับศักยภาพจริง",
        custom_directive="",
        recommended_voice="Puck",
        emoji="🚀",
    ),
    GuruPresetTemplate(
        preset_id="growth_disruptor",
        speaker_id=SpeakerId.GROWTH_GURU,
        label_th="Asymmetric Disruptor",
        short_name="Disruptor",
        display_name="Asymmetric Disruptor",
        archetype="Long-Horizon S-Curve & Operating Leverage",
        speaking_style="กระตือรือร้น วิสัยทัศน์ไกล มองการขยาย TAM และผลตอบแทนแบบไม่สมมาตรในระยะหลายปี",
        custom_directive="วิเคราะห์ S-Curve, Operating Leverage 3 ปี (R&D% & EBIT Margin%) และเทียบ Fundamental Growth กับ Implied Growth จากราคาตลาด",
        recommended_voice="Zephyr",
        emoji="⚡",
    ),
    GuruPresetTemplate(
        preset_id="growth_eps_momentum",
        speaker_id=SpeakerId.GROWTH_GURU,
        label_th="Earnings Revision Momentum",
        display_name="Growth Revision Pro",
        short_name="Revision",
        archetype="EPS Revision & Fundamental Momentum",
        speaking_style="คมชัด ฉับไว เน้นดู EPS Revision 30 วัน, Guidance Beat และแนวโน้มมาร์จิน",
        custom_directive="เจาะทิศทาง eps_revision_net_30d, การปรับ Margin ใน Guidance ล่าสุด และ Catalyst จาก Transcript/ข่าวในช่วง 1-2 ไตรมาส",
        recommended_voice="Enceladus",
        emoji="📈",
    ),
    # Macro & Tactical Guru (3 presets)
    GuruPresetTemplate(
        preset_id="default",
        speaker_id=SpeakerId.MACRO_TACTICAL_GURU,
        label_th="Marks & Druckenmiller (ค่าเริ่มต้น)",
        display_name="Macro & Tactical Guru",
        short_name="Macro",
        archetype="Howard Marks / Druckenmiller Cycle & Flow",
        speaking_style="อ่านเข็มนาฬิกาวัฏจักรตลาด สภาพคล่อง เงินเฟ้อ และจังหวะการจัดสรรเงินอย่างยืดหยุ่น",
        custom_directive="",
        recommended_voice="Aoede",
        emoji="🌊",
    ),
    GuruPresetTemplate(
        preset_id="macro_dalio_allweather",
        speaker_id=SpeakerId.MACRO_TACTICAL_GURU,
        label_th="Ray Dalio All-Weather",
        display_name="Dalio Regime Strategist",
        short_name="Dalio",
        archetype="Ray Dalio Macro Regime & Risk Balance",
        speaking_style="เป็นระบบ อ้างอิงระบอบเศรษฐกิจ (Growth vs Inflation) และความสัมพันธ์ระหว่างสินทรัพย์",
        custom_directive="ระบุระบอบ Growth vs Inflation และตรวจ Sensitivity ต่อ Rate Shock +100bps ว่าหุ้นนี้ช่วยกระจายหรือซ้ำเติมความเสี่ยงพอร์ต",
        recommended_voice="Iapetus",
        emoji="🌐",
    ),
    GuruPresetTemplate(
        preset_id="macro_flow_tactician",
        speaker_id=SpeakerId.MACRO_TACTICAL_GURU,
        label_th="Institutional Flow Tactician",
        display_name="Flow & Rotation Chair",
        short_name="Tactical",
        archetype="Sector Rotation & Institutional Flow",
        speaking_style="เฉียบขาด เน้นอ่านทิศทาง Smart Money, Bond Yield และจังหวะ Technical / Stance",
        custom_directive="เชื่อมโยง Asset Allocation & Sector Rotation ใน @Macro เข้ากับ Tactical Stage, R:R และ Smart Money Flow เพื่อวางจุดเข้าไม้",
        recommended_voice="Callirrhoe",
        emoji="🎯",
    ),
    # Skeptic Guru (3 presets)
    GuruPresetTemplate(
        preset_id="default",
        speaker_id=SpeakerId.SKEPTIC_GURU,
        label_th="Burry & Chanos Bear (ค่าเริ่มต้น)",
        display_name="Skeptic / Devil's Advocate",
        short_name="Skeptic",
        archetype="Chanos / Burry Risk & Falsification",
        speaking_style="เข้ม ตรงไปตรงมา ตั้งคำถามทวนกระแสเพื่อตรวจหาจุดอ่อนและสมมติฐานที่เปราะบาง",
        custom_directive="",
        recommended_voice="Orus",
        emoji="🛡️",
    ),
    GuruPresetTemplate(
        preset_id="skeptic_forensic_sleuth",
        speaker_id=SpeakerId.SKEPTIC_GURU,
        label_th="Accounting & Debt Sleuth",
        display_name="Forensic Risk Sleuth",
        short_name="Forensic",
        archetype="Forensic Bear & Balance Sheet Critic",
        speaking_style="เยือกเย็น เจาะจง จับสัญญาณเตือนหนี้สิน (`net_debt_ebitda`), ลูกหนี้ และธง Data Quality",
        custom_directive="ขุดหาจุดเปราะบางในงบดุล: Synthetic Credit Rating, ICR, Net Debt/EBITDA, Data Quality Flags และยอดขาย Insider Open-Market",
        recommended_voice="Autonoe",
        emoji="🔎",
    ),
    GuruPresetTemplate(
        preset_id="skeptic_taleb_tailrisk",
        speaker_id=SpeakerId.SKEPTIC_GURU,
        label_th="Taleb Tail-Risk Guardian",
        display_name="Tail-Risk Critic",
        short_name="TailRisk",
        archetype="Nassim Taleb Antifragile Stress Tester",
        speaking_style="ดุดัน เฉียบขาด ทดสอบภาวะวิกฤตที่ตลาดประมาทความเสี่ยงหางอ้วน (Fat-Tail / Black Swan)",
        custom_directive="Stress-Test ความเสี่ยงหางอ้วนจาก Macro Risk Scenarios, Terminal Value%, Historical MDD/Amihud และ Port Overconcentration",
        recommended_voice="Fenrir",
        emoji="⚠️",
    ),
)

PRESET_BY_SEAT_AND_ID: Dict[Tuple[str, str], GuruPresetTemplate] = {
    (p.speaker_id.value, p.preset_id): p for p in GURU_PRESET_CATALOG
}


# Deep backend analytical rubrics (Priority Metrics, Decision Thresholds, Round 2 Cross-Exam)
# kept out of UI custom_directive (<textarea maxLength=500>) and injected at runtime.
PRESET_ANALYTICAL_RUBRIC_MAP: Dict[Tuple[str, str], str] = {
    # Hermes Moderator
    ("hermes_moderator", "default"): (
        "1) Priority Metrics: action_stance_hint, DATA QUALITY FLAGS รายหุ้น, Goal Tension & Required CAGR, Cash% & Bucket Room.\n"
        "2) Thresholds: ห้ามสรุป BUY หาก action_stance_hint เป็น WAIT/TRIM/REDUCE; ตรวจ exclude_targets & risk_limits ครบ.\n"
        "3) Focus: ชั่งน้ำหนักหลักฐานเชิงตัวเลขของทุกสายเทียบพอร์ตและเป้าหมายผู้ใช้โดยไม่เอนเอียงตามลำดับการพูด"
    ),
    ("hermes_moderator", "hermes_risk_chair"): (
        "1) Priority Metrics: Top-1 (>15%) & Top-3 (>50%) Concentration, Look-Through overlap, Max Drawdown limit ใน @Essence, "
        "Portfolio Cash%, Goal priority_tier='needs' และ Tension Level, DATA QUALITY FLAGS.\n"
        "2) Decision Thresholds: ห้ามอนุมัติ BUY หรือเพิ่มน้ำหนักหาก Single-Stock > 15% หรือ Bucket เต็ม/เกินเป้า หรือ Goal 'needs' "
        "มีความตึงตัวสูง; ต้องระบุ Liquidity Cushion (ADTV/EmergencyReserve) ก่อนสรุปขนาดไม้.\n"
        "3) Playbook Focus: บังคับให้ kill_switches ทุกข้อระบุตัวแปร + เกณฑ์ตัวเลข (เช่น Net Debt/EBITDA, MDD%, Margin) ที่วัดผลได้ชัดเจน"
    ),
    ("hermes_moderator", "hermes_action_coach"): (
        "1) Priority Metrics: mos_entry_price_today, buy_zone_min..max, Watchlist Target Price & Basis, Cash THB/USD available, "
        "monthly_investable_contribution และ Bucket Remaining Room.\n"
        "2) Decision Thresholds: ทุกคำแนะนำใน action_steps ต้องแปลงเป็นตัวเลขจับต้องได้ (ช่วงราคาเข้า, สัดส่วน/จำนวนไม้, Trigger เติม/ลดไม้); "
        "หากหุ้นยังสูงกว่า Watchlist Target หรือ Buy Zone ให้ตั้งแผน Staged Limit Order รอจังหวะอย่างเป็นรูปธรรม.\n"
        "3) Playbook Focus: แยกทางเลือกอนุรักษ์นิยม vs เชิงรุกให้ผู้ใช้ตัดสินใจทำตามได้ทันทีในสัปดาห์นี้"
    ),
    # Value Guru
    ("value_guru", "default"): (
        "1) Priority Metrics: mos_entry_price_today, current_margin_of_safety_pct, fcf_yield_pct, pe_vs_peer_avg_pct, "
        "normalized_mid_cycle_ebit_margin_pct (ถ้าเป็น cyclical), Watchlist Target Price.\n"
        "2) Thresholds: BUY เมื่อ MoS >= 15–20% และ TV% < 75%; ระวัง Peak Margin Normalization ในหุ้นวัฏจักร.\n"
        "3) Round 2 Cross-Exam: ทดสอบว่า Growth ที่ตลาด Price-in (`market_implied_growth_pct`) คุ้มค่าส่วนเผื่อความปลอดภัยจริงหรือไม่"
    ),
    ("value_guru", "value_graham_deep"): (
        "1) Priority Metrics: fcf_yield_pct เทียบ US10Y/Bond Yield ใน @Macro, current_ratio, net_debt_ebitda, de_ratio_pct, "
        "pe_vs_peer_avg_pct, price_percentile_5y, mos_entry_price_today และ synthetic_credit_rating / interest_coverage_ratio.\n"
        "2) Decision Thresholds (Red Lines): เกณฑ์ซื้อเข้มงวดต้องมี Margin of Safety >= 25–30%, Current Ratio >= 1.5x, "
        "Net Debt/EBITDA < 2.0x; หาก FCF Yield ต่ำกว่าผลตอบแทนพันธบัตร 10 ปี ให้ทักท้วงทันที (ยกเว้นมี Asset/Net-Cash Backstop ชัด); "
        "หาก Valuation พึ่ง Terminal Value > 75% ให้ปรับเป็น HOLD/UNDERWEIGHT.\n"
        "3) Round 2 Cross-Exam: ท้าทายฝั่ง Growth และ Quality ว่ากำลังจ่ายแพงเกิน Intrinsic Value โดยไร้ Balance Sheet Backstop หรือไม่"
    ),
    ("value_guru", "value_klarman_margin"): (
        "1) Priority Metrics: Bear DCF Target & Sensitivity Min Target Price, macro_yield_shock_100bps_delta, "
        "historical mdd_pct / volatility_pct, adtv_local_currency / amihud_illiquidity_20d, normalized_mid_cycle_ebit_margin_pct, Portfolio Cash%.\n"
        "2) Decision Thresholds (Red Lines): ประเมิน Downside กรณี Bear Case ก่อนเสมอ — หากความเสี่ยงขาดทุนเทียบแนวรับ Bear "
        "เกินกว่า Upside 2:1 หรือเป็นหุ้นวัฏจักรที่ Normalized Margin ต่ำกว่าปัจจุบันมาก ให้ชะลอและเก็บ Cash รอ Market Dislocation; "
        "ตรวจสอบสภาพคล่อง (ADTV/Amihud) หากสภาพคล่องตื้นต้องบวกส่วนลด MoS เพิ่ม.\n"
        "3) Round 2 Cross-Exam: ตั้งคำถามว่าหากสมมติฐานดีที่สุดไม่เกิดขึ้นจริง นักลงทุนจะเสียเงินต้นกี่เปอร์เซ็นต์"
    ),
    # Quality Guru
    ("quality_guru", "default"): (
        "1) Priority Metrics: roic_pct, piotroski_f_score (9 pts), ocf_to_net_income, 3Y Trajectory (Gross/EBIT Margin, CapEx/D&A), sector_health_score.\n"
        "2) Thresholds: แยก Compounder (ROIC >= 15%, F-Score >= 7, OCF/NI >= 0.85x) ออกจาก Value Trap.\n"
        "3) Round 2 Cross-Exam: ตรวจสอบว่า Growth ที่พูดถึงสร้าง ROIIC สูงจริงหรือเพียงทุ่ม CapEx แลกยอดขาย"
    ),
    ("quality_guru", "quality_akre_compounder"): (
        "1) Priority Metrics: Akre 3-Legged Stool — (1) roic_pct, (2) Sector KPIs & sector_health_score + Pricing Power / Moat "
        "ใน narrative_analysis, (3) 3-Year Historical Trajectory (Gross/EBIT Margin stability, R&D%, FCF, capex_to_depreciation_ratio, "
        "reinvestment_lifecycle_phase) และ fcf_cagr_3y / net_shareholder_yield_pct.\n"
        "2) Decision Thresholds (Red Lines): Compounder แท้ต้องมี ROIC >= 15–20% ต่อเนื่อง, Gross/EBIT Margin มั่นคงหรือขยายตัวใน 3Y Trend, "
        "และมี Reinvestment Runway ยาว (ไม่ใช่ทุ่ม CapEx เกินค่าเสื่อมแล้ว FCF ทรุดไร้ผลตอบแทน); ห้ามขายเพียงเพราะ Valuation ตึงเล็กน้อยถ้า Moat ยังแกร่ง.\n"
        "3) Round 2 Cross-Exam: ถกกับสาย Value และ Macro เรื่องระยะเวลาการทบต้น 5–10 ปี และความยั่งยืนของ ROIIC"
    ),
    ("quality_guru", "quality_forensic_auditor"): (
        "1) Priority Metrics: ocf_to_net_income, sloan_accrual, dupont_breakdown, GAAP vs Non-GAAP Operating Margin & metric_basis "
        "(วัดผลกระทบ SBC/Adjustments), และ piotroski_f_score แยกรายข้อย่อย (AccrualOK, NoDilution, GMUp, LevDown).\n"
        "2) Decision Thresholds (Red Lines): เตือนทันทีหาก OCF/Net Income < 0.85x, Sloan Accrual สูง, Non-GAAP สูงกว่า GAAP Operating Margin > 5% "
        "จาก Stock-Based Compensation หนัก, หรือ Piotroski `NoDilution:FAIL`; "
        "ห้ามให้ BUY หากงบกำไรไม่แปลงเป็นเงินสดจริง (ปล่อยเรื่องเครดิต/ICR และ Insider Sell ให้สาย Skeptic).\n"
        "3) Round 2 Cross-Exam: ซักค้านความเห็น Growth โดยเปิดโปงส่วนต่างกำไรทางบัญชี (Non-GAAP vs GAAP/OCF) และผลกระทบ Dilution"
    ),
    # Growth Guru
    ("growth_guru", "default"): (
        "1) Priority Metrics: revenue_growth_yoy_pct, market_implied_growth_pct, eps_revision_net_30d, guidance_summary, 3Y Operating Leverage.\n"
        "2) Thresholds: BUY เมื่อ Fundamental Growth เหนือ Implied Growth พร้อม EPS Revisions เป็นบวกและมี Catalyst 6–18 เดือน.\n"
        "3) Round 2 Cross-Exam: พิสูจน์ความคุ้มค่าของ Multiple ด้วยฐานกำไรใหม่และ Operating Leverage จากข้อมูลจริง"
    ),
    ("growth_guru", "growth_disruptor"): (
        "1) Priority Metrics: market_implied_growth_pct เทียบ revenue_growth_yoy_pct / net_income_growth_yoy_pct, "
        "3-Year Historical Trajectory (Rev, EBIT Margin expansion, R&D% of Rev, capex_to_depreciation_ratio, reinvestment_lifecycle_phase), "
        "company_archetype, consensus_target_price High & YouTube/Vault Catalysts.\n"
        "2) Decision Thresholds (Red Lines): มองหาจุด Inflection ของ Operating Leverage (R&D สูงเพื่อสร้าง S-Curve แล้ว EBIT Margin เริ่มพุ่ง) "
        "และความสมมาตรของ Upside 3–5 ปี; แต่ต้องมีเหตุผลรองรับเสมอหากต้องจ่ายแพงกว่า `market_implied_growth_pct` หรือติด FLAG `implied_growth_above_upper_bound`.\n"
        "3) Round 2 Cross-Exam: โต้แย้งสาย Value ว่าการใช้ Multiple อดีตมองข้ามผลตอบแทนทบต้นจาก TAM ใหม่และ Margin Inflection"
    ),
    ("growth_guru", "growth_eps_momentum"): (
        "1) Priority Metrics: eps_revision_net_30d, eps_estimate_change_30d_pct, earnings_momentum_score, "
        "earnings_guidance_context (operating_margin_trajectory, verified_claims, Exec Quotes), ข่าวล่าสุดใน SEC:NEWS_NOTES & @Digest, "
        "และ tactical_setup (price_stage, breakout_entry_status).\n"
        "2) Decision Thresholds (Red Lines): ยึดกฎ Estimate Revision First — ต้องเห็น eps_revision_net_30d > 0 หรือ Guidance Beat พร้อม Margin ขยายตัว; "
        "หาก Revision ติดลบหรือผู้บริหารส่งสัญญาณชะลอ แม้หุ้นถูกก็ห้ามเชียร์ BUY จนกว่าจะมี Inflection รอบงบถัดไป.\n"
        "3) Round 2 Cross-Exam: ชี้ตัวเร่งระยะ 1–2 ไตรมาส (Near-Term Beat/Raise) เทียบความเสี่ยงด้านเทคนิคกับสาย Macro"
    ),
    # Macro & Tactical Guru
    ("macro_tactical_guru", "default"): (
        "1) Priority Metrics: @Macro Regime/Allocation/Sector Rotation/Thai Stance, @Digest Transmission Chain, tactical_setup, macro_yield_shock_100bps_delta.\n"
        "2) Thresholds: วาง Tranche & Stop Loss ให้สอดคล้อง Price Stage, R:R >= 2.0x, Bucket Room และ Cash Reserve.\n"
        "3) Round 2 Cross-Exam: แปลงมุมมองพื้นฐานเป็นจังหวะเข้าซื้อ/ลดไม้พร้อมระดับราคาและ Risk Trigger ที่ชัดเจน"
    ),
    ("macro_tactical_guru", "macro_dalio_allweather"): (
        "1) Priority Metrics: @Macro Regime Probabilities, Regional Macro States (Growth vs Inflation vs Monetary Score ของ US/TH/CN/EU), "
        "Asset Allocation Stance, macro_yield_shock_50bps/100bps_delta, Portfolio Bucket Actual vs Target, FX THB/USD exposure & Cash%.\n"
        "2) Decision Thresholds (Red Lines): ประเมิน Correlation & Stress Balance ก่อน — หุ้นนี้ช่วยกระจายความเสี่ยงในระบอบเศรษฐกิจปัจจุบัน "
        "หรือซ้ำเติมความเสี่ยงอัตราดอกเบี้ย/เงินเฟ้อเดิมของพอร์ต; หากพอร์ตกระจุกตัวกลุ่ม Tech/Growth เกิน Bucket Target ให้แนะ Rebalance ไปสินทรัพย์/ถังเงินที่ช่วยถ่วงดุล.\n"
        "3) Round 2 Cross-Exam: เตือนกูรูที่มองหุ้นเดี่ยวแบบสุญญากาศ ให้เห็นอิทธิพลของ Regime Shifts และ Rate Shock ต่อ Correlation ของทั้งพอร์ต"
    ),
    ("macro_tactical_guru", "macro_flow_tactician"): (
        "1) Priority Metrics: @Macro Sector Rotation & Thailand Investor Flows (Inst/Foreign/Retail), @Digest Transmission Chain, "
        "tactical_setup (STAGE_1..4, SMA50/200, buy_zone_min..max, breakout_trigger_price, invalidation_stop_loss, current_rr_ratio), "
        "smart_money_flags (institutional_ownership_pct, short_interest_pct, short_squeeze_risk).\n"
        "2) Decision Thresholds (Red Lines): ห้ามไล่ซื้อหาก R:R < 1.5x หรือ STAGE_4_MARKDOWN หลุด SMA200; ต้องระบุจุดเข้าเป็น Tranche "
        "(เช่น ไม้ 1 Pullback Buy Zone, ไม้ 2 Breakout Trigger) พร้อมราคา Stop Loss ที่ชัดเจน.\n"
        "3) Round 2 Cross-Exam: กดดันสาย Fundamental ให้รับความจริงเรื่อง Institutional Flow และจังหวะเข้าทำเพื่อลด Opportunity Cost"
    ),
    # Skeptic Guru
    ("skeptic_guru", "default"): (
        "1) Priority Metrics: DATA QUALITY FLAGS รายหุ้น, thesis_falsifiers, thesis_delta (breached_falsifiers), net_debt_ebitda, "
        "synthetic_credit_rating, interest_coverage_ratio, terminal_value_pct_of_ev.\n"
        "2) Thresholds: บังคับวาง 1–2 Measurable Kill-Switches (Metric+Condition+Threshold) ทุกเทิร์น; เตือนเมื่อ TV% > 75% หรือติด Negative FCF.\n"
        "3) Round 2 Cross-Exam: เจาะจุดอ่อนงบการเงิน/สมมติฐาน Margin ที่ฝ่าย Bull มองข้าม"
    ),
    ("skeptic_guru", "skeptic_forensic_sleuth"): (
        "1) Priority Metrics: DATA QUALITY FLAGS, synthetic_credit_rating, interest_coverage_ratio, benchmark_cost_of_debt_pct, "
        "net_debt_ebitda, current_ratio, de_ratio_pct, และ insider_conviction "
        "(open_market_s_value_usd_str vs rule_10b5_1_s_value_usd_str, amendment_unresolved).\n"
        "2) Decision Thresholds (Red Lines): จับผิดทันทีเมื่อ ICR < 3.0x, Net Debt/EBITDA > 3.0x, มี Open-Market Insider Sell ขนาดใหญ่ "
        "(นอกเหนือแผน 10b5-1), หรือติดธง `share_dilution_risk` / `10b51_unfiltered` / `quote_ohlcv_mismatch`; ทุกเทิร์นต้องจบด้วย 1–2 Kill-Switches.\n"
        "3) Round 2 Cross-Exam: ท้าทายทั้ง Value และ Growth ด้วยความเสี่ยงหนี้สิน สภาพคล่องระยะสั้น และสัญญาณเตือนจากพฤติกรรม Insider"
    ),
    ("skeptic_guru", "skeptic_taleb_tailrisk"): (
        "1) Priority Metrics: @Macro risk_scenarios (Tail Risk, Prob, Trigger, Hedge), historical mdd_pct & volatility_pct & beta, "
        "amihud_illiquidity_20d / ADTV, terminal_value_pct_of_ev, Portfolio Top-1 (>15%) / Top-3 (>50%) / Look-Through overlap, "
        "และ Portfolio Goal priority_tier='needs' & tension_level / EmergencyReserve.\n"
        "2) Decision Thresholds (Red Lines): ทดสอบ Risk of Ruin — หากความผันผวน/MDD ในอดีต (~-30%..-50%) รวมกับ Position Concentration "
        "อาจทำให้ Goal ระดับ 'needs' พังหรือบีบให้ต้องขายขาดทุนกลางวิกฤต ต้องสั่งลดขนาดไม้ทันที; เตือนเมื่อ TV% > 75% (Fragile Duration Risk) "
        "และระบุ Black-Swan Kill-Switch ทุกครั้ง.\n"
        "3) Round 2 Cross-Exam: ชี้จุดเปราะบางแบบ Asymmetric Fragility ที่คนอื่นคิดว่าความเสี่ยงต่ำ เพราะดูแค่ค่าเฉลี่ยในสภาวะปกติ"
    ),
}


def get_preset_analytical_rubric(speaker_id: str, preset_id: Optional[str]) -> str:
    """Return backend analytical rubric for a seat & preset (falls back to default if missing)."""
    clean_preset = (preset_id or "default").strip() or "default"
    return PRESET_ANALYTICAL_RUBRIC_MAP.get(
        (speaker_id, clean_preset),
        PRESET_ANALYTICAL_RUBRIC_MAP.get((speaker_id, "default"), ""),
    )


SEAT_ROLE_CONTEXT_MAP: Dict[str, Dict[str, Any]] = {
    SpeakerId.HERMES_MODERATOR.value: {
        "role_lens_label": "ประธานโต๊ะ & พอร์ตจริง",
        "focus_summary": (
            "ร้อยเรียงเหตุผลของทุกสายเทียบ @Essence 8 หมวด, เงินสด/Bucket และเป้าหมายพอร์ต "
            "เพื่อสรุปแผนแบ่งไม้และ Kill-Switch (รับบท Devil's Advocate แทนเมื่อปิด Skeptic)"
        ),
        "primary_evidence_labels": (
            "@Essence 8 หมวด & กฎเหล็ก",
            "@Portfolio สัดส่วน, เงินสด & เป้าหมาย",
            "มติ & หลักฐานทุกสายในวง",
        ),
    },
    SpeakerId.VALUE_GURU.value: {
        "role_lens_label": "มูลค่า & Margin of Safety",
        "focus_summary": (
            "ชั่งราคาตลาดเทียบ Intrinsic Value (DCF / Reverse DCF), Margin of Safety และ "
            "FCF Yield เทียบ WACC โดยไม่ไล่ราคา (รองรับ Justified PBV/DDM สำหรับแบงก์)"
        ),
        "primary_evidence_labels": (
            "DCF / Reverse DCF & MoS",
            "FCF Yield vs WACC & Credit",
            "Normalized Margin & Watchlist",
        ),
    },
    SpeakerId.QUALITY_GURU.value: {
        "role_lens_label": "คูเมือง, ROIC & คุณภาพงบ",
        "focus_summary": (
            "คัดความทนทานของ Economic Moat และคุณภาพกำไรผ่าน ROIC vs WACC, "
            "Piotroski F-Score, OCF/Net Income, Sloan Accrual และเทรนด์งบ 3 ปี"
        ),
        "primary_evidence_labels": (
            "ROIC vs WACC & Moat",
            "Piotroski F-Score & งบ 3 ปี",
            "OCF/NI, Sloan & SBC Dilution",
        ),
    },
    SpeakerId.GROWTH_GURU.value: {
        "role_lens_label": "การเติบโต & Expectation Gap",
        "focus_summary": (
            "วัด Expectation Gap ระหว่างอัตราโตจริง/Guidance กับที่ตลาด Price-in แล้ว "
            "รวมถึง EPS Revision 30 วัน, Operating Leverage และ Catalyst จาก @Digest"
        ),
        "primary_evidence_labels": (
            "Growth vs Implied Growth",
            "Guidance & EPS Revision 30d",
            "Operating Leverage & @Digest",
        ),
    },
    SpeakerId.MACRO_TACTICAL_GURU.value: {
        "role_lens_label": "มหภาค, Flow & จังหวะเข้าทำ",
        "focus_summary": (
            "อ่านระบอบ @Macro, ความอ่อนไหวดอกเบี้ย +50/100bps, กราฟเทคนิค "
            "(Price Stage, Buy Zone, R:R, Stop Loss) และ Smart Money Flow"
        ),
        "primary_evidence_labels": (
            "@Macro Regime & Sector",
            "Tactical Stage, Buy Zone & Stop",
            "Rate Shock +100bps & Flow",
        ),
    },
    SpeakerId.SKEPTIC_GURU.value: {
        "role_lens_label": "จับผิดงบ & Kill-Switch",
        "focus_summary": (
            "ทำ Pre-Mortem หาจุดแตกหักของ Thesis, Data Quality Flags, Dilution/ภาระหนี้, "
            "Insider Open-Market Sell และบังคับวาง Kill-Switch วัดผลได้ทุกเทิร์น"
        ),
        "primary_evidence_labels": (
            "DATA QUALITY FLAGS & หนี้",
            "Thesis Falsifiers & Tail Risk",
            "Insider Sell & Kill-Switch",
        ),
    },
}


# Specialized per-preset UI focus overrides (default presets inherit SEAT_ROLE_CONTEXT_MAP).
PRESET_ROLE_CONTEXT_MAP: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("hermes_moderator", "hermes_risk_chair"): {
        "role_lens_label": "คุมเพดานเสี่ยง & Drawdown",
        "focus_summary": (
            "คุมเพดานกระจุกตัว Top-1/Top-3, สภาพคล่องและ Max Drawdown เทียบ @Essence และ Goal 'needs' "
            "ก่อนอนุมัติเพิ่มน้ำหนักพร้อมกำหนด Kill-Switch เชิงตัวเลข"
        ),
        "primary_evidence_labels": (
            "สัดส่วน Top-1 / Top-3 & Overlap",
            "Max Drawdown & สภาพคล่อง (ADTV)",
            "Goal Tension & Kill-Switch วัดผล",
        ),
    },
    ("hermes_moderator", "hermes_action_coach"): {
        "role_lens_label": "โค้ชวางแผนแบ่งไม้จริง",
        "focus_summary": (
            "แปลงข้อถกเถียงของทุกสายเป็นช่วงราคาเข้าซื้อ/ลดไม้, แผน Staged Limit Order, "
            "และสัดส่วนเทียบเงินสดจริงในพอร์ตที่ทำตามได้ทันที"
        ),
        "primary_evidence_labels": (
            "โซนราคาเข้า & Watchlist Target",
            "เงินสดพร้อมใช้ & Budget รายเดือน",
            "แผนแบ่งไม้ระยะสั้น-กลาง",
        ),
    },
    ("value_guru", "value_graham_deep"): {
        "role_lens_label": "Deep Value & งบดุลแข็งแกร่ง",
        "focus_summary": (
            "คัดหุ้นราคาต่ำกว่ามูลค่าจริงแบบเข้มงวด (MoS 25–30%+) ตรวจความแข็งแกร่งสินทรัพย์หมุนเวียน, "
            "หนี้สุทธิต่ำ และเทียบ FCF Yield กับผลตอบแทนพันธบัตร 10 ปี"
        ),
        "primary_evidence_labels": (
            "MoS >= 25% & Percentile 5 ปี",
            "FCF Yield เทียบ Bond Yield 10Y",
            "Current Ratio >= 1.5x & หนี้สุทธิ",
        ),
    },
    ("value_guru", "value_klarman_margin"): {
        "role_lens_label": "Asymmetric Downside & รอ Dislocation",
        "focus_summary": (
            "คำนวณความเสี่ยงขาลงกรณี Bear DCF และ Max Drawdown ในอดีตก่อนเสมอ เทียบสภาพคล่อง "
            "และถ่วงน้ำหนักการถือเงินสดรอจังหวะตลาดปรับฐานแรง"
        ),
        "primary_evidence_labels": (
            "Bear Case & Sensitivity ขาลง",
            "Historical MDD & Normalized Margin",
            "สภาพคล่อง (ADTV/Amihud) & Cash%",
        ),
    },
    ("quality_guru", "quality_akre_compounder"): {
        "role_lens_label": "Compounder ทบต้น 5–10 ปี",
        "focus_summary": (
            "ประเมินเก้าอี้ 3 ขาของ Chuck Akre: ROIC สูงต่อเนื่อง, คูเมือง/Pricing Power "
            "และโอกาสลงทุนซ้ำระยะยาว (CapEx เทียบค่าเสื่อมและเทรนด์มาร์จิน 3 ปี)"
        ),
        "primary_evidence_labels": (
            "ROIC >= 15–20% & Moat Durability",
            "Reinvestment Runway (CapEx/D&A)",
            "เสถียรภาพ Margin & FCF 3 ปี",
        ),
    },
    ("quality_guru", "quality_forensic_auditor"): {
        "role_lens_label": "จับคุณภาพกำไร & Dilution",
        "focus_summary": (
            "แยกกำไรจริงออกจากกำไรทางบัญชี: ตรวจเงินสด OCF เทียบกำไรสุทธิ, Sloan Accrual, "
            "ส่วนต่าง GAAP vs Non-GAAP (SBC) และ Piotroski F-Score"
        ),
        "primary_evidence_labels": (
            "OCF / กำไรสุทธิ & Sloan Accrual",
            "ส่วนต่าง GAAP vs Non-GAAP (SBC)",
            "Piotroski F-Score & DuPont",
        ),
    },
    ("growth_guru", "growth_disruptor"): {
        "role_lens_label": "S-Curve & Operating Leverage",
        "focus_summary": (
            "ล่าการขยาย TAM และจุดเปลี่ยน Operating Leverage ในรอบ 3–5 ปี (งบ R&D ต่อรายได้ "
            "เทียบการขยายตัวของ EBIT Margin และ Upside แบบไม่สมมาตร)"
        ),
        "primary_evidence_labels": (
            "S-Curve & Growth vs Implied Growth",
            "Operating Leverage 3 ปี (R&D / EBIT)",
            "TAM Catalyst & Consensus High",
        ),
    },
    ("growth_guru", "growth_eps_momentum"): {
        "role_lens_label": "EPS Revision & งบ/Guidance ระยะสั้น",
        "focus_summary": (
            "ยึดสัญญาณ Estimate Revision ในรอบ 30 วัน, คำแถลงผู้บริหาร (Guidance Beat/Raise) "
            "และข่าวตัวเร่งช่วง 1–2 ไตรมาสข้างหน้า"
        ),
        "primary_evidence_labels": (
            "การปรับประมาณการ EPS 30 วัน",
            "Guidance ผู้บริหาร & Executive Quotes",
            "Catalyst ข่าวล่าสุด & @Digest",
        ),
    },
    ("macro_tactical_guru", "macro_dalio_allweather"): {
        "role_lens_label": "Macro Regime & Risk Parity",
        "focus_summary": (
            "อ่านระบอบ Growth เทียบ Inflation ของ US/TH/CN/EU และความอ่อนไหวต่อ Rate Shock +100bps "
            "เพื่อดูว่าหุ้นนี้ช่วยกระจายหรือซ้ำเติมความเสี่ยงทั้งพอร์ต"
        ),
        "primary_evidence_labels": (
            "@Macro Regime & Regional States",
            "Rate Shock +50/+100bps Impact",
            "Bucket Balance, FX THB/USD & Cash%",
        ),
    },
    ("macro_tactical_guru", "macro_flow_tactician"): {
        "role_lens_label": "Flow สถาบัน, Price Stage & จุดเข้า",
        "focus_summary": (
            "ผสาน Sector Rotation และ Fund Flow เข้ากับ Price Stage, เส้น SMA50/200, "
            "Risk:Reward (R:R), โซนเข้าแบ่งไม้ และจุดตัดขาดทุน"
        ),
        "primary_evidence_labels": (
            "Sector Rotation & Fund Flow ไทย/เทศ",
            "Price Stage, Buy Zone & R:R",
            "Short Interest & Smart Money",
        ),
    },
    ("skeptic_guru", "skeptic_forensic_sleuth"): {
        "role_lens_label": "จับผิดหนี้สิน, Flags & Insider",
        "focus_summary": (
            "สแกนความเสี่ยงงบดุล (Synthetic Credit Rating, ICR, หนี้สุทธิ/EBITDA), "
            "Data Quality Flags และยอดขายหุ้น Insider นอกแผน 10b5-1 พร้อม Kill-Switch"
        ),
        "primary_evidence_labels": (
            "Credit Rating, ICR & หนี้สุทธิ/EBITDA",
            "DATA QUALITY FLAGS & Dilution",
            "Insider Open-Market Sell & Kill-Switch",
        ),
    },
    ("skeptic_guru", "skeptic_taleb_tailrisk"): {
        "role_lens_label": "Tail-Risk & Risk of Ruin",
        "focus_summary": (
            "ทดสอบวิกฤตหางอ้วน (Black Swan): ชั่ง MDD/ความผันผวนในอดีต, สภาพคล่อง Amihud, "
            "สัดส่วนพึ่งพา Terminal Value > 75% และการกระจุกตัวที่อาจทำ Goal 'needs' พัง"
        ),
        "primary_evidence_labels": (
            "@Macro Risk Scenarios & Tail Hedge",
            "MDD อดีต, สภาพคล่อง & TV% > 75%",
            "กระจุกตัวพอร์ต & Goal 'needs' Protection",
        ),
    },
}


def get_seat_ui_context(speaker_id: str, preset_id: Optional[str] = "default") -> Dict[str, Any]:
    """Return UI-friendly seat context, primary evidence labels, and active analytical rubric."""
    clean_preset = (preset_id or "default").strip() or "default"
    base = SEAT_ROLE_CONTEXT_MAP.get(speaker_id, {})
    preset_override = PRESET_ROLE_CONTEXT_MAP.get((speaker_id, clean_preset), {})
    labels = preset_override.get("primary_evidence_labels") or base.get("primary_evidence_labels") or ()
    return {
        "role_lens_label": str(preset_override.get("role_lens_label") or base.get("role_lens_label") or ""),
        "focus_summary": str(preset_override.get("focus_summary") or base.get("focus_summary") or ""),
        "primary_evidence_labels": list(labels),
        "analytical_rubric": get_preset_analytical_rubric(speaker_id, clean_preset),
    }


@dataclass(frozen=True)
class GuruPersonaSpec:
    speaker_id: SpeakerId
    display_name: str
    short_name: str
    archetype: str
    voice_name: str
    accent_color: str
    prompt_file: str
    role_model_keywords: Tuple[str, ...]
    fact_grounding_fields: Tuple[str, ...]
    debate_order: int
    emoji: str = "🎙️"
    pitch: float = 1.0
    rate: float = 1.0
    enabled: bool = True
    preset_id: str = "default"
    speaking_style: str = ""
    custom_directive: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id.value,
            "display_name": self.display_name,
            "short_name": self.short_name,
            "archetype": self.archetype,
            "voice_name": self.voice_name,
            "accent_color": self.accent_color,
            "prompt_file": self.prompt_file,
            "role_model_keywords": list(self.role_model_keywords),
            "fact_grounding_fields": list(self.fact_grounding_fields),
            "debate_order": self.debate_order,
            "emoji": self.emoji,
            "pitch": self.pitch,
            "rate": self.rate,
            "enabled": self.enabled,
            "preset_id": self.preset_id,
            "speaking_style": self.speaking_style,
            "custom_directive": self.custom_directive,
        }


GURU_PERSONA_CATALOG: Tuple[GuruPersonaSpec, ...] = (
    GuruPersonaSpec(
        speaker_id=SpeakerId.HERMES_MODERATOR,
        display_name="Hermes (Moderator)",
        short_name="Hermes",
        archetype="Moderator & Portfolio Distiller",
        voice_name="Kore",
        accent_color="#0ea5e9",
        prompt_file="HERMES_MODERATOR.md",
        role_model_keywords=(),
        fact_grounding_fields=("all_context_cards", "confirmed_investment_axis", "portfolio_advisory"),
        debate_order=0,
        emoji="🪶",
        pitch=1.0,
        rate=1.04,
        enabled=True,
        preset_id="default",
        speaking_style="กลาง ชัดเจน ร้อยเรียงข้อถกเถียงของทุกฝ่ายให้ตกผลึกเทียบกับพอร์ตจริงของผู้ใช้",
    ),
    GuruPersonaSpec(
        speaker_id=SpeakerId.VALUE_GURU,
        display_name="Value & MoS Guru",
        short_name="Value",
        archetype="Buffett / Graham Value Investor",
        voice_name="Charon",
        accent_color="#10b981",
        prompt_file="VALUE_GURU.md",
        role_model_keywords=(
            "buffett",
            "graham",
            "klarman",
            "greenblatt",
            "damodaran",
            "value",
            "บัฟเฟตต์",
            "เกรแฮม",
            "คลาร์แมน",
            "คุณค่า",
            "แวลู",
        ),
        fact_grounding_fields=("dcf_result", "mos_entry_price_today", "fcf_yield_pct", "valuation_sanity_audit"),
        debate_order=1,
        emoji="🏛️",
        pitch=0.90,
        rate=0.96,
        enabled=True,
        preset_id="default",
        speaking_style="สุขุม รอบคอบ แบบนักลงทุนอาวุโส ชั่งน้ำหนักราคาเทียบมูลค่าที่แท้จริงเสมอ",
    ),
    GuruPersonaSpec(
        speaker_id=SpeakerId.QUALITY_GURU,
        display_name="Quality & Moat Guru",
        short_name="Quality",
        archetype="Munger / Terry Smith Compounder",
        voice_name="Fenrir",
        accent_color="#6366f1",
        prompt_file="QUALITY_GURU.md",
        role_model_keywords=(
            "munger",
            "terry smith",
            "smith",
            "pabrai",
            "akre",
            "spier",
            "quality",
            "moat",
            "มังเจอร์",
            "เอเคอร์",
            "เทอร์รี สมิธ",
            "คุณภาพ",
        ),
        fact_grounding_fields=("roic_pct", "piotroski_f_score", "ocf_to_net_income", "sloan_accrual", "dupont_breakdown"),
        debate_order=2,
        emoji="💎",
        pitch=0.95,
        rate=1.0,
        enabled=True,
        preset_id="default",
        speaking_style="เฉียบคม มองทะลุโครงสร้างอุตสาหกรรม ROIC, Pricing Power และความยั่งยืนของคูเมือง",
    ),
    GuruPersonaSpec(
        speaker_id=SpeakerId.GROWTH_GURU,
        display_name="Growth & Expectations Guru",
        short_name="Growth",
        archetype="Peter Lynch / Fisher Growth",
        voice_name="Puck",
        accent_color="#f59e0b",
        prompt_file="GROWTH_GURU.md",
        role_model_keywords=(
            "lynch",
            "fisher",
            "baillie",
            "wood",
            "coleman",
            "roehl",
            "growth",
            "ลินช์",
            "ฟิชเชอร์",
            "เติบโต",
            "การเติบโต",
        ),
        fact_grounding_fields=("reverse_dcf", "market_implied_growth_pct", "earnings_guidance_context", "eps_revision_net_30d"),
        debate_order=3,
        emoji="🚀",
        pitch=1.08,
        rate=1.06,
        enabled=True,
        preset_id="default",
        speaking_style="มีพลัง มองหาอัตราโตที่ตลาดยังประเมินไม่ครบ เทียบ Implied Growth กับศักยภาพจริง",
    ),
    GuruPersonaSpec(
        speaker_id=SpeakerId.MACRO_TACTICAL_GURU,
        display_name="Macro & Tactical Guru",
        short_name="Macro",
        archetype="Howard Marks / Druckenmiller Cycle & Flow",
        voice_name="Aoede",
        accent_color="#ec4899",
        prompt_file="MACRO_TACTICAL_GURU.md",
        role_model_keywords=(
            "dalio",
            "druckenmiller",
            "marks",
            "soros",
            "macro",
            "tactical",
            "ดาลิโอ",
            "ดรัคเคนมิลเลอร์",
            "ดรัค",
            "มหภาค",
        ),
        fact_grounding_fields=("macro_strategy_direction", "sensitivity_matrix", "tactical_setup", "smart_money_flags"),
        debate_order=4,
        emoji="🌊",
        pitch=1.04,
        rate=1.02,
        enabled=True,
        preset_id="default",
        speaking_style="อ่านเข็มนาฬิกาวัฏจักรตลาด สภาพคล่อง เงินเฟ้อ และจังหวะการจัดสรรเงินอย่างยืดหยุ่น",
    ),
    GuruPersonaSpec(
        speaker_id=SpeakerId.SKEPTIC_GURU,
        display_name="Skeptic / Devil's Advocate",
        short_name="Skeptic",
        archetype="Chanos / Burry Risk & Falsification",
        voice_name="Orus",
        accent_color="#ef4444",
        prompt_file="SKEPTIC_GURU.md",
        role_model_keywords=(
            "chanos",
            "burry",
            "einhorn",
            "taleb",
            "short",
            "skeptic",
            "bear",
            "tail risk",
            "forensic",
            "เบอร์รี่",
            "ทาเล็บ",
            "ความเสี่ยง",
        ),
        fact_grounding_fields=("data_quality_flags", "thesis_falsifiers", "breached_falsifiers", "tail_risks", "net_debt_ebitda"),
        debate_order=5,
        emoji="🛡️",
        pitch=0.86,
        rate=0.98,
        enabled=True,
        preset_id="default",
        speaking_style="เข้ม ตรงไปตรงมา ตั้งคำถามทวนกระแสเพื่อตรวจหาจุดอ่อนและสมมติฐานที่เปราะบาง",
    ),
)

PERSONA_BY_ID: Dict[SpeakerId, GuruPersonaSpec] = {spec.speaker_id: spec for spec in GURU_PERSONA_CATALOG}

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RESERVED_PROMPT_TAG_RE = re.compile(
    r"(?i)\[(?:NO_EXTERNAL_TOOLS|INSTRUMENT IDENTITY ANCHOR|MANDATORY DEVIL|PORTFOLIO_STATE|ROUNDTABLE CONSISTENCY VERIFIER|CUSTOM GURU PERSONA|STRICT FACT-GROUNDING)[^\]]*\]"
)


def sanitize_custom_persona_text(raw: Optional[str], max_len: int) -> str:
    """Strip control bytes and reserved prompt tags from user-customized persona strings."""
    text = str(raw or "")
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _RESERVED_PROMPT_TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def merge_persona_overrides(
    base_catalog: Sequence[GuruPersonaSpec] = GURU_PERSONA_CATALOG,
    saved_profiles: Optional[Sequence[Dict[str, Any]]] = None,
    session_overrides: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[GuruPersonaSpec, ...]:
    """Deterministically merge base personas -> DB saved profiles -> per-session overrides.

    Invariants enforced:
    - All 6 core seats are always returned, sorted by `(debate_order, default_order)`.
    - `hermes_moderator` stays forced `enabled=True` at `debate_order=0`.
    - At least 2 non-Hermes panel gurus must remain `enabled=True` (raises `ValueError` otherwise).
    - `voice_name` must exist in `GEMINI_VOICE_CATALOG`, and `pitch`/`rate` are clamped cleanly.
    """
    saved_map: Dict[str, Dict[str, Any]] = {
        str(row.get("speaker_id", "")): dict(row)
        for row in (saved_profiles or ())
        if isinstance(row, dict) and row.get("speaker_id")
    }
    session_map: Dict[str, Dict[str, Any]] = {}
    valid_ids = {s.speaker_id.value for s in base_catalog}
    for row in session_overrides or ():
        if not isinstance(row, dict):
            continue
        sid = str(row.get("speaker_id") or "")
        if sid not in valid_ids:
            raise ValueError(f"Unknown guru speaker_id: '{sid}'")
        session_map[sid] = dict(row)

    merged: List[Tuple[int, int, GuruPersonaSpec]] = []
    for default_idx, base in enumerate(base_catalog):
        sid = base.speaker_id.value
        patch: Dict[str, Any] = {}
        if sid in saved_map:
            patch.update({k: v for k, v in saved_map[sid].items() if v is not None})
        if sid in session_map:
            patch.update({k: v for k, v in session_map[sid].items() if v is not None})

        display_name = sanitize_custom_persona_text(patch.get("display_name", base.display_name), 60) or base.display_name
        short_name = sanitize_custom_persona_text(patch.get("short_name", base.short_name), 24) or base.short_name
        archetype = sanitize_custom_persona_text(patch.get("archetype", base.archetype), 80) or base.archetype
        speaking_style = sanitize_custom_persona_text(patch.get("speaking_style", base.speaking_style), 200)
        custom_directive = sanitize_custom_persona_text(patch.get("custom_directive", base.custom_directive), 500)
        preset_id = sanitize_custom_persona_text(patch.get("preset_id", base.preset_id), 48) or "default"
        emoji = sanitize_custom_persona_text(patch.get("emoji", base.emoji), 8) or base.emoji

        raw_color = str(patch.get("accent_color", base.accent_color) or "").strip()
        accent_color = raw_color if _HEX_COLOR_RE.match(raw_color) else base.accent_color

        raw_voice = str(patch.get("voice_name", base.voice_name) or "").strip()
        if raw_voice in ALLOWED_VOICE_NAMES:
            voice_name = raw_voice
        elif sid in session_map and "voice_name" in session_map[sid]:
            raise ValueError(f"Unsupported voice_name '{raw_voice}' for {sid}")
        else:
            voice_name = base.voice_name

        try:
            pitch = float(patch.get("pitch", base.pitch))
        except (TypeError, ValueError):
            pitch = base.pitch
        pitch = round(min(1.25, max(0.75, pitch)), 2)

        try:
            rate = float(patch.get("rate", base.rate))
        except (TypeError, ValueError):
            rate = base.rate
        rate = round(min(1.25, max(0.80, rate)), 2)

        if base.speaker_id == SpeakerId.HERMES_MODERATOR:
            enabled = True
            debate_order = 0
        else:
            enabled = bool(patch.get("enabled", base.enabled))
            try:
                debate_order = int(patch.get("debate_order", base.debate_order))
            except (TypeError, ValueError):
                debate_order = base.debate_order
            debate_order = max(1, min(99, debate_order))

        spec = GuruPersonaSpec(
            speaker_id=base.speaker_id,
            display_name=display_name,
            short_name=short_name,
            archetype=archetype,
            voice_name=voice_name,
            accent_color=accent_color,
            prompt_file=base.prompt_file,
            role_model_keywords=base.role_model_keywords,
            fact_grounding_fields=base.fact_grounding_fields,
            debate_order=debate_order,
            emoji=emoji,
            pitch=pitch,
            rate=rate,
            enabled=enabled,
            preset_id=preset_id,
            speaking_style=speaking_style,
            custom_directive=custom_directive,
        )
        has_session_order = sid in session_map and session_map[sid].get("debate_order") is not None
        has_saved_order = sid in saved_map and saved_map[sid].get("debate_order") is not None
        order_tier = 0 if has_session_order else (1 if has_saved_order else 2)
        merged.append(
            (
                0 if base.speaker_id == SpeakerId.HERMES_MODERATOR else 1,
                (debate_order, order_tier, default_idx),
                spec,
            )
        )

    merged.sort(key=lambda item: (item[0], item[1]))
    final_specs: List[GuruPersonaSpec] = []
    panel_seq = 1
    for _, _, spec in merged:
        if spec.speaker_id == SpeakerId.HERMES_MODERATOR:
            final_specs.append(spec)
        else:
            final_specs.append(
                GuruPersonaSpec(
                    speaker_id=spec.speaker_id,
                    display_name=spec.display_name,
                    short_name=spec.short_name,
                    archetype=spec.archetype,
                    voice_name=spec.voice_name,
                    accent_color=spec.accent_color,
                    prompt_file=spec.prompt_file,
                    role_model_keywords=spec.role_model_keywords,
                    fact_grounding_fields=spec.fact_grounding_fields,
                    debate_order=panel_seq,
                    emoji=spec.emoji,
                    pitch=spec.pitch,
                    rate=spec.rate,
                    enabled=spec.enabled,
                    preset_id=spec.preset_id,
                    speaking_style=spec.speaking_style,
                    custom_directive=spec.custom_directive,
                )
            )
            panel_seq += 1

    active_panel_count = sum(
        1 for spec in final_specs if spec.speaker_id != SpeakerId.HERMES_MODERATOR and spec.enabled
    )
    if active_panel_count < 2:
        raise ValueError("ต้องเปิดใช้งานกูรูเข้าร่วมประชุมอย่างน้อย 2 คน (ไม่รวม Hermes Moderator)")

    return tuple(final_specs)


@dataclass(frozen=True)
class PastRoundtableScorecard:
    previous_session_id: str
    previous_date: str
    primary_ticker: str
    benchmark_ticker: str
    entry_price: Optional[str] = None
    settlement_price: Optional[str] = None
    raw_return_pct: Optional[str] = None
    benchmark_return_pct: Optional[str] = None
    alpha_pct: Optional[str] = None
    previous_final_rating: str = "REVIEW"
    outcome_label: str = "pending"  # "outperformed" | "lagged" | "neutral" | "pending" | "lookahead_skipped"
    reflection_lesson: Optional[str] = None
    settlement_status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["PastRoundtableScorecard"]:
        if not data:
            return None
        return cls(
            previous_session_id=str(data.get("previous_session_id", "")),
            previous_date=str(data.get("previous_date", "")),
            primary_ticker=str(data.get("primary_ticker", "")),
            benchmark_ticker=str(data.get("benchmark_ticker", "")),
            entry_price=data.get("entry_price"),
            settlement_price=data.get("settlement_price"),
            raw_return_pct=data.get("raw_return_pct"),
            benchmark_return_pct=data.get("benchmark_return_pct"),
            alpha_pct=data.get("alpha_pct"),
            previous_final_rating=str(data.get("previous_final_rating", "REVIEW")),
            outcome_label=str(data.get("outcome_label", "pending")),
            reflection_lesson=data.get("reflection_lesson"),
            settlement_status=str(data.get("settlement_status", "pending")),
        )


@dataclass(frozen=True)
class InstrumentIdentityAnchor:
    ticker: str
    company_name: str
    market: str  # "US" | "TH" | "GLOBAL"
    sector: str
    industry: str
    currency: str
    benchmark_ticker: str
    current_price: Optional[str] = None
    verified: bool = True

    def to_prompt_block(self) -> str:
        price_str = f"{self.current_price} {self.currency}" if self.current_price else "N/A"
        return (
            f"[INSTRUMENT IDENTITY ANCHOR — VERIFIED]\n"
            f"- Ticker: {self.ticker} | Market: {self.market} | Benchmark: {self.benchmark_ticker}\n"
            f"- Company: {self.company_name}\n"
            f"- Sector / Industry: {self.sector} / {self.industry}\n"
            f"- Currency: {self.currency} | Reference Price: {price_str}\n"
            f"(Do NOT confuse this company with any other ticker or homonym.)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["InstrumentIdentityAnchor"]:
        if not data:
            return None
        return cls(
            ticker=str(data.get("ticker", "")),
            company_name=str(data.get("company_name", "")),
            market=str(data.get("market", "US")),
            sector=str(data.get("sector", "Unknown")),
            industry=str(data.get("industry", "Unknown")),
            currency=str(data.get("currency", "USD")),
            benchmark_ticker=str(data.get("benchmark_ticker", "SPY")),
            current_price=data.get("current_price"),
            verified=bool(data.get("verified", True)),
        )


@dataclass(frozen=True)
class PortfolioHoldingGuardState:
    attachment_state: ContextAttachmentState = ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME
    portfolio_attached: bool = False
    is_held: bool = False
    units: Optional[str] = None
    avg_cost: Optional[str] = None
    current_weight_pct: Optional[str] = None
    unrealized_pnl_pct: Optional[str] = None
    bucket_id: Optional[str] = None
    portfolio_total_nav_thb: Optional[str] = None
    portfolio_cash_pct: Optional[str] = None
    state_hash: str = ""
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attachment_state"] = self.attachment_state.value
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PortfolioHoldingGuardState":
        if not data:
            return cls()
        raw_state = str(data.get("attachment_state", ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME.value))
        try:
            state_enum = ContextAttachmentState(raw_state)
        except ValueError:
            state_enum = ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME
        return cls(
            attachment_state=state_enum,
            portfolio_attached=bool(data.get("portfolio_attached", False)),
            is_held=bool(data.get("is_held", False)),
            units=data.get("units"),
            avg_cost=data.get("avg_cost"),
            current_weight_pct=data.get("current_weight_pct"),
            unrealized_pnl_pct=data.get("unrealized_pnl_pct"),
            bucket_id=data.get("bucket_id"),
            portfolio_total_nav_thb=data.get("portfolio_total_nav_thb"),
            portfolio_cash_pct=data.get("portfolio_cash_pct"),
            state_hash=str(data.get("state_hash", "")),
            summary_text=str(data.get("summary_text", "")),
        )


@dataclass(frozen=True)
class FrozenContextPacket:
    session_id: str
    created_at_iso: str
    agenda_prompt: str
    context_refs: Tuple[str, ...] = ()
    primary_ticker: Optional[str] = None
    instrument_anchor: Optional[InstrumentIdentityAnchor] = None
    missing_prerequisite_tickers: Tuple[str, ...] = ()
    portfolio_guard: PortfolioHoldingGuardState = field(default_factory=PortfolioHoldingGuardState)
    equity_report_text: str = ""
    equity_data: Dict[str, Any] = field(default_factory=dict)
    macro_report_text: str = ""
    digest_report_text: str = ""
    essence_report_text: str = ""
    user_role_models: Tuple[str, ...] = ()
    prioritized_speakers: Tuple[str, ...] = ()
    confirmed_axis_summary: Dict[str, Any] = field(default_factory=dict)
    past_scorecard: Optional[PastRoundtableScorecard] = None
    data_quality_flags: Tuple[str, ...] = ()
    action_stance_hint: Optional[str] = None
    verified_numeric_facts: Tuple[str, ...] = ()
    active_personas: Tuple[Dict[str, Any], ...] = ()
    packet_fingerprint_sha256: str = ""
    secondary_anchors: Tuple[InstrumentIdentityAnchor, ...] = ()
    secondary_scorecards: Tuple[PastRoundtableScorecard, ...] = ()
    ticker_data_quality_flags: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    portfolio_id: str = "default"
    debate_intensity: str = "balanced"
    debate_depth: str = "concise"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at_iso": self.created_at_iso,
            "agenda_prompt": self.agenda_prompt,
            "context_refs": list(self.context_refs),
            "primary_ticker": self.primary_ticker,
            "instrument_anchor": self.instrument_anchor.to_dict() if self.instrument_anchor else None,
            "missing_prerequisite_tickers": list(self.missing_prerequisite_tickers),
            "portfolio_guard": self.portfolio_guard.to_dict(),
            "equity_report_text": self.equity_report_text,
            "equity_data": self.equity_data,
            "macro_report_text": self.macro_report_text,
            "digest_report_text": self.digest_report_text,
            "essence_report_text": self.essence_report_text,
            "user_role_models": list(self.user_role_models),
            "prioritized_speakers": list(self.prioritized_speakers),
            "confirmed_axis_summary": self.confirmed_axis_summary,
            "past_scorecard": self.past_scorecard.to_dict() if self.past_scorecard else None,
            "data_quality_flags": list(self.data_quality_flags),
            "action_stance_hint": self.action_stance_hint,
            "verified_numeric_facts": list(self.verified_numeric_facts),
            "active_personas": list(self.active_personas),
            "packet_fingerprint_sha256": self.packet_fingerprint_sha256,
            "secondary_anchors": [a.to_dict() for a in self.secondary_anchors],
            "secondary_scorecards": [s.to_dict() for s in self.secondary_scorecards],
            "ticker_data_quality_flags": {
                str(k): list(v) for k, v in (self.ticker_data_quality_flags or {}).items()
            },
            "portfolio_id": self.portfolio_id,
            "debate_intensity": self.debate_intensity,
            "debate_depth": self.debate_depth,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FrozenContextPacket":
        if not data:
            return cls(session_id="", created_at_iso="", agenda_prompt="")
        raw_sec_anchors = [
            InstrumentIdentityAnchor.from_dict(a)
            for a in (data.get("secondary_anchors") or ())
            if isinstance(a, dict)
        ]
        raw_sec_scorecards = [
            PastRoundtableScorecard.from_dict(s)
            for s in (data.get("secondary_scorecards") or ())
            if isinstance(s, dict)
        ]
        raw_ticker_dq = data.get("ticker_data_quality_flags") or {}
        parsed_ticker_dq: Dict[str, Tuple[str, ...]] = (
            {
                str(k).strip().upper(): tuple(str(f) for f in (v or ()) if f)
                for k, v in raw_ticker_dq.items()
                if k
            }
            if isinstance(raw_ticker_dq, dict)
            else {}
        )
        return cls(
            session_id=str(data.get("session_id", "")),
            created_at_iso=str(data.get("created_at_iso", "")),
            agenda_prompt=str(data.get("agenda_prompt", "")),
            context_refs=tuple(data.get("context_refs") or ()),
            primary_ticker=data.get("primary_ticker"),
            instrument_anchor=InstrumentIdentityAnchor.from_dict(data.get("instrument_anchor")),
            missing_prerequisite_tickers=tuple(data.get("missing_prerequisite_tickers") or ()),
            portfolio_guard=PortfolioHoldingGuardState.from_dict(data.get("portfolio_guard")),
            equity_report_text=str(data.get("equity_report_text", "")),
            equity_data=dict(data.get("equity_data") or {}),
            macro_report_text=str(data.get("macro_report_text", "")),
            digest_report_text=str(data.get("digest_report_text", "")),
            essence_report_text=str(data.get("essence_report_text", "")),
            user_role_models=tuple(data.get("user_role_models") or ()),
            prioritized_speakers=tuple(data.get("prioritized_speakers") or ()),
            confirmed_axis_summary=dict(data.get("confirmed_axis_summary") or {}),
            past_scorecard=PastRoundtableScorecard.from_dict(data.get("past_scorecard")),
            data_quality_flags=tuple(data.get("data_quality_flags") or ()),
            action_stance_hint=data.get("action_stance_hint"),
            verified_numeric_facts=tuple(data.get("verified_numeric_facts") or ()),
            active_personas=tuple(data.get("active_personas") or ()),
            packet_fingerprint_sha256=str(data.get("packet_fingerprint_sha256", "")),
            secondary_anchors=tuple(a for a in raw_sec_anchors if a is not None),
            secondary_scorecards=tuple(s for s in raw_sec_scorecards if s is not None),
            ticker_data_quality_flags=parsed_ticker_dq,
            portfolio_id=str(data.get("portfolio_id") or "default"),
            debate_intensity=normalize_debate_intensity(data.get("debate_intensity"), default="balanced"),
            debate_depth=normalize_debate_depth(data.get("debate_depth"), default="concise"),
        )


def resolve_personas_from_packet(packet: Optional[FrozenContextPacket]) -> Tuple[GuruPersonaSpec, ...]:
    """Resolve effective guru specs from packet.active_personas with safe legacy fallback."""
    if packet is None or not packet.active_personas:
        return GURU_PERSONA_CATALOG
    try:
        return merge_persona_overrides(GURU_PERSONA_CATALOG, packet.active_personas, None)
    except Exception:
        return GURU_PERSONA_CATALOG


@dataclass(frozen=True)
class RoundtableSession:
    session_id: str
    title: str
    primary_ticker: Optional[str]
    benchmark_ticker: Optional[str]
    entry_price_at_session: Optional[str]
    agenda_prompt: str
    context_refs: Tuple[str, ...]
    frozen_packet_json: str
    packet_fingerprint_sha256: str
    decision_playbook_json: Optional[str]
    final_rating: str
    settlement_status: str
    settled_return_pct: Optional[str]
    settled_alpha_pct: Optional[str]
    settled_at_date: Optional[str]
    reflection_lesson: Optional[str]
    status: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class RoundtableTurn:
    turn_id: str
    session_id: str
    seq: int
    speaker_id: str
    round_stage: str
    replying_to: Optional[str]
    spoken_dialogue: str
    evidence_bullets: Tuple[str, ...]
    stance_tag: str
    audio_cache_key: Optional[str]
    created_at: float
    sizing_posture: Optional[str] = None


def match_prioritized_speakers(role_models: Sequence[str]) -> Tuple[str, ...]:
    """Deterministically map user ConfirmedInvestmentAxis role_models to speaker_ids."""
    if not role_models:
        return ()
    joined = " ".join(str(r).lower() for r in role_models if r)
    matched: List[str] = []
    for spec in GURU_PERSONA_CATALOG:
        if spec.speaker_id == SpeakerId.HERMES_MODERATOR:
            continue
        if any(kw in joined for kw in spec.role_model_keywords):
            matched.append(spec.speaker_id.value)
    return tuple(matched)


_CANONICAL_STANCE_VALUES = {s.value for s in RoundtableStance}


def normalize_5tier_stance(raw: str) -> str:
    """Normalize arbitrary rating text to canonical 5-tier + REVIEW sentinel."""
    text = (str(raw) if raw is not None else "").strip().upper()
    if text in _CANONICAL_STANCE_VALUES:
        return text
    if "STRONG BUY" in text or text == "BUY":
        return RoundtableStance.BUY.value
    if "OVERWEIGHT" in text or "ACCUMULATE" in text or "BULL" in text:
        return RoundtableStance.OVERWEIGHT.value
    if "UNDERWEIGHT" in text or "TRIM" in text or "REDUCE" in text:
        return RoundtableStance.UNDERWEIGHT.value
    if "STRONG SELL" in text or "SELL" in text or "BEAR" in text:
        return RoundtableStance.SELL.value
    if "HOLD" in text or "NEUTRAL" in text:
        return RoundtableStance.HOLD.value
    return RoundtableStance.REVIEW.value


_ALLOWED_DEBATE_INTENSITIES = {d.value for d in DebateIntensity}
_ALLOWED_DEBATE_DEPTHS = {d.value for d in DebateDepth}


def normalize_debate_intensity(raw: Any, *, default: str = DebateIntensity.CROSSFIRE.value) -> str:
    val = str(raw or "").strip().lower()
    if val in _ALLOWED_DEBATE_INTENSITIES:
        return val
    return default if default in _ALLOWED_DEBATE_INTENSITIES else DebateIntensity.CROSSFIRE.value


def normalize_debate_depth(raw: Any, *, default: str = DebateDepth.DEEP_YOUTUBE.value) -> str:
    val = str(raw or "").strip().lower()
    if val in _ALLOWED_DEBATE_DEPTHS:
        return val
    return default if default in _ALLOWED_DEBATE_DEPTHS else DebateDepth.DEEP_YOUTUBE.value


_STANCE_POLARITY: Dict[str, int] = {
    "BUY": 2,
    "OVERWEIGHT": 1,
    "HOLD": 0,
    "REVIEW": 0,
    "UNDERWEIGHT": -1,
    "SELL": -2,
}

_PILLAR_KEYWORD_MAP: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Valuation & MOS",
        (
            "mos",
            "intrinsic",
            "dcf",
            "reverse dcf",
            "p/e",
            "pe=",
            "multiple",
            "ev/ebitda",
            "upside",
            "implied growth",
            "valuation",
            "มูลค่า",
            "ส่วนเผื่อ",
            "ราคาแพง",
            "ราคาถูก",
        ),
    ),
    (
        "Quality, Moat & Capital",
        (
            "roic",
            "wacc",
            "moat",
            "margin",
            "f-score",
            "f_score",
            "ocf",
            "sloan",
            "sbc",
            "dilution",
            "buyback",
            "roe",
            "คุณภาพ",
            "คูเมือง",
            "กำไร",
            "อัตรากำไร",
        ),
    ),
    (
        "Growth & Catalysts",
        (
            "revenue",
            "guidance",
            "eps",
            "revision",
            "cagr",
            "catalyst",
            "transcript",
            "operating leverage",
            "เติบโต",
            "รายได้",
            "ตัวเร่ง",
            "ประมาณการ",
            "คาดการณ์",
        ),
    ),
    (
        "Macro, Cycle & Portfolio",
        (
            "macro",
            "fed",
            "yield",
            "rate",
            "fx",
            "usd/thb",
            "stage_",
            "sma",
            "stop loss",
            "r:r",
            "cash",
            "bucket",
            "look-through",
            "concentration",
            "goals",
            "มหภาค",
            "ดอกเบี้ย",
            "พอร์ต",
            "เงินสด",
            "ถังเงิน",
        ),
    ),
    (
        "Downside, Credit & Falsifiers",
        (
            "downside",
            "debt",
            "ebitda",
            "icr",
            "credit",
            "mdd",
            "beta",
            "insider",
            "falsifier",
            "kill-switch",
            "kill switch",
            "tail risk",
            "stale",
            "honest gap",
            "red flag",
            "ความเสี่ยง",
            "หนี้สิน",
            "จุดหักล้าง",
            "จุดตาย",
        ),
    ),
)


def _turn_field(turn: Any, key: str, alt: str = "") -> Any:
    if isinstance(turn, dict):
        return turn.get(key) if key in turn else turn.get(alt)
    val = getattr(turn, key, None)
    if val is None and alt:
        val = getattr(turn, alt, None)
    return val


def analyze_debate_pillar_coverage(turns: Sequence[Any]) -> Dict[str, Any]:
    """Track which core investment pillars have been raised and challenged across turns."""
    raised_by: Dict[str, List[str]] = {p: [] for p, _ in _PILLAR_KEYWORD_MAP}
    challenged_by: Dict[str, List[str]] = {p: [] for p, _ in _PILLAR_KEYWORD_MAP}

    for turn in turns or ():
        spk = str(_turn_field(turn, "speaker_id") or "").strip()
        if not spk or spk in (SpeakerId.USER.value, SpeakerId.HERMES_MODERATOR.value):
            continue
        stage = str(_turn_field(turn, "round_stage") or "").strip().lower()
        reply_to = str(_turn_field(turn, "replying_to_speaker_id", "replying_to") or "").strip()
        dialogue = str(_turn_field(turn, "spoken_dialogue") or "")
        bullets = _turn_field(turn, "evidence_bullets") or ()
        haystack = (dialogue + " " + " ".join(str(b) for b in bullets)).lower()

        is_challenge_turn = stage in (RoundStage.REBUTTAL.value, RoundStage.CROSSFIRE.value) and bool(reply_to)
        for pillar, keywords in _PILLAR_KEYWORD_MAP:
            if any(kw in haystack for kw in keywords):
                if spk not in raised_by[pillar]:
                    raised_by[pillar].append(spk)
                if is_challenge_turn and spk not in challenged_by[pillar]:
                    challenged_by[pillar].append(spk)

    unchallenged = [
        pillar
        for pillar, _ in _PILLAR_KEYWORD_MAP
        if not challenged_by[pillar]
    ]
    return {
        "raised_by": raised_by,
        "challenged_by": challenged_by,
        "unchallenged_pillars": unchallenged,
    }


def select_crossfire_duels(
    turns: Sequence[Any],
    ordered_gurus: Sequence[str],
    *,
    max_duels: int = 2,
) -> List[Tuple[str, str]]:
    """Deterministically schedule Round 2.5 Crossfire duel turns [(speaker_id, target_speaker_id), ...].

    Prioritizes:
    1. Highest stance-polarity conflict (Bull vs Bear / Cautious).
    2. Explicit Round 2 challenges (giving the targeted/earlier speaker Right of Reply first!).
    3. Distinct speakers across Duel #1 and Duel #2 when >= 4 panel gurus are enabled.
    """
    panel = [str(g) for g in ordered_gurus if g and g not in (SpeakerId.HERMES_MODERATOR.value, SpeakerId.USER.value)]
    if len(panel) < 2 or max_duels <= 0:
        return []

    latest_stance: Dict[str, str] = {g: "HOLD" for g in panel}
    latest_seq: Dict[str, int] = {g: idx for idx, g in enumerate(panel)}
    attacks: set[Tuple[str, str]] = set()

    for idx, turn in enumerate(turns or ()):
        spk = str(_turn_field(turn, "speaker_id") or "").strip()
        if spk not in latest_stance:
            continue
        latest_stance[spk] = normalize_5tier_stance(str(_turn_field(turn, "stance_tag") or "HOLD"))
        latest_seq[spk] = idx
        stage = str(_turn_field(turn, "round_stage") or "").strip().lower()
        target = str(_turn_field(turn, "replying_to_speaker_id", "replying_to") or "").strip()
        if stage == RoundStage.REBUTTAL.value and target in latest_stance and target != spk:
            attacks.add((spk, target))

    polarity = {g: _STANCE_POLARITY.get(latest_stance[g], 0) for g in panel}

    def _pair_score(a: str, b: str) -> Tuple[int, int]:
        diff = abs(polarity[a] - polarity[b]) * 10
        attack_bonus = 6 if ((a, b) in attacks or (b, a) in attacks) else 0
        cross_camp = 4 if ((polarity[a] > 0 and polarity[b] <= 0) or (polarity[b] > 0 and polarity[a] <= 0)) else 0
        # Break complete ties toward opposing archetypes (earlier vs later seat)
        seat_gap = abs(panel.index(a) - panel.index(b))
        return (diff + attack_bonus + cross_camp, seat_gap)

    candidates: List[Tuple[Tuple[int, int], str, str]] = []
    for i in range(len(panel)):
        for j in range(i + 1, len(panel)):
            g1, g2 = panel[i], panel[j]
            candidates.append((_pair_score(g1, g2), g1, g2))
    candidates.sort(key=lambda item: item[0], reverse=True)

    schedule: List[Tuple[str, str]] = []
    used: set[str] = set()

    for _, g1, g2 in candidates:
        if len(schedule) >= max_duels * 2:
            break
        if len(panel) >= 4 and (g1 in used or g2 in used):
            continue
        if len(panel) == 3 and schedule and (g1 in used and g2 in used):
            continue
        if (g1, g2) in schedule and (g2, g1) in schedule:
            continue

        # Order speaker 1: give Right of Reply first to whichever speaker was attacked by the other
        # or spoke earlier in Round 2 (e.g., Value Guru speaking early gets immediate rebuttal chance).
        if (g2, g1) in attacks and (g1, g2) not in attacks:
            first, second = g1, g2
        elif (g1, g2) in attacks and (g2, g1) not in attacks:
            first, second = g2, g1
        elif latest_seq.get(g1, 0) <= latest_seq.get(g2, 0):
            first, second = g1, g2
        else:
            first, second = g2, g1

        schedule.append((first, second))
        schedule.append((second, first))
        used.add(first)
        used.add(second)

    # If only 2 gurus exist and max_duels == 2, allow a second back-and-forth exchange cleanly
    if len(panel) == 2 and max_duels == 2 and len(schedule) == 2:
        first, second = schedule[0]
        schedule.append((first, second))
        schedule.append((second, first))

    return schedule[: max_duels * 2]


_CANONICAL_PANEL_SEQUENCE: Tuple[str, ...] = (
    SpeakerId.VALUE_GURU.value,
    SpeakerId.QUALITY_GURU.value,
    SpeakerId.GROWTH_GURU.value,
    SpeakerId.MACRO_TACTICAL_GURU.value,
    SpeakerId.SKEPTIC_GURU.value,
)

_AGENDA_LEAD_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        SpeakerId.VALUE_GURU.value,
        (
            "มูลค่า",
            "margin of safety",
            "dcf",
            "mos",
            "ราคาถูก",
            "ราคาแพง",
            "เข้าซื้อตรงไหนดี",
            "fair value",
            "intrinsic",
            "p/e",
        ),
    ),
    (
        SpeakerId.MACRO_TACTICAL_GURU.value,
        (
            "จังหวะ",
            "แนวรับ",
            "แนวต้าน",
            "กราฟ",
            "เทคนิค",
            "ดอกเบี้ย",
            "เฟด",
            "fed",
            "yield",
            "มหภาค",
            "macro",
            "flow",
            "usd/thb",
            "stop loss",
            "breakout",
            "tactical",
            "stage",
        ),
    ),
    (
        SpeakerId.SKEPTIC_GURU.value,
        (
            "ความเสี่ยง",
            "จุดตาย",
            "จับผิด",
            "หนี้สิน",
            "turnaround",
            "ฟองสบู่",
            "bubble",
            "overvalued",
            "credit",
            "dilution",
            "insider",
            "falsifier",
            "kill-switch",
            "kill switch",
            "red flag",
            "ขายหนี",
        ),
    ),
    (
        SpeakerId.GROWTH_GURU.value,
        (
            "เติบโต",
            "growth",
            "guidance",
            "catalyst",
            "s-curve",
            "tam",
            "eps",
            "revision",
            "backlog",
            "operating leverage",
            "ai",
            "รายได้โต",
        ),
    ),
    (
        SpeakerId.QUALITY_GURU.value,
        (
            "moat",
            "roic",
            "คูเมือง",
            "compounder",
            "คุณภาพกิจการ",
            "คุณภาพกำไร",
            "f-score",
            "sloan",
            "pricing power",
            "ทบต้น",
        ),
    ),
)


def suggest_opening_order_for_agenda(
    agenda_prompt: str,
    personas: Sequence[GuruPersonaSpec],
) -> List[str]:
    """Return speaking order for enabled panel seats, promoting an agenda-matched lead only on default order."""
    ordered_specs = [
        spec
        for spec in sorted(personas, key=lambda s: s.debate_order)
        if spec.speaker_id != SpeakerId.HERMES_MODERATOR and spec.enabled
    ]
    ordered_ids = [spec.speaker_id.value for spec in ordered_specs]
    if len(ordered_ids) <= 1:
        return ordered_ids

    # Respect any user-customized debate_order (relative order differs from canonical sequence)
    expected_canonical = [sid for sid in _CANONICAL_PANEL_SEQUENCE if sid in ordered_ids]
    if ordered_ids != expected_canonical:
        return ordered_ids

    text = (agenda_prompt or "").strip().lower()
    if not text:
        return ordered_ids

    scores: Dict[str, int] = {}
    for sid, keywords in _AGENDA_LEAD_KEYWORDS:
        if sid not in ordered_ids:
            continue
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[sid] = score

    if not scores:
        return ordered_ids

    # Pick highest scoring domain (tie-break preserves canonical order so generic balanced prompts keep Value first)
    best_score = max(scores.values())
    lead_id = next((sid for sid in ordered_ids if scores.get(sid, 0) == best_score), ordered_ids[0])
    if lead_id == ordered_ids[0]:
        return ordered_ids
    return [lead_id] + [sid for sid in ordered_ids if sid != lead_id]

