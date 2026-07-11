/** ดึง video id 11 ตัวอักษรจาก URL ของ YouTube — คืน null ถ้าไม่ใช่โดเมน YouTube จริง
 * หรือรูปแบบ id ไม่ถูกต้อง (กัน embed URL แปลกปลอมจากข้อมูลอ้างอิงภายนอก) */
export function youtubeVideoId(url: string): string | null {
  try {
    const parsed = new URL(url)
    if (!/(^|\.)youtube\.com$|(^|\.)youtu\.be$/i.test(parsed.hostname)) return null
    const candidate = parsed.hostname.endsWith('youtu.be') ? parsed.pathname.slice(1) : parsed.searchParams.get('v')
    return candidate && /^[\w-]{11}$/.test(candidate) ? candidate : null
  } catch {
    return null
  }
}
