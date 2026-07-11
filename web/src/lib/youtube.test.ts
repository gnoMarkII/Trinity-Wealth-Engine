// @vitest-environment node — test ล้วนๆ ไม่แตะ DOM ข้าม jsdom ให้รันเร็วขึ้น
import { describe, expect, it } from 'vitest'
import { youtubeVideoId } from './youtube'

describe('youtubeVideoId', () => {
  it('ดึง id จาก youtube.com/watch?v= และ youtu.be/', () => {
    expect(youtubeVideoId('https://www.youtube.com/watch?v=dQw4w9WgXcQ')).toBe('dQw4w9WgXcQ')
    expect(youtubeVideoId('https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s')).toBe('dQw4w9WgXcQ')
    expect(youtubeVideoId('https://youtu.be/dQw4w9WgXcQ')).toBe('dQw4w9WgXcQ')
  })

  it('ปฏิเสธโดเมนปลอมที่แค่มีคำว่า youtube', () => {
    expect(youtubeVideoId('https://evilyoutube.com/watch?v=dQw4w9WgXcQ')).toBeNull()
    expect(youtubeVideoId('https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ')).toBeNull()
  })

  it('ปฏิเสธ id ที่รูปแบบไม่ถูกต้อง (ต้องเป็น 11 ตัวอักษร)', () => {
    expect(youtubeVideoId('https://www.youtube.com/watch?v=short')).toBeNull()
    expect(youtubeVideoId('https://www.youtube.com/watch?v=')).toBeNull()
    expect(youtubeVideoId('https://www.youtube.com/watch')).toBeNull()
  })

  it('string ที่ไม่ใช่ URL คืน null ไม่ throw', () => {
    expect(youtubeVideoId('not a url')).toBeNull()
    expect(youtubeVideoId('')).toBeNull()
  })
})
