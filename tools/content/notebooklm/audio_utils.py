"""Audio Utilities สำหรับ NotebookLM Pipeline และ Discord Notifier

จัดการบีบอัดไฟล์เสียงพอดแคสต์ (Voice/Dialogue) ให้มีขนาดเล็ก (<8-10 MB)
ด้วย Bitrate 48k Mono เพื่อให้ส่งขึ้น Discord และจัดเก็บใน Vault ได้อย่างมีประสิทธิภาพ
"""
import os
from pathlib import Path
import subprocess
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)


def get_ffmpeg_binary() -> Optional[str]:
    """ค้นหา ffmpeg binary จาก imageio-ffmpeg หรือ system PATH"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    import shutil
    sys_exe = shutil.which("ffmpeg")
    if sys_exe:
        return sys_exe

    return None


def compress_audio_for_discord(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    target_bitrate: str = "48k",
    max_size_bytes: Optional[int] = None,
    max_size_mb: Optional[float] = None,
    force: bool = False,
) -> Path:
    """บีบอัดไฟล์เสียงสำหรับ Voice / Podcast เป็น Mono AAC/M4A คุณภาพสูงแต่ขนาดเล็ก

    Args:
        input_path: เส้นทางไฟล์เสียงต้นฉบับ
        output_path: เส้นทางไฟล์ผลลัพธ์ (ถ้า None จะสร้างชื่อ <stem>_compressed.m4a หรือเขียนทับตามบริบท)
        target_bitrate: Audio bitrate เช่น '48k' หรือ '64k'
        max_size_bytes: ขนาด byte สูงสุดที่ยอมรับได้โดยไม่ต้องบีบอัด (ถ้าไม่ระบุ จะอ่านจาก get_max_discord_audio_bytes())
        max_size_mb: ขนาด MB สูงสุด (legacy parameter)
        force: บังคับบีบอัดเสมอแม้ไฟล์จะเล็กกว่า limit

    Returns:
        Path ของไฟล์เสียงที่บีบอัดแล้ว (หรือไฟล์เดิมหากไม่จำเป็น/ล้มเหลว)
    """
    src = Path(input_path).resolve()
    if not src.exists() or not src.is_file():
        logger.warning("Audio file not found for compression: %s", src)
        return src

    if max_size_bytes is not None:
        threshold_bytes = max_size_bytes
    elif max_size_mb is not None:
        threshold_bytes = int(max_size_mb * 1024 * 1024)
    else:
        from core.discord_notifier import get_max_discord_audio_bytes
        threshold_bytes = get_max_discord_audio_bytes()

    orig_size_bytes = src.stat().st_size
    orig_size_mb = orig_size_bytes / (1024 * 1024)
    if not force and orig_size_bytes <= threshold_bytes:
        logger.info("Audio file %s is already %.2f MB (<= %.2f MB), skipping compression.", src.name, orig_size_mb, threshold_bytes / (1024 * 1024))
        return src


    ffmpeg_exe = get_ffmpeg_binary()
    if not ffmpeg_exe:
        logger.warning("ffmpeg binary not found. Skipping audio compression for %s.", src.name)
        return src

    if output_path is None:
        dst = src.parent / f"{src.stem}_compressed.m4a"
    else:
        dst = Path(output_path).resolve()

    dst_tmp = dst.parent / f".tmp_{dst.name}"

    try:
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(src),
            "-ac", "1",           # Downmix to Mono (Voice podcast doesn't need stereo)
            "-c:a", "aac",        # Standard AAC encoder compatible with Discord & all browsers
            "-b:a", target_bitrate,
            "-v", "error",
            str(dst_tmp),
        ]
        logger.info("Compressing audio %s (%.2f MB) -> target %s...", src.name, orig_size_mb, target_bitrate)
        subprocess.run(cmd, check=True, capture_output=True)

        if dst_tmp.exists() and dst_tmp.stat().st_size > 0:
            if dst.exists():
                dst.unlink()
            dst_tmp.rename(dst)
            new_size_mb = dst.stat().st_size / (1024 * 1024)
            reduction = (1 - (new_size_mb / orig_size_mb)) * 100 if orig_size_mb > 0 else 0
            logger.info("Audio compression complete: %.2f MB -> %.2f MB (-%.1f%%) saved to %s", orig_size_mb, new_size_mb, reduction, dst.name)
            return dst
        else:
            logger.warning("Compressed output was empty, keeping original audio: %s", src.name)
            return src
    except Exception as e:
        logger.warning("Audio compression failed for %s (%s). Keeping original audio.", src.name, e)
        if dst_tmp.exists():
            try:
                dst_tmp.unlink()
            except Exception:
                pass
        return src
