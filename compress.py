#!/usr/bin/env python3
"""
compress.py - Nén ảnh JPEG ở các mức chất lượng khác nhau
Tạo tập ảnh đầu vào cho thí nghiệm QP vs PSNR vs SNR

Sử dụng: python3 compress.py [--input IMAGE] [--outdir OUTPUT_DIR]
"""
import os
import argparse
import numpy as np
from PIL import Image
import io

# Mức chất lượng JPEG cần thử nghiệm
# PIL quality: 1 (tệ nhất) → 95 (tốt nhất)
QUALITY_LEVELS = [10, 20, 30, 50, 70, 90]

# Ảnh mặc định (ảnh gốc của khóa luận cũ)
DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Kết quả", "truyền ảnh", "input.jpg"
)


def compress_jpeg(img: Image.Image, quality: int) -> bytes:
    """Nén ảnh PIL thành bytes JPEG với quality cho trước."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False, progressive=False)
    return buf.getvalue()


def compute_psnr(original: Image.Image, compressed: Image.Image) -> float:
    """Tính PSNR giữa ảnh gốc và ảnh nén (dB)."""
    orig = np.array(original.convert("RGB"), dtype=np.float64)
    comp = np.array(compressed.resize(original.size).convert("RGB"), dtype=np.float64)
    mse = np.mean((orig - comp) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(description="Tạo tập ảnh JPEG nén ở nhiều mức chất lượng")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Ảnh đầu vào (mặc định: input.jpg của khóa luận)")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"), help="Thư mục lưu kết quả")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load ảnh gốc
    print(f"[+] Load ảnh gốc: {args.input}")
    original = Image.open(args.input).convert("RGB")
    w, h = original.size
    print(f"    Kích thước: {w}x{h} pixels")

    # Lưu ảnh gốc không nén (reference)
    orig_path = os.path.join(args.outdir, "original_raw.bmp")
    original.save(orig_path, format="BMP")
    print(f"    Reference BMP: {orig_path}")

    print(f"\n{'Quality':>10} {'File size (bytes)':>20} {'PSNR (dB)':>12} {'Output file':>35}")
    print("-" * 80)

    results = []
    for q in QUALITY_LEVELS:
        # Nén JPEG
        jpeg_bytes = compress_jpeg(original, q)
        size_bytes = len(jpeg_bytes)

        # Đo PSNR của ảnh nén (codec distortion, không có noise kênh)
        compressed_img = Image.open(io.BytesIO(jpeg_bytes))
        codec_psnr = compute_psnr(original, compressed_img)

        # Lưu file JPEG
        fname = f"input_q{q:02d}.jpg"
        fpath = os.path.join(args.outdir, fname)
        with open(fpath, "wb") as f:
            f.write(jpeg_bytes)

        results.append((q, size_bytes, codec_psnr))
        print(f"{q:>10} {size_bytes:>20,} {codec_psnr:>12.2f} {fname:>35}")

    print("-" * 80)
    print(f"\n[+] Đã lưu {len(QUALITY_LEVELS)} ảnh nén vào: {args.outdir}")
    print("\nNhận xét:")
    print(f"  Quality 10: file nhỏ (~{results[0][1]//1000}KB), PSNR thấp (~{results[0][2]:.1f}dB)")
    print(f"  Quality 90: file lớn (~{results[-1][1]//1000}KB), PSNR cao (~{results[-1][2]:.1f}dB)")
    print("\n[*] Các file này sẽ được dùng làm đầu vào cho thí nghiệm truyền OFDM")

    return results


if __name__ == "__main__":
    main()
