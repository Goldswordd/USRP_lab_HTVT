#!/usr/bin/env python3
"""
measure_psnr.py - Đo PSNR giữa ảnh gốc và ảnh nhận được
Dùng sau khi thu với usrp_rx.py

Sử dụng:
    python3 measure_psnr.py --original ORIGINAL.jpg --received RECEIVED_FILE
    python3 measure_psnr.py --original input_q30.jpg --received output_usrp.jpg
"""
import os
import sys
import argparse
import io
import numpy as np
from PIL import Image


def compute_psnr_bytes(original: Image.Image, received_bytes: bytes) -> dict:
    """
    Đo PSNR và các metrics khác.
    Trả về dict với các thông số chất lượng.
    """
    result = {
        "status": "ok",
        "psnr_db": 0.0,
        "ssim": None,
        "mse": None,
        "received_bytes": len(received_bytes),
        "error": None,
    }

    if not received_bytes:
        result["status"] = "empty"
        result["error"] = "File rỗng"
        return result

    try:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # Cho phép load JPEG bị truncated (mất packet cuối)
        received = Image.open(io.BytesIO(received_bytes)).convert("RGB")
    except Exception as e:
        result["status"] = "decode_error"
        result["error"] = f"Không thể decode JPEG: {e}"
        return result

    orig_arr = np.array(original.convert("RGB"), dtype=np.float64)

    # Resize nếu kích thước khác nhau (do padding/truncation)
    if received.size != original.size:
        received = received.resize(original.size, Image.LANCZOS)
    recv_arr = np.array(received, dtype=np.float64)

    mse = np.mean((orig_arr - recv_arr) ** 2)
    result["mse"] = float(mse)

    if mse < 1e-10:
        result["psnr_db"] = 100.0
    else:
        result["psnr_db"] = float(10 * np.log10(255.0 ** 2 / mse))

    # Per-channel PSNR
    for i, ch in enumerate(["R", "G", "B"]):
        mse_ch = np.mean((orig_arr[:, :, i] - recv_arr[:, :, i]) ** 2)
        psnr_ch = 100.0 if mse_ch < 1e-10 else 10 * np.log10(255.0 ** 2 / mse_ch)
        result[f"psnr_{ch}"] = float(psnr_ch)

    return result


def main():
    parser = argparse.ArgumentParser(description="Đo PSNR ảnh nhận qua kênh vô tuyến")
    parser.add_argument("--original",  required=True, help="Ảnh gốc (JPEG)")
    parser.add_argument("--received",  required=True, help="File bytes nhận được")
    parser.add_argument("--ref-quality", type=int, default=None,
                        help="JPEG quality của ảnh gốc (để so sánh codec PSNR)")
    args = parser.parse_args()

    # Load ảnh gốc
    if not os.path.exists(args.original):
        print(f"[ERR] Không tìm thấy: {args.original}")
        sys.exit(1)
    original = Image.open(args.original).convert("RGB")
    print(f"\n[+] Ảnh gốc: {args.original} | Size: {original.size} | {os.path.getsize(args.original):,} bytes")

    # Load ảnh nhận
    if not os.path.exists(args.received):
        print(f"[ERR] Không tìm thấy: {args.received}")
        sys.exit(1)
    with open(args.received, "rb") as f:
        received_bytes = f.read()
    print(f"[+] File nhận: {args.received} | {len(received_bytes):,} bytes")

    # Đo PSNR
    metrics = compute_psnr_bytes(original, received_bytes)

    print(f"\n{'─'*50}")
    print(f"  KẾT QUẢ ĐO CHẤT LƯỢNG")
    print(f"{'─'*50}")
    print(f"  Status:   {metrics['status']}")

    if metrics['status'] == 'ok':
        print(f"  PSNR:     {metrics['psnr_db']:.2f} dB")
        print(f"  MSE:      {metrics['mse']:.4f}")
        if 'psnr_R' in metrics:
            print(f"  PSNR R/G/B: {metrics['psnr_R']:.1f} / {metrics['psnr_G']:.1f} / {metrics['psnr_B']:.1f} dB")

        # Đánh giá chất lượng
        psnr = metrics['psnr_db']
        if psnr >= 40:
            quality_label = "EXCELLENT (khó phân biệt bằng mắt)"
        elif psnr >= 30:
            quality_label = "GOOD (chấp nhận được)"
        elif psnr >= 20:
            quality_label = "FAIR (có thể thấy artifacts)"
        else:
            quality_label = "POOR (chất lượng kém)"
        print(f"  Đánh giá: {quality_label}")
    else:
        print(f"  Lỗi: {metrics['error']}")

    # So sánh với codec PSNR
    if args.ref_quality:
        import io
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=args.ref_quality,
                      optimize=False, progressive=False)
        codec_metrics = compute_psnr_bytes(original, buf.getvalue())
        print(f"\n  Codec PSNR (Q={args.ref_quality}, no channel): {codec_metrics['psnr_db']:.2f} dB")
        if metrics['status'] == 'ok':
            degradation = codec_metrics['psnr_db'] - metrics['psnr_db']
            print(f"  Suy giảm do kênh: {degradation:.2f} dB")

    print(f"{'─'*50}\n")
    return metrics


if __name__ == "__main__":
    main()
