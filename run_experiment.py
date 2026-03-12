#!/usr/bin/env python3
"""
run_experiment.py - Chạy toàn bộ thí nghiệm QP vs PSNR vs SNR
Đề tài A: "Đánh giá chất lượng truyền ảnh nén JPEG qua kênh vô tuyến OFDM"

Pipeline:
    Với mỗi (quality, snr_db):
        1. Nén ảnh gốc → JPEG tại mức quality
        2. TX headless: JPEG bytes → OFDM+AWGN(snr_db) → IQ file
        3. RX headless: IQ file → OFDM decode → received bytes
        4. Đo PSNR: original vs received
        5. Lưu kết quả vào CSV

Sử dụng: python3 run_experiment.py [--input IMAGE] [--outdir DIR] [--quick]
"""
import os
import sys
import time
import csv
import io
import argparse
import subprocess
import shutil
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated JPEG (mất packet cuối)

# ─── Thư mục gốc của project ───────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ─── Ảnh gốc ───────────────────────────────────────────────────────────────
DEFAULT_INPUT = os.path.join(
    HERE, "..", "Kết quả", "truyền ảnh", "input.jpg"
)

# ─── Không gian tham số ────────────────────────────────────────────────────
QUALITY_LEVELS = [10, 20, 30, 50, 70, 90]
SNR_DB_LEVELS  = [5, 10, 15, 20, 25, 30, 35, 40]

# Quick mode (để test nhanh)
QUALITY_QUICK  = [20, 50, 80]
SNR_DB_QUICK   = [10, 20, 30]


# ─────────────────────────────────────────────────────────────────────────────
def compress_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False, progressive=False)
    return buf.getvalue()


# Số null bytes để warmup Schmidl-Cox sync (16 OFDM packets × 96 bytes)
SYNC_WARMUP_BYTES = 16 * 96  # = 1536 bytes


def extract_jpeg(data: bytes) -> bytes:
    """
    Tìm và trích xuất JPEG từ stream bytes có thể có prefix.
    Tìm SOI marker (0xFF 0xD8) và trả về từ đó đến cuối.
    """
    soi = data.find(b'\xff\xd8')
    if soi == -1:
        return data  # Không tìm thấy marker → trả về nguyên
    return data[soi:]


def compute_psnr(img_original: Image.Image, received_bytes: bytes) -> float:
    """
    Tính PSNR giữa ảnh gốc và bytes nhận được.
    Tự động loại bỏ sync warmup prefix bằng cách tìm JPEG SOI marker.
    Trả về 0.0 nếu không decode được (file bị lỗi).
    """
    if not received_bytes:
        return 0.0
    # Tìm JPEG SOI marker để bỏ qua sync warmup prefix
    jpeg_bytes = extract_jpeg(received_bytes)
    try:
        received_img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        orig_arr = np.array(img_original.convert("RGB"), dtype=np.float64)
        recv_arr = np.array(
            received_img.resize(img_original.size, Image.LANCZOS).convert("RGB"),
            dtype=np.float64
        )
        mse = np.mean((orig_arr - recv_arr) ** 2)
        if mse < 1e-10:
            return 100.0  # Perfect
        return float(10 * np.log10(255.0 ** 2 / mse))
    except Exception:
        # Không load được ảnh → file bị lỗi
        return 0.0


def run_tx(input_file: str, iq_file: str, snr_db: float) -> bool:
    """Chạy OFDM TX headless. Trả về True nếu thành công."""
    cmd = [
        sys.executable,
        os.path.join(HERE, "ofdm_tx_headless.py"),
        "--input",  input_file,
        "--output", iq_file,
        "--snr",    str(snr_db),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"      [TX ERR] {result.stderr[-200:]}")
            return False
        return os.path.exists(iq_file) and os.path.getsize(iq_file) > 0
    except subprocess.TimeoutExpired:
        print("      [TX] Timeout!")
        return False


def run_rx(iq_file: str, output_file: str, timeout: float = 30.0) -> bool:
    """Chạy OFDM RX headless. Trả về True nếu nhận được dữ liệu."""
    cmd = [
        sys.executable,
        os.path.join(HERE, "ofdm_rx_headless.py"),
        "--input",   iq_file,
        "--output",  output_file,
        "--timeout", str(timeout),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout) + 30)
        if result.returncode != 0:
            print(f"      [RX ERR] {result.stderr[-200:]}")
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0
    except subprocess.TimeoutExpired:
        print("      [RX] Timeout!")
        return False


def estimate_rx_timeout(file_size_bytes: int, samp_rate: int = 100_000) -> float:
    """
    Ước lượng thời gian nhận cần thiết (giây).
    QPSK: 2 bits/symbol, 48 carriers, samp_rate / (FFT_LEN + CP_LEN) symbols/sec
    """
    fft_len = 64
    cp_len  = 16
    bits_per_symbol = 2    # QPSK
    n_carriers = 48
    payload_bits_per_ofdm_sym = bits_per_symbol * n_carriers  # 96 bits = 12 bytes

    symbols_per_sec = samp_rate / (fft_len + cp_len)          # 1250 sym/s @ 100kHz
    bytes_per_sec   = payload_bits_per_ofdm_sym * symbols_per_sec / 8

    # overhead: CRC (4 bytes), header symbols, sync words → x2
    estimated_sec = (file_size_bytes / bytes_per_sec) * 2
    return max(estimated_sec + 5.0, 15.0)  # tối thiểu 15 giây


def run_experiment(
    input_image: str,
    quality_levels: list,
    snr_levels: list,
    outdir: str,
    verbose: bool = True
) -> str:
    """
    Chạy toàn bộ thí nghiệm.
    Trả về đường dẫn file CSV kết quả.
    """
    os.makedirs(outdir, exist_ok=True)

    # Load ảnh gốc
    original = Image.open(input_image).convert("RGB")
    print(f"\n[+] Ảnh gốc: {input_image} | Size: {original.size}")
    print(f"[+] Tham số: quality={quality_levels}, SNR={snr_levels} dB")
    print(f"[+] Tổng số điểm thí nghiệm: {len(quality_levels) * len(snr_levels)}")

    csv_path = os.path.join(outdir, "results.csv")
    tmp_dir  = os.path.join(outdir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "jpeg_quality", "snr_db",
            "psnr_db", "file_size_bytes", "received_bytes",
            "codec_psnr_db",  # PSNR chỉ do nén (không có noise)
        ])

    total_points = len(quality_levels) * len(snr_levels)
    done = 0
    t_start = time.time()

    for q in quality_levels:
        # Nén ảnh ở mức quality q
        jpeg_bytes = compress_jpeg(original, q)
        jpeg_size  = len(jpeg_bytes)

        # PSNR chỉ do codec (không có noise kênh) — so với ảnh gốc raw
        codec_psnr  = compute_psnr(original, jpeg_bytes)

        # Lưu ảnh nén với warmup prefix (null bytes để Schmidl-Cox sync lock trước)
        # Không có warmup: sync acquisition mất 2-5 packets đầu → mất JPEG header → file hỏng
        input_path = os.path.join(tmp_dir, f"input_q{q:02d}.jpg")
        with open(input_path, "wb") as f:
            f.write(bytes(SYNC_WARMUP_BYTES))  # 1536 null bytes
            f.write(jpeg_bytes)

        print(f"\n{'─'*60}")
        print(f"[Q={q:2d}] File size: {jpeg_size:,} bytes | Codec PSNR: {codec_psnr:.1f} dB")

        timeout = estimate_rx_timeout(jpeg_size)

        for snr in snr_levels:
            iq_file  = os.path.join(tmp_dir, f"iq_q{q:02d}_snr{snr:02d}.bin")
            out_file = os.path.join(tmp_dir, f"recv_q{q:02d}_snr{snr:02d}.jpg")

            t0 = time.time()

            # 1) TX
            tx_ok = run_tx(input_path, iq_file, float(snr))

            if not tx_ok:
                psnr_db = 0.0
                recv_bytes = 0
            else:
                # 2) RX
                rx_ok = run_rx(iq_file, out_file, timeout)

                if not rx_ok:
                    psnr_db = 0.0
                    recv_bytes = 0
                else:
                    recv_bytes = os.path.getsize(out_file)
                    with open(out_file, "rb") as f:
                        recv_data = f.read()
                    psnr_db = compute_psnr(original, recv_data)

            elapsed = time.time() - t0
            done += 1

            # Lưu kết quả
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([q, snr, round(psnr_db, 2), jpeg_size,
                                 recv_bytes, round(codec_psnr, 2)])

            # In tiến độ
            status = "OK" if psnr_db > 0 else "FAIL"
            eta = (time.time() - t_start) / done * (total_points - done)
            print(f"  SNR={snr:2d}dB | PSNR={psnr_db:6.2f}dB | "
                  f"{recv_bytes:,}/{jpeg_size:,}B | {status} | "
                  f"{elapsed:.1f}s | ETA: {eta:.0f}s")

            # Xóa file tạm để tiết kiệm disk
            for f in [iq_file]:
                if os.path.exists(f):
                    os.remove(f)

    # Dọn dẹp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"[+] Thí nghiệm hoàn thành! {done} điểm trong {total_time:.0f}s")
    print(f"[+] Kết quả: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run JPEG-over-OFDM experiment")
    parser.add_argument("--input",  default=DEFAULT_INPUT,
                        help="Ảnh đầu vào")
    parser.add_argument("--outdir", default=RESULTS,
                        help="Thư mục lưu kết quả")
    parser.add_argument("--quick",  action="store_true",
                        help="Quick mode (3 quality × 3 SNR)")
    parser.add_argument("--snr-min",  type=int, default=5)
    parser.add_argument("--snr-max",  type=int, default=40)
    parser.add_argument("--snr-step", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERR] Không tìm thấy ảnh: {args.input}")
        sys.exit(1)

    if args.quick:
        qualities = QUALITY_QUICK
        snr_levels = SNR_DB_QUICK
        print("[*] Quick mode: 3x3 = 9 điểm")
    else:
        qualities  = QUALITY_LEVELS
        snr_levels = list(range(args.snr_min, args.snr_max + 1, args.snr_step))

    csv_path = run_experiment(
        input_image   = args.input,
        quality_levels= qualities,
        snr_levels    = snr_levels,
        outdir        = args.outdir
    )

    print(f"\n[*] Tiếp theo: python3 plot_results.py --csv {csv_path}")


if __name__ == "__main__":
    main()
