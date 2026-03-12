#!/usr/bin/env python3
"""
run_experiment_raw.py - Thí nghiệm truyền ảnh qua OFDM dùng RAW PIXELS

Khác với run_experiment.py (dùng JPEG bytes):
  - TX: Nén ảnh → JPEG → giải nén → lấy raw pixels → truyền qua OFDM
  - RX: Nhận raw bytes → fill 0 cho packet bị drop → ghép lại thành ảnh
  - PSNR: so với ảnh gốc raw (không qua nén)

Ưu điểm:
  - PSNR giảm MƯỢT theo SNR (không có cliff effect của JPEG)
  - Phân tách rõ ràng: D_total = D_codec (JPEG quality) + D_channel (packet loss)
  - Luôn decode được ảnh kể cả khi có packet loss

Sơ đồ distortion:
  Original → [JPEG Q] → Codec_Image → [OFDM + AWGN] → Received_Image
       └─────────────────────────────────────────────→ PSNR

Sử dụng: python3 run_experiment_raw.py [--input IMAGE] [--quick] [--size WxH]
"""
import os, sys, io, time, csv, argparse, subprocess, shutil
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_protocol import pack_data, unpack_data, compute_n_packets

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

DEFAULT_INPUT = os.path.join(HERE, "..", "Kết quả", "truyền ảnh", "input.jpg")

QUALITY_LEVELS = [10, 20, 30, 50, 70, 90]
SNR_DB_LEVELS  = [5, 10, 15, 20, 25, 30, 35, 40]
QUALITY_QUICK  = [10, 30, 70]
SNR_DB_QUICK   = [10, 20, 30, 40]

PACKET_LEN      = 96   # bytes per OFDM packet (phải khớp với ofdm_tx/rx_headless.py)
SYNC_WARMUP_PKT = 16   # số packets warmup (null bytes) cho Schmidl-Cox sync
SYNC_WARMUP     = SYNC_WARMUP_PKT * PACKET_LEN  # = 1536 bytes


def get_codec_image(original: Image.Image, quality: int, size: tuple) -> np.ndarray:
    """
    Nén → giải nén ảnh tại mức quality, resize về size.
    Trả về numpy array (H, W, 3) uint8 = ảnh sau nén codec, trước khi truyền.
    """
    img = original.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False, progressive=False)
    codec_img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.array(codec_img, dtype=np.uint8)


def pixels_to_bytes(arr: np.ndarray) -> bytes:
    """numpy uint8 array → bytes."""
    return arr.tobytes()


def bytes_to_pixels(data: bytes, shape: tuple) -> np.ndarray:
    """
    bytes → numpy array với shape (H, W, 3).
    Padding 0 nếu data thiếu (packet loss concealment).
    """
    expected = shape[0] * shape[1] * shape[2]
    if len(data) < expected:
        data = data + bytes(expected - len(data))
    return np.frombuffer(data[:expected], dtype=np.uint8).reshape(shape)


def compute_psnr(ref: np.ndarray, recv: np.ndarray) -> float:
    """PSNR giữa hai numpy arrays (uint8, 0-255)."""
    ref_f  = ref.astype(np.float64)
    recv_f = recv.astype(np.float64)
    mse    = np.mean((ref_f - recv_f) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def run_tx(input_file: str, iq_file: str, snr_db: float) -> bool:
    cmd = [sys.executable, os.path.join(HERE, "ofdm_tx_headless.py"),
           "--input", input_file, "--output", iq_file, "--snr", str(snr_db)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and os.path.exists(iq_file) and os.path.getsize(iq_file) > 0
    except subprocess.TimeoutExpired:
        return False


def run_rx(iq_file: str, output_file: str, timeout: float) -> bool:
    cmd = [sys.executable, os.path.join(HERE, "ofdm_rx_headless.py"),
           "--input", iq_file, "--output", output_file, "--timeout", str(timeout)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout) + 30)
        return os.path.exists(output_file)
    except subprocess.TimeoutExpired:
        return False


def estimate_timeout(payload_bytes: int) -> float:
    """Ước lượng timeout dựa trên kích thước payload."""
    samp_rate = 100_000
    ofdm_sym_rate = samp_rate / 80   # = 1250 sym/sec
    bytes_per_sec = ofdm_sym_rate * 48 * 2 / 8  # QPSK, 48 carriers
    return max((payload_bytes / bytes_per_sec) * 3 + 10, 20.0)


def run_experiment(input_image: str, quality_levels: list, snr_levels: list,
                   image_size: tuple, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    original = Image.open(input_image).convert("RGB").resize(image_size, Image.LANCZOS)
    orig_arr = np.array(original, dtype=np.uint8)
    H, W = image_size[1], image_size[0]
    shape = (H, W, 3)
    pixel_bytes = H * W * 3

    print(f"\n[+] Ảnh: {image_size} = {pixel_bytes:,} bytes raw | {H*W*3//PACKET_LEN} packets")
    print(f"[+] Quality: {quality_levels}, SNR: {snr_levels} dB")
    print(f"[+] Tổng: {len(quality_levels)*len(snr_levels)} điểm\n")

    csv_path = os.path.join(outdir, "results_raw.csv")
    tmp_dir  = os.path.join(outdir, "tmp_raw")
    os.makedirs(tmp_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "jpeg_quality", "snr_db", "psnr_db",
            "codec_psnr_db", "reception_rate", "jpeg_size_bytes"
        ])

    timeout = estimate_timeout(pixel_bytes + SYNC_WARMUP)
    total = len(quality_levels) * len(snr_levels)
    done = 0; t0_exp = time.time()

    for q in quality_levels:
        # Codec image (JPEG compress → decompress → raw pixels)
        codec_arr   = get_codec_image(original, q, image_size)
        codec_psnr  = compute_psnr(orig_arr, codec_arr)
        codec_bytes = pixels_to_bytes(codec_arr)

        # JPEG file size (để báo cáo compression ratio)
        buf = io.BytesIO()
        Image.fromarray(codec_arr).save(buf, format="JPEG", quality=q)
        jpeg_size = buf.tell()

        # TX file: dùng app_protocol (sequence numbers để detect gaps)
        tx_path = os.path.join(tmp_dir, f"tx_q{q:02d}.bin")
        tx_data = pack_data(codec_bytes, warmup_packets=SYNC_WARMUP_PKT)
        with open(tx_path, "wb") as f:
            f.write(tx_data)

        print(f"[Q={q:2d}] Codec PSNR: {codec_psnr:.1f}dB | "
              f"JPEG: {jpeg_size//1000:.1f}KB | Raw: {pixel_bytes//1000:.1f}KB")

        for snr in snr_levels:
            iq_path  = os.path.join(tmp_dir, f"iq_q{q:02d}_snr{snr:02d}.bin")
            rx_path  = os.path.join(tmp_dir, f"rx_q{q:02d}_snr{snr:02d}.bin")
            t0 = time.time()

            tx_ok = run_tx(tx_path, iq_path, float(snr))
            rx_ok = run_rx(iq_path, rx_path, timeout) if tx_ok else False

            if rx_ok and os.path.exists(rx_path):
                with open(rx_path, "rb") as f:
                    recv_all = f.read()

                # Dùng app_protocol để unpack: detect gaps, fill với 0x80 (gray)
                recv_data, stats = unpack_data(recv_all, pixel_bytes, SYNC_WARMUP_PKT)
                reception_rate   = stats["packet_reception_rate"]

                # Ghép ảnh với concealment đúng vị trí
                recv_arr = bytes_to_pixels(recv_data, shape)
                psnr_db  = compute_psnr(orig_arr, recv_arr)
            else:
                stats = {"packet_reception_rate": 0.0, "lost_data_packets": compute_n_packets(pixel_bytes)}
                reception_rate   = 0.0
                psnr_db          = 0.0

            done += 1
            eta = (time.time() - t0_exp) / done * (total - done)
            print(f"  SNR={snr:2d}dB | PSNR={psnr_db:6.2f}dB | "
                  f"RX={100*reception_rate:.0f}% | {time.time()-t0:.1f}s | ETA:{eta:.0f}s")

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    q, snr, round(psnr_db, 2),
                    round(codec_psnr, 2), round(reception_rate, 3), jpeg_size
                ])

            # Cleanup IQ file (lớn)
            for p in [iq_path]:
                if os.path.exists(p):
                    os.remove(p)

        print()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[+] Done! {done} points in {time.time()-t0_exp:.0f}s → {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Raw pixel OFDM image experiment")
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=RESULTS)
    parser.add_argument("--quick",  action="store_true")
    parser.add_argument("--size",   default="64x64",
                        help="Kích thước ảnh WxH (mặc định 64x64 để simulation nhanh)")
    args = parser.parse_args()

    w, h   = map(int, args.size.split("x"))
    qs     = QUALITY_QUICK if args.quick else QUALITY_LEVELS
    snrs   = SNR_DB_QUICK  if args.quick else SNR_DB_LEVELS

    print(f"[*] Image size: {w}x{h} | Quality: {qs} | SNR: {snrs}")
    if args.quick:
        print("[*] Quick mode")

    csv_path = run_experiment(
        input_image   = args.input,
        quality_levels= qs,
        snr_levels    = snrs,
        image_size    = (w, h),
        outdir        = args.outdir,
    )
    print(f"\n[*] Vẽ đồ thị: python3 plot_results.py --csv {csv_path}")


if __name__ == "__main__":
    main()
