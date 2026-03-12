#!/usr/bin/env python3
"""
plot_results.py - Vẽ đồ thị kết quả thí nghiệm JPEG-over-OFDM
Tạo 3 đồ thị:
  1. PSNR vs SNR (một đường per JPEG quality) — đồ thị chính
  2. PSNR vs JPEG quality (một đường per SNR) — đánh đổi codec vs kênh
  3. BER proxy: tỷ lệ bytes nhận / bytes gửi vs SNR

Sử dụng: python3 plot_results.py --csv results/results.csv [--outdir results]
"""
import os
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive, không cần display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Màu sắc đẹp cho từng mức quality ────────────────────────────────────
COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
MARKERS = ["o", "s", "^", "D", "v", "*"]


def load_csv(csv_path: str) -> dict:
    """
    Load CSV → dict[quality] = list of dicts
    Hỗ trợ 2 format:
      - results.csv (JPEG bytes): file_size_bytes, received_bytes
      - results_raw.csv (raw pixels): jpeg_size_bytes, reception_rate
    """
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        is_raw = "reception_rate" in fieldnames

        for row in reader:
            q    = int(row["jpeg_quality"])
            snr  = float(row["snr_db"])
            psnr = float(row["psnr_db"])
            codec_psnr = float(row.get("codec_psnr_db", 0))

            if is_raw:
                # results_raw.csv: jpeg_size_bytes = kích thước file JPEG (để tính bitrate)
                fsize = int(row.get("jpeg_size_bytes", 0))
                recv_rate = float(row.get("reception_rate", 0))
            else:
                fsize = int(row.get("file_size_bytes", 0))
                rbytes = int(row.get("received_bytes", 0))
                recv_rate = rbytes / fsize if fsize > 0 else 0

            if q not in data:
                data[q] = []
            data[q].append({
                "snr": snr, "psnr": psnr,
                "file_size": fsize,
                "recv_rate": recv_rate,   # tỷ lệ 0..1
                "codec_psnr": codec_psnr,
            })

    for q in data:
        data[q].sort(key=lambda x: x["snr"])
    return data


def plot_psnr_vs_snr(data: dict, outdir: str):
    """
    Đồ thị chính: PSNR (dB) vs SNR (dB)
    Mỗi đường = một mức JPEG quality
    Có thêm đường codec PSNR (PSNR không có kênh, đường trần)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("PSNR vs SNR — Raw Pixel OFDM Transmission (QPSK, AWGN)\n"
                 "Đề tài A: Truyền ảnh qua kênh vô tuyến OFDM — so sánh JPEG quality", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    qualities = sorted(data.keys())
    for i, q in enumerate(qualities):
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        points = data[q]

        snrs  = [p["snr"]  for p in points]
        psnrs = [p["psnr"] for p in points]

        # Đường đo được
        fsize_kb = points[0]['file_size'] / 1024
        label = f"Q={q} ({fsize_kb:.1f}KB)" if fsize_kb > 0 else f"Q={q}"
        ax.plot(snrs, psnrs, f"-{marker}",
                color=color, label=label,
                linewidth=2, markersize=7)

        # Đường codec PSNR (giới hạn trên — không có noise kênh)
        if points[0]["codec_psnr"] > 0:
            ax.axhline(y=points[0]["codec_psnr"], color=color,
                       linestyle=":", alpha=0.4, linewidth=1)

    # Ngưỡng chất lượng chấp nhận được
    ax.axhline(y=30, color="gray", linestyle="--", linewidth=1.5, label="30 dB (chấp nhận được)")
    ax.axhline(y=40, color="darkgray", linestyle="-.", linewidth=1, label="40 dB (tốt)")

    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))

    plt.tight_layout()
    out = os.path.join(outdir, "psnr_vs_snr.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu: {out}")
    return out


def plot_reception_rate(data: dict, outdir: str):
    """
    Đồ thị 2: Tỷ lệ bytes nhận được / bytes gửi vs SNR
    (Proxy cho BER / packet loss rate)
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Tỷ lệ bytes nhận được (%)", fontsize=12)
    ax.set_title("Tỷ lệ packet nhận được vs SNR\n(CRC check — phản ánh Packet Error Rate)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    qualities = sorted(data.keys())
    for i, q in enumerate(qualities):
        points = data[q]
        snrs  = [p["snr"] for p in points]
        rates = [100 * p["recv_rate"] for p in points]
        ax.plot(snrs, rates, f"-{MARKERS[i % len(MARKERS)]}",
                color=COLORS[i % len(COLORS)],
                label=f"Q={q}", linewidth=2)

    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="black", linestyle="--", linewidth=1, alpha=0.5, label="100% (perfect)")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(outdir, "reception_rate.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu: {out}")


def plot_rate_distortion(data: dict, outdir: str):
    """
    Đồ thị 3: Rate-Distortion curve
    X = file size (bits) = rate, Y = PSNR (distortion)
    Một đường per SNR, các điểm là các mức quality
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel("Bitrate (KB)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion Curve tại các mức SNR\n"
                 "Điểm trên mỗi đường = JPEG quality tăng dần", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Thu thập các mức SNR unique
    snr_set = sorted({p["snr"] for q in data for p in data[q]})
    snr_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snr_set)))

    for i, snr in enumerate(snr_set):
        sizes = []
        psnrs = []
        for q in sorted(data.keys()):
            pts = [p for p in data[q] if p["snr"] == snr]
            if pts:
                sizes.append(pts[0]["file_size"] / 1024)  # KB
                psnrs.append(pts[0]["psnr"])
        if sizes:
            ax.plot(sizes, psnrs, "-o", color=snr_colors[i],
                    label=f"SNR={snr:.0f}dB", linewidth=2, markersize=6)

    # Codec-only reference
    qs = sorted(data.keys())
    ref_sizes = [data[q][0]["file_size"] / 1024 for q in qs]
    ref_psnrs = [data[q][0]["codec_psnr"] for q in qs if data[q][0]["codec_psnr"] > 0]
    if ref_psnrs:
        ax.plot(ref_sizes[:len(ref_psnrs)], ref_psnrs, "k--", linewidth=1.5,
                label="Codec only (no channel)", alpha=0.6)

    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    out = os.path.join(outdir, "rate_distortion.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu: {out}")


def print_summary(data: dict):
    """In bảng tóm tắt kết quả ra terminal."""
    print("\n" + "=" * 70)
    print(f"{'PSNR (dB) — Quality vs SNR':^70}")
    print("=" * 70)

    snrs = sorted({p["snr"] for q in data for p in data[q]})
    qs   = sorted(data.keys())

    # Header
    header = f"{'Q\\SNR':>8}" + "".join(f"{s:>8.0f}" for s in snrs)
    print(header)
    print("-" * len(header))

    for q in qs:
        row = f"{'Q='+str(q):>8}"
        snr_map = {p["snr"]: p["psnr"] for p in data[q]}
        for s in snrs:
            val = snr_map.get(s, -1)
            if val <= 0:
                row += f"{'FAIL':>8}"
            else:
                row += f"{val:>8.1f}"
        print(row)

    print("=" * 70)
    print("Đơn vị: dB | FAIL = CRC failed / không decode được ảnh")


def main():
    parser = argparse.ArgumentParser(description="Vẽ đồ thị kết quả thí nghiệm")
    parser.add_argument("--csv",    required=True,  help="File CSV kết quả")
    parser.add_argument("--outdir", default=None,   help="Thư mục lưu ảnh (mặc định: cùng thư mục CSV)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERR] Không tìm thấy CSV: {args.csv}")
        return

    outdir = args.outdir or os.path.dirname(args.csv)
    os.makedirs(outdir, exist_ok=True)

    print(f"[+] Đọc dữ liệu từ: {args.csv}")
    data = load_csv(args.csv)

    print(f"[+] Tìm thấy {len(data)} mức quality: {sorted(data.keys())}")

    print_summary(data)

    print("\n[+] Vẽ đồ thị...")
    plot_psnr_vs_snr(data, outdir)
    plot_reception_rate(data, outdir)
    plot_rate_distortion(data, outdir)

    print(f"\n[*] Xong! Các đồ thị đã lưu vào: {outdir}")


if __name__ == "__main__":
    main()
