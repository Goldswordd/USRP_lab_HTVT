# LAB GUIDE — USRP N210 OFDM Image Transmission
## Đề tài A: Đánh giá chất lượng truyền ảnh nén JPEG qua kênh vô tuyến OFDM

---

## PHẦN 1: CHUẨN BỊ TRƯỚC KHI LÊN LAB (làm ở nhà tối nay)

### 1.1 Chạy thí nghiệm mô phỏng (không cần USRP)

```bash
cd /home/johnw/Documents/Master-UET-doc/USRP/experiment

# Bước 1: Nén ảnh ở các mức quality khác nhau
python3 compress.py
# Output: results/input_q10.jpg, input_q20.jpg, ...

# Bước 2: Chạy quick test (3x3 = 9 điểm, ~5 phút)
python3 run_experiment.py --quick
# Output: results/results.csv

# Bước 3: Vẽ đồ thị
python3 plot_results.py --csv results/results.csv
# Output: results/psnr_vs_snr.png, rate_distortion.png, reception_rate.png
```

### 1.2 Kiểm tra kết nối USRP (tại lab)

```bash
# Kiểm tra USRP TX (PC1)
uhd_find_devices --args "addr=192.168.10.2"

# Kiểm tra USRP RX (PC2)
uhd_find_devices --args "addr=192.168.10.3"

# Test nhanh (đo công suất)
uhd_fft --freq 2.4e9 --rate 1e6 --gain 30 --args "addr=192.168.10.3"
```

---

## PHẦN 2: THÍ NGHIỆM THỰC CHIẾN VỚI USRP N210

### Sơ đồ kết nối

```
[PC1 TX]──GigE──[USRP N210 #1]──Antenna──] [──Antenna──[USRP N210 #2]──GigE──[PC2 RX]
                 TX/RX port                              RX2 port
                 2.4 GHz                                 2.4 GHz
```

**Lưu ý antenna:**
- USRP TX: kết nối vào cổng **TX/RX**
- USRP RX: kết nối vào cổng **RX2**
- Khoảng cách thử nghiệm ban đầu: ~1-2 mét để đảm bảo nhận được

### 2.1 Tạo ảnh thử nghiệm

```bash
# Trên PC TX:
cd /path/to/experiment

python3 compress.py
# Sẽ tạo: results/input_q10.jpg, input_q30.jpg, input_q50.jpg, input_q70.jpg, input_q90.jpg
```

### 2.2 Chạy USRP TX (PC1 — máy kết nối USRP TX)

```bash
# Phát ảnh nén Q=30 qua không khí
python3 usrp_tx.py \
    --input results/input_q30.jpg \
    --addr 192.168.10.2 \
    --freq 2.4e9 \
    --gain 30

# Giữ chạy trong khi RX đang thu
# Ctrl+C để dừng
```

### 2.3 Chạy USRP RX (PC2 — máy kết nối USRP RX)

```bash
# Thu trong 60 giây
python3 usrp_rx.py \
    --output received_q30.jpg \
    --addr 192.168.10.3 \
    --freq 2.4e9 \
    --gain 30 \
    --time 60
```

### 2.4 Đo PSNR

```bash
# Sau khi RX xong:
python3 measure_psnr.py \
    --original results/input_q30.jpg \
    --received received_q30.jpg \
    --ref-quality 30
```

---

## PHẦN 3: SWEEP THÍ NGHIỆM THỦ CÔNG

Vì hardware thật không có GUI SNR slider, thay đổi SNR bằng cách:

### Cách 1: Thay đổi khoảng cách TX-RX
| Khoảng cách | Ước lượng SNR |
|-------------|---------------|
| 0.5 m       | ~35-40 dB     |
| 1 m         | ~30-35 dB     |
| 2 m         | ~25-30 dB     |
| 5 m         | ~20-25 dB     |
| 10 m        | ~15-20 dB     |

### Cách 2: Thay đổi TX gain
```bash
# SNR cao (TX gain cao)
python3 usrp_tx.py --input results/input_q30.jpg --gain 40 ...

# SNR thấp (TX gain thấp)
python3 usrp_tx.py --input results/input_q30.jpg --gain 10 ...
```

### Cách 3: Thêm attenuator vật lý giữa TX và RX antenna

### Script tự động đo (manual sweep)

```bash
# Chạy trên PC RX sau khi điều chỉnh khoảng cách/gain
for q in 10 30 50 70 90; do
    echo "=== Quality $q ==="
    python3 usrp_rx.py --output recv_q${q}_d${DISTANCE}m.jpg --time 30
    python3 measure_psnr.py \
        --original results/input_q${q}.jpg \
        --received recv_q${q}_d${DISTANCE}m.jpg
done
```

---

## PHẦN 4: BUG CỦA CODE GỐC (đã fix trong code mới)

| Vấn đề | Code gốc | Code mới (đã fix) |
|--------|----------|-------------------|
| Payload modulation | RX dùng 16QAM, TX dùng QPSK | Cả TX và RX đều dùng QPSK |
| Costas loop order | 16 (cho 16QAM) | 4 (cho QPSK) |
| Hardcoded paths | `/home/cgminh/...` | Command-line arguments |
| GUI dependency | Qt5 GUI bắt buộc | Headless, không cần display |

---

## PHẦN 5: THÔNG SỐ OFDM HỆ THỐNG

```
FFT Length:        64 subcarriers
Cyclic Prefix:     16 samples (25%)
Occupied carriers: 48 (data + pilot)
Pilot carriers:    4 (positions: -21, -7, +7, +21)
Header modulation: BPSK  (1 bit/symbol, robust)
Payload modulation:QPSK  (2 bits/symbol)
Frame sync:        Schmidl-Cox algorithm
Channel est.:      Pilot-aided (simpledfe)
Error detection:   CRC32 per packet (96 bytes)
Sample rate:       1 MHz (USRP) / 100 kHz (simulation)
Center frequency:  2.4 GHz (ISM band, unlicensed)
```

**Throughput lý thuyết:**
- QPSK, 48 carriers, 1 MHz sample rate
- Symbol rate = 1MHz / (64+16) = 12500 symbols/sec
- Payload rate = 12500 × 48 × 2 bits / 8 = 150 Kbps
- Sau overhead (header, CRC, CP): ~100 Kbps thực tế

---

## PHẦN 6: TROUBLESHOOTING

### Không detect được USRP
```bash
# Kiểm tra IP
ping 192.168.10.2

# Kiểm tra UHD version
uhd_config_info --print-all

# Nếu IP khác, check:
uhd_find_devices
```

### RX không decode được
1. Tăng RX gain: `--gain 40`
2. Giảm khoảng cách TX-RX
3. Kiểm tra antenna cắm đúng cổng (TX→TX/RX, RX→RX2)
4. Đảm bảo TX và RX cùng `--freq`
5. Chạy `uhd_fft` để xem spectrum tại RX, đảm bảo thấy tín hiệu

### Tín hiệu TX quá mạnh (saturation)
- Giảm `--gain` xuống còn 15-20 dB
- Tăng khoảng cách

### PSNR = 0 (không decode được ảnh)
- File nhận bị corrupt → tăng SNR hoặc kiểm tra đồng bộ
- Thử chạy simulation trước để verify code

---

## PHẦN 7: KẾT QUẢ KỲ VỌNG

Dựa trên simulation với AWGN channel:

| SNR (dB) | PSNR Q=10 | PSNR Q=30 | PSNR Q=50 | PSNR Q=90 |
|----------|-----------|-----------|-----------|-----------|
| 5        | FAIL      | FAIL      | FAIL      | FAIL      |
| 10       | ~15-20    | FAIL      | FAIL      | FAIL      |
| 15       | ~25-28    | ~20-25    | FAIL      | FAIL      |
| 20       | ~28-30    | ~27-30    | ~25-28    | ~18-22    |
| 25       | ~29-30    | ~30-33    | ~30-33    | ~28-32    |
| 30       | ~30       | ~33-35    | ~34-37    | ~35-40    |
| 35+      | ~30       | ~34-36    | ~36-38    | ~38-42    |

**Quan sát quan trọng:**
- "Cliff effect": dưới SNR ngưỡng → mất hoàn toàn (PSNR ≈ 0)
- Q thấp (file nhỏ) → ít bị ảnh hưởng bởi kênh (ít bit cần truyền đúng)
- Q cao (file lớn) → cần SNR cao hơn để đảm bảo chất lượng

---

*Prepared for UET Electronics Engineering Lab*
*Contact: thầy hướng dẫn hoặc [johnw] nếu có vấn đề*
