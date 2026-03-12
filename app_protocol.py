"""
app_protocol.py - Application-layer packetization với sequence numbers
Dùng để detect và locate packet loss trong OFDM stream

Cấu trúc mỗi "application packet" (khớp với OFDM packet_len = 96 bytes):
  [seq_num: 2 bytes] [data_len: 2 bytes] [payload: 92 bytes]
  = 96 bytes total

Khi packet bị drop (CRC fail trong OFDM layer):
  - Receiver phát hiện gap qua seq_num discontinuity
  - Fill gap bằng zeros (concealment)

Ưu điểm vs không có seq numbers:
  - Packet bị mất ở GIỮA stream → không dịch offset toàn bộ data về sau
  - PSNR giảm đúng theo tỷ lệ packet loss (smooth)
"""

OFDM_PACKET_SIZE = 96  # bytes (phải khớp PACKET_LEN trong flowgraph)
APP_HEADER_SIZE  = 4   # 2 bytes seq_num + 2 bytes data_len
APP_PAYLOAD_SIZE = OFDM_PACKET_SIZE - APP_HEADER_SIZE  # = 92 bytes

WARMUP_SEQ = 0xFFFF  # Sequence number đặc biệt cho warmup packets


def pack_data(data: bytes, warmup_packets: int = 16) -> bytes:
    """
    Đóng gói data thành stream of 96-byte app packets, có warmup prefix.

    Warmup packets: [0xFFFF, 0x0000, 0x00 × 92] × warmup_packets
    Data packets:   [seq_num, data_len, payload(≤92)]

    Returns bytes sẵn sàng để truyền qua OFDM TX.
    """
    result = bytearray()

    # Warmup packets (cho Schmidl-Cox sync)
    warmup_pkt = WARMUP_SEQ.to_bytes(2, "big") + (0).to_bytes(2, "big") + bytes(APP_PAYLOAD_SIZE)
    for _ in range(warmup_packets):
        result += warmup_pkt

    # Data packets với sequence numbers
    seq = 0
    offset = 0
    while offset < len(data):
        chunk = data[offset: offset + APP_PAYLOAD_SIZE]
        data_len = len(chunk)
        # Pad chunk đến APP_PAYLOAD_SIZE nếu cần (last packet)
        chunk = chunk + bytes(APP_PAYLOAD_SIZE - data_len)

        pkt = seq.to_bytes(2, "big") + data_len.to_bytes(2, "big") + chunk
        assert len(pkt) == OFDM_PACKET_SIZE
        result += pkt

        seq += 1
        offset += APP_PAYLOAD_SIZE

    return bytes(result)


def unpack_data(received: bytes, expected_data_bytes: int,
                warmup_packets: int = 16) -> tuple:
    """
    Giải đóng gói received bytes thành data với packet loss concealment.

    Trả về (data: bytes, stats: dict)
    data: đúng expected_data_bytes, missing packets filled với 0x80 (gray)
    stats: thông tin packet loss
    """
    n_data_packets = (expected_data_bytes + APP_PAYLOAD_SIZE - 1) // APP_PAYLOAD_SIZE
    total_expected_bytes = expected_data_bytes

    # Parse received stream thành list of (seq, data_len, payload)
    received_pkts = {}  # seq_num → payload bytes
    warmup_count = 0

    n_rx_packets = len(received) // OFDM_PACKET_SIZE

    for i in range(n_rx_packets):
        pkt = received[i * OFDM_PACKET_SIZE: (i + 1) * OFDM_PACKET_SIZE]
        seq     = int.from_bytes(pkt[0:2], "big")
        datalen = int.from_bytes(pkt[2:4], "big")
        payload = pkt[4: 4 + datalen] if datalen > 0 else b""

        if seq == WARMUP_SEQ:
            warmup_count += 1
            continue

        if seq < 65535:  # valid data packet
            received_pkts[seq] = payload

    # Reconstruct data với concealment
    result = bytearray()
    received_count = 0
    lost_count = 0

    for seq in range(n_data_packets):
        if seq in received_pkts:
            payload = received_pkts[seq]
            # Calculate expected size for this packet
            remaining = expected_data_bytes - seq * APP_PAYLOAD_SIZE
            expected_payload = min(APP_PAYLOAD_SIZE, remaining)
            # Pad/trim to expected size
            payload = payload[:expected_payload]
            if len(payload) < expected_payload:
                payload = payload + bytes(expected_payload - len(payload))
            result += payload
            received_count += 1
        else:
            # Packet lost: fill với 0x80 (gray = mid-range pixel value)
            remaining = expected_data_bytes - seq * APP_PAYLOAD_SIZE
            fill_size = min(APP_PAYLOAD_SIZE, remaining)
            result += bytes([0x80] * fill_size)  # gray concealment
            lost_count += 1

    data = bytes(result[:expected_data_bytes])

    stats = {
        "total_data_packets": n_data_packets,
        "received_data_packets": received_count,
        "lost_data_packets": lost_count,
        "warmup_packets_rx": warmup_count,
        "packet_reception_rate": received_count / max(n_data_packets, 1),
    }

    return data, stats


def compute_n_packets(data_bytes: int) -> int:
    """Tính số data packets cần để truyền data_bytes."""
    return (data_bytes + APP_PAYLOAD_SIZE - 1) // APP_PAYLOAD_SIZE


def compute_tx_file_size(data_bytes: int, warmup_packets: int = 16) -> int:
    """Tính tổng số bytes của TX file (warmup + data packets)."""
    return (warmup_packets + compute_n_packets(data_bytes)) * OFDM_PACKET_SIZE
