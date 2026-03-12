#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usrp_tx.py - OFDM Transmitter cho USRP N210 thật
Dùng tại lab: một máy tính kết nối với USRP TX

Sử dụng:
    python3 usrp_tx.py --input INPUT_FILE [--addr 192.168.10.2] [--freq 2.4e9] [--gain 30]

Thay đổi so với simulation:
    - Bỏ channel_model + file_sink
    - Thêm uhd.usrp_sink (USRP N210)
    - Sample rate: 1 MHz (ổn định với N210 qua GigE)
    - Thêm head block để giới hạn số samples (tránh loop vô hạn)
"""
import os
import sys
import argparse
import time
import threading

from gnuradio import blocks, digital, fft as gr_fft, gr, uhd
import pmt

# ─── Parameters ─────────────────────────────────────────────────────────────
FFT_LEN    = 64
CP_LEN     = FFT_LEN // 4
PACKET_LEN = 96
PKT_TAG    = "packet_len"
SAMP_RATE  = 1_000_000      # 1 MHz (an toàn với N210 qua GigE)
TX_GAIN_DB = 30             # Có thể chỉnh, tùy khoảng cách TX-RX

OCCUPIED_CARRIERS = [
    list(range(-26, -21)) + list(range(-20, -7)) +
    list(range(-6, 0))    + list(range(1, 7))    +
    list(range(8, 21))    + list(range(22, 27))
]
PILOT_CARRIERS = ((-21, -7, 7, 21,),)
PILOT_SYMBOLS  = ((1, 1, 1, -1,),)

SYNC_WORD1 = [
    0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356,
    0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356,
    0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356,
    0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356,
    0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356,
    0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356,
    0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356,
    0., 0., 0., 0., 0., 0.
]
SYNC_WORD2 = [
    0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1,
   -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1,
    0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1,
    1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0
]


class USRPTransmitter(gr.top_block):
    """
    OFDM TX dùng USRP N210.
    Phát ảnh JPEG nén ra không khí tại tần số center_freq.
    """

    def __init__(self, input_file: str, usrp_addr: str = "192.168.10.2",
                 center_freq: float = 2.4e9, tx_gain: float = TX_GAIN_DB,
                 samp_rate: int = SAMP_RATE):
        gr.top_block.__init__(self, "USRP_OFDM_TX")

        payload_mod = digital.constellation_qpsk()
        header_mod  = digital.constellation_bpsk()
        hdr_format  = digital.header_format_ofdm(OCCUPIED_CARRIERS, 1, PKT_TAG)

        # ── Source ──────────────────────────────────────────────────────────
        self.file_src = blocks.file_source(
            gr.sizeof_char, input_file,
            True   # repeat=True → phát lặp lại để RX có đủ thời gian sync
        )
        self.file_src.set_begin_tag(pmt.PMT_NIL)

        # Giới hạn để không phát mãi mãi:
        # ~500 packets là đủ cho ảnh nhỏ + thời gian sync
        file_size  = os.path.getsize(input_file)
        n_repeats  = max(10, 2048 * PACKET_LEN // file_size)  # ~2MB data tổng
        head_bytes = file_size * n_repeats
        self.head  = blocks.head(gr.sizeof_char, head_bytes)

        # ── TX chain (giống ofdm_tx_headless.py) ────────────────────────────
        self.s2ts       = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, PACKET_LEN, PKT_TAG)
        self.crc_tx     = digital.crc32_bb(False, PKT_TAG, True)
        self.proto_fmt  = digital.protocol_formatter_bb(hdr_format, PKT_TAG)
        self.repack_hdr = blocks.repack_bits_bb(8, header_mod.bits_per_symbol(), PKT_TAG, False, gr.GR_LSB_FIRST)
        self.map_hdr    = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.repack_pld = blocks.repack_bits_bb(8, payload_mod.bits_per_symbol(), PKT_TAG, False, gr.GR_LSB_FIRST)
        self.map_pld    = digital.chunks_to_symbols_bc(payload_mod.points(), 1)
        self.mux        = blocks.tagged_stream_mux(gr.sizeof_gr_complex * 1, PKT_TAG, 0)
        self.carrier_alloc = digital.ofdm_carrier_allocator_cvc(
            FFT_LEN, OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS,
            (SYNC_WORD1, SYNC_WORD2), PKT_TAG, True
        )
        self.ifft   = gr_fft.fft_vcc(FFT_LEN, False, (), True, 1)
        self.cp_add = digital.ofdm_cyclic_prefixer(FFT_LEN, FFT_LEN + CP_LEN, 0, PKT_TAG)
        self.scale  = blocks.multiply_const_cc(0.05)

        # ── USRP Sink ────────────────────────────────────────────────────────
        self.usrp_sink = uhd.usrp_sink(
            f"addr={usrp_addr}",
            uhd.stream_args(cpu_format="fc32", args="", channels=[0]),
            PKT_TAG,
        )
        self.usrp_sink.set_samp_rate(samp_rate)
        self.usrp_sink.set_center_freq(center_freq, 0)
        self.usrp_sink.set_gain(tx_gain, 0)
        self.usrp_sink.set_antenna("TX/RX", 0)

        print(f"[USRP TX] Freq: {center_freq/1e9:.3f} GHz | "
              f"Rate: {samp_rate/1e6:.1f} MHz | Gain: {tx_gain} dB | "
              f"Addr: {usrp_addr}")

        # ── Connections ──────────────────────────────────────────────────────
        self.connect(self.file_src, self.head, self.s2ts, self.crc_tx)
        self.connect(self.crc_tx, self.repack_pld, self.map_pld)
        self.connect((self.map_pld, 0), (self.mux, 1))
        self.connect(self.crc_tx, self.proto_fmt, self.repack_hdr, self.map_hdr)
        self.connect((self.map_hdr, 0), (self.mux, 0))
        self.connect(self.mux, self.carrier_alloc, self.ifft,
                     self.cp_add, self.scale, self.usrp_sink)


def main():
    parser = argparse.ArgumentParser(description="USRP N210 OFDM Transmitter")
    parser.add_argument("--input", required=True,           help="File ảnh JPEG đầu vào")
    parser.add_argument("--addr",  default="192.168.10.2",  help="IP của USRP TX")
    parser.add_argument("--freq",  type=float, default=2.4e9, help="Tần số trung tâm (Hz)")
    parser.add_argument("--gain",  type=float, default=30.0,  help="TX gain (dB)")
    parser.add_argument("--rate",  type=int,   default=SAMP_RATE, help="Sample rate (Hz)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERR] Không tìm thấy file: {args.input}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"USRP OFDM Transmitter")
    print(f"{'='*50}")
    print(f"Input: {args.input} ({os.path.getsize(args.input):,} bytes)")
    print(f"USRP addr: {args.addr}")
    print(f"Ấn Ctrl+C để dừng\n")

    tb = USRPTransmitter(
        input_file  = args.input,
        usrp_addr   = args.addr,
        center_freq = args.freq,
        tx_gain     = args.gain,
        samp_rate   = args.rate,
    )

    tb.start()
    print("[TX] Đang phát... Ctrl+C để dừng")
    try:
        tb.wait()
    except KeyboardInterrupt:
        tb.stop()
        tb.wait()
    print("[TX] Dừng.")


if __name__ == "__main__":
    main()
