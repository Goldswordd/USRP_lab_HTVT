#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ofdm_tx_headless.py - OFDM Transmitter không có GUI
Dựa trên code của khóa luận cũ (cgminh), headless version để sweep tự động

Sử dụng:
    python3 ofdm_tx_headless.py --input INPUT_FILE --output IQ_FILE --snr SNR_DB

Flowgraph:
    file_source → stream_to_tagged_stream → crc32
        → [header path] repack → BPSK symbols ─┐
        → [payload path] repack → QPSK symbols ─┤
                                                 ├→ tagged_stream_mux
                                                      → carrier_allocator → IFFT → CP_add
                                                      → scale(0.05) → channel_model(SNR)
                                                      → file_sink (IQ complex)
"""
import os
import sys
import argparse

from gnuradio import blocks, channels, digital, fft as gr_fft, gr
from gnuradio.fft import window
import pmt


# ─────────────── OFDM Parameters ───────────────────────────────────────────
FFT_LEN       = 64
CP_LEN        = FFT_LEN // 4        # = 16
PACKET_LEN    = 96                  # bytes per OFDM packet
PKT_TAG       = "packet_len"        # byte-count tag

OCCUPIED_CARRIERS = [
    list(range(-26, -21)) + list(range(-20, -7)) +
    list(range(-6, 0))    + list(range(1, 7))    +
    list(range(8, 21))    + list(range(22, 27))
]  # 48 subcarriers

PILOT_CARRIERS = ((-21, -7, 7, 21,),)
PILOT_SYMBOLS  = ((1, 1, 1, -1,),)

# Schmidl-Cox sync words (từ IEEE 802.11a-based OFDM trong GNU Radio)
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
# ─────────────────────────────────────────────────────────────────────────────


class OFDMTransmitter(gr.top_block):
    """
    OFDM Transmitter không GUI.
    Đọc bytes từ file → OFDM encode → thêm AWGN → lưu IQ ra file.
    """

    def __init__(self, input_file: str, output_file: str, snr_db: float):
        gr.top_block.__init__(self, "OFDM_TX_Headless")

        # ── Modulation ──────────────────────────────────────────────────────
        payload_mod = digital.constellation_qpsk()   # 2 bits/symbol
        header_mod  = digital.constellation_bpsk()   # 1 bit/symbol

        # ── Header formatter ────────────────────────────────────────────────
        hdr_format = digital.header_format_ofdm(OCCUPIED_CARRIERS, 1, PKT_TAG)

        # ── Noise amplitude (đã hiệu chỉnh theo signal RMS đo được)
        # Signal RMS sau toàn bộ OFDM chain (đo thực tế) ≈ 0.58
        # (cao hơn scale(0.05) vì sync words có amplitude √2 × nhiều subcarriers)
        # SNR (power dB) = 20*log10(signal_rms / noise_vol) = snr_db
        # → noise_vol = 0.58 × 10^(-snr_db/20)
        SIGNAL_RMS = 0.58
        noise_vol = SIGNAL_RMS * 10 ** (-snr_db / 20.0)

        # ── Blocks ──────────────────────────────────────────────────────────
        self.file_src = blocks.file_source(
            gr.sizeof_char, input_file, False  # repeat=False → dừng khi hết file
        )
        self.file_src.set_begin_tag(pmt.PMT_NIL)

        self.s2ts = blocks.stream_to_tagged_stream(
            gr.sizeof_char, 1, PACKET_LEN, PKT_TAG
        )
        self.crc_tx = digital.crc32_bb(False, PKT_TAG, True)  # False=append, not check

        # Header path
        self.proto_fmt   = digital.protocol_formatter_bb(hdr_format, PKT_TAG)
        self.repack_hdr  = blocks.repack_bits_bb(
            8, header_mod.bits_per_symbol(), PKT_TAG, False, gr.GR_LSB_FIRST
        )
        self.map_hdr = digital.chunks_to_symbols_bc(header_mod.points(), 1)

        # Payload path
        self.repack_pld  = blocks.repack_bits_bb(
            8, payload_mod.bits_per_symbol(), PKT_TAG, False, gr.GR_LSB_FIRST
        )
        self.map_pld = digital.chunks_to_symbols_bc(payload_mod.points(), 1)

        # OFDM mux + encoding
        self.mux = blocks.tagged_stream_mux(gr.sizeof_gr_complex * 1, PKT_TAG, 0)
        self.carrier_alloc = digital.ofdm_carrier_allocator_cvc(
            FFT_LEN, OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS,
            (SYNC_WORD1, SYNC_WORD2), PKT_TAG, True
        )
        self.ifft     = gr_fft.fft_vcc(FFT_LEN, False, (), True, 1)
        self.cp_add   = digital.ofdm_cyclic_prefixer(
            FFT_LEN, FFT_LEN + CP_LEN, 0, PKT_TAG
        )
        self.scale    = blocks.multiply_const_cc(0.05)
        self.tag_gate = blocks.tag_gate(gr.sizeof_gr_complex * 1, False)

        # Channel model (AWGN, đơn giản)
        self.channel = channels.channel_model(
            noise_voltage=noise_vol,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0 + 0.0j],   # Pure AWGN, không phase offset (code gốc dùng 1+1j → phase 45° gây lỗi sync)
            noise_seed=42,
            block_tags=False
        )

        # Output IQ file
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex * 1, output_file, False
        )
        self.file_sink.set_unbuffered(False)

        # ── Connections ─────────────────────────────────────────────────────
        # Source → tagged packets → CRC
        self.connect(self.file_src, self.s2ts, self.crc_tx)

        # Payload branch
        self.connect(self.crc_tx, self.repack_pld, self.map_pld)
        self.connect((self.map_pld, 0), (self.mux, 1))

        # Header branch
        self.connect(self.crc_tx, self.proto_fmt, self.repack_hdr, self.map_hdr)
        self.connect((self.map_hdr, 0), (self.mux, 0))

        # OFDM chain
        self.connect(
            self.mux, self.carrier_alloc, self.ifft,
            self.cp_add, self.scale, self.tag_gate,
            self.channel, self.file_sink
        )


def main():
    parser = argparse.ArgumentParser(description="OFDM TX headless (AWGN simulation)")
    parser.add_argument("--input",  required=True, help="File đầu vào (bytes)")
    parser.add_argument("--output", required=True, help="File đầu ra (IQ complex float)")
    parser.add_argument("--snr",    type=float, default=20.0, help="SNR (dB), mặc định 20")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERR] Không tìm thấy file đầu vào: {args.input}")
        sys.exit(1)

    file_size = os.path.getsize(args.input)
    print(f"[TX] Input: {args.input} ({file_size} bytes)")
    print(f"[TX] SNR: {args.snr:.1f} dB | Output IQ: {args.output}")

    tb = OFDMTransmitter(args.input, args.output, args.snr)
    tb.run()  # Blocks until file_source exhausted

    iq_size = os.path.getsize(args.output) if os.path.exists(args.output) else 0
    print(f"[TX] Done. IQ file: {iq_size:,} bytes ({iq_size//8:,} complex samples)")


if __name__ == "__main__":
    main()
