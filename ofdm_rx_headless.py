#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ofdm_rx_headless.py - OFDM Receiver không có GUI
BUG FIX từ code gốc: payload_mod được đổi từ 16QAM → QPSK (khớp với TX)
                      costas_loop order: 16 → 4

Sử dụng:
    python3 ofdm_rx_headless.py --input IQ_FILE --output OUT_FILE [--timeout 30]

Flowgraph:
    file_source(IQ) → ofdm_sync → freq_correction → header_payload_demux
        → [header]  FFT → chanest → equalize → serialize → costas(2) → BPSK decode → header_parser
        → [payload] FFT →           equalize → serialize → costas(4) → QPSK decode
                                                             → repack → CRC check → file_sink
"""
import os
import sys
import argparse
import time
import threading

from gnuradio import analog, blocks, digital, fft as gr_fft, gr
import pmt


# ─────────────── OFDM Parameters (phải khớp với TX) ─────────────────────────
FFT_LEN    = 64
CP_LEN     = FFT_LEN // 4
PACKET_LEN = 96
PKT_TAG    = "packet_len"
FRAME_TAG  = "frame_len"   # Tag key nội bộ của OFDM demux
PHASE_BW   = 0.0628        # Costas loop bandwidth

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
# ─────────────────────────────────────────────────────────────────────────────


class OFDMReceiver(gr.top_block):
    """
    OFDM Receiver không GUI.
    Đọc IQ complex từ file → OFDM decode → lưu bytes ra file.

    FIX so với code gốc:
        - payload_mod: 16QAM → QPSK  (khớp với TX)
        - costas order: 16 → 4
    """

    def __init__(self, input_file: str, output_file: str, samp_rate: int = 100_000):
        gr.top_block.__init__(self, "OFDM_RX_Headless")

        # ── Modulation (PHẢI khớp với TX) ───────────────────────────────────
        payload_mod = digital.constellation_qpsk()   # FIX: gốc là 16QAM
        header_mod  = digital.constellation_bpsk()

        # ── Equalizers ──────────────────────────────────────────────────────
        header_equalizer = digital.ofdm_equalizer_simpledfe(
            FFT_LEN, header_mod.base(), OCCUPIED_CARRIERS,
            PILOT_CARRIERS, PILOT_SYMBOLS
        )
        payload_equalizer = digital.ofdm_equalizer_simpledfe(
            FFT_LEN, payload_mod.base(), OCCUPIED_CARRIERS,
            PILOT_CARRIERS, PILOT_SYMBOLS, 1
        )

        # ── Header formatter (dùng để parse header) ─────────────────────────
        header_formatter = digital.packet_header_ofdm(
            OCCUPIED_CARRIERS, n_syms=1,
            len_tag_key=PKT_TAG,
            frame_len_tag_key=FRAME_TAG,
            bits_per_header_sym=header_mod.bits_per_symbol(),
            bits_per_payload_sym=payload_mod.bits_per_symbol(),
            scramble_header=False
        )

        # ── Blocks ──────────────────────────────────────────────────────────
        # Source: đọc IQ complex từ file TX
        self.file_src = blocks.file_source(
            gr.sizeof_gr_complex * 1, input_file, False
        )
        self.file_src.set_begin_tag(pmt.PMT_NIL)

        # Synchronization
        self.delay = blocks.delay(gr.sizeof_gr_complex * 1, FFT_LEN + CP_LEN)
        self.sync  = digital.ofdm_sync_sc_cfb(FFT_LEN, CP_LEN, False, 0.9)

        # Frequency correction
        self.freq_mod  = analog.frequency_modulator_fc(-2.0 / FFT_LEN)
        self.multiply  = blocks.multiply_vcc(1)

        # Header/Payload demux
        self.hpd = digital.header_payload_demux(
            3,              # header_len (OFDM symbols)
            FFT_LEN,        # items_per_symbol
            CP_LEN,         # guard_interval
            FRAME_TAG,      # length_tag_key
            "",             # trigger_tag_key
            True,           # output_symbols (complex)
            gr.sizeof_gr_complex,
            "rx_time",
            samp_rate,
            (),
            0
        )

        # FFT for header and payload
        self.fft_hdr = gr_fft.fft_vcc(FFT_LEN, True, (), True, 1)
        self.fft_pld = gr_fft.fft_vcc(FFT_LEN, True, (), True, 1)

        # Channel estimation (từ sync words)
        self.chanest = digital.ofdm_chanest_vcvc(
            SYNC_WORD1, SYNC_WORD2, 1, 0, 3, False
        )

        # Frame equalizers
        self.eq_hdr = digital.ofdm_frame_equalizer_vcvc(
            header_equalizer.base(), CP_LEN, FRAME_TAG, True, 1
        )
        self.eq_pld = digital.ofdm_frame_equalizer_vcvc(
            payload_equalizer.base(), CP_LEN, FRAME_TAG, True, 0
        )

        # Serializers (vector → stream)
        self.ser_hdr = digital.ofdm_serializer_vcc(
            FFT_LEN, OCCUPIED_CARRIERS, FRAME_TAG, "", 0, "", True
        )
        self.ser_pld = digital.ofdm_serializer_vcc(
            FFT_LEN, OCCUPIED_CARRIERS, FRAME_TAG, PKT_TAG, 1, "", True
        )

        # Costas loops (phase tracking)
        self.costas_hdr = digital.costas_loop_cc(PHASE_BW, 2, False)   # BPSK
        self.costas_pld = digital.costas_loop_cc(PHASE_BW, 4, False)   # QPSK (FIX: gốc là 16)

        # Demodulation
        self.dec_hdr = digital.constellation_decoder_cb(header_mod.base())
        self.dec_pld = digital.constellation_decoder_cb(payload_mod.base())

        # Header parser → sends frame length to demux via message
        self.hdr_parser = digital.packet_headerparser_b(header_formatter.base())

        # Repack bits + CRC check
        self.repack_rx = blocks.repack_bits_bb(
            payload_mod.bits_per_symbol(), 8, PKT_TAG, True, gr.GR_LSB_FIRST
        )
        self.crc_rx = digital.crc32_bb(True, PKT_TAG, True)  # True=check, not append

        # Output
        self.file_sink = blocks.file_sink(gr.sizeof_char * 1, output_file, False)
        self.file_sink.set_unbuffered(True)  # Flush ngay để monitor output

        # ── Connections ─────────────────────────────────────────────────────

        # Sync: split signal → delay path + sync path
        self.connect(self.file_src, self.delay)
        self.connect(self.file_src, self.sync)

        # Frequency correction
        self.connect((self.sync, 0), self.freq_mod)
        self.connect((self.freq_mod, 0), (self.multiply, 0))
        self.connect((self.delay, 0),    (self.multiply, 1))

        # Header/Payload demux input
        self.connect((self.multiply, 0), (self.hpd, 0))
        self.connect((self.sync, 1),     (self.hpd, 1))   # trigger port

        # Header path: hpd[0] → FFT → chanest → eq → serialize → costas → decode → parse
        self.connect((self.hpd, 0), self.fft_hdr, self.chanest,
                     self.eq_hdr, self.ser_hdr, self.costas_hdr,
                     self.dec_hdr, self.hdr_parser)

        # Payload path: hpd[1] → FFT → eq → serialize → costas → decode → repack → CRC → sink
        self.connect((self.hpd, 1), self.fft_pld, self.eq_pld,
                     self.ser_pld, self.costas_pld, self.dec_pld,
                     self.repack_rx, self.crc_rx, self.file_sink)

        # Message: header_parser → header_payload_demux
        self.msg_connect(
            (self.hdr_parser, "header_data"),
            (self.hpd, "header_data")
        )


def run_receiver(input_file: str, output_file: str, timeout: float = 30.0,
                 samp_rate: int = 100_000) -> int:
    """
    Chạy receiver với timeout.
    Trả về số bytes nhận được.
    """
    tb = OFDMReceiver(input_file, output_file, samp_rate)
    tb.start()

    # Monitor output file size để biết khi nào xong
    def _stopper():
        time.sleep(timeout)
        tb.stop()

    t = threading.Thread(target=_stopper, daemon=True)
    t.start()

    tb.wait()
    t.join(0)

    received = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    return received


def main():
    parser = argparse.ArgumentParser(description="OFDM RX headless")
    parser.add_argument("--input",   required=True,        help="File IQ complex (từ TX)")
    parser.add_argument("--output",  required=True,        help="File output bytes")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout (giây)")
    parser.add_argument("--samp-rate", type=int, default=100_000, help="Sample rate (Hz)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERR] Không tìm thấy IQ file: {args.input}")
        sys.exit(1)

    iq_size = os.path.getsize(args.input)
    print(f"[RX] IQ input: {args.input} ({iq_size:,} bytes)")
    print(f"[RX] Output: {args.output}")
    print(f"[RX] Timeout: {args.timeout}s")

    received = run_receiver(args.input, args.output, args.timeout, args.samp_rate)
    print(f"[RX] Done. Nhận được: {received:,} bytes")


if __name__ == "__main__":
    main()
