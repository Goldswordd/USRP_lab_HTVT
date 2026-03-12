#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usrp_rx.py - OFDM Receiver cho USRP N210 thật
Dùng tại lab: một máy tính kết nối với USRP RX

Sử dụng:
    python3 usrp_rx.py --output OUTPUT_FILE [--addr 192.168.10.3] [--freq 2.4e9] [--gain 30]

Sau khi nhận xong, đo PSNR với ảnh gốc:
    python3 measure_psnr.py --original ORIGINAL.jpg --received OUTPUT_FILE
"""
import os
import sys
import argparse
import time
import signal

from gnuradio import analog, blocks, digital, fft as gr_fft, gr, uhd
import pmt

# ─── Parameters (phải khớp với usrp_tx.py) ───────────────────────────────
FFT_LEN    = 64
CP_LEN     = FFT_LEN // 4
PACKET_LEN = 96
PKT_TAG    = "packet_len"
FRAME_TAG  = "frame_len"
PHASE_BW   = 0.0628
SAMP_RATE  = 1_000_000
RX_GAIN_DB = 30

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


class USRPReceiver(gr.top_block):
    """
    OFDM RX dùng USRP N210.
    Thu tín hiệu từ không khí → OFDM decode → lưu bytes.
    """

    def __init__(self, output_file: str, usrp_addr: str = "192.168.10.3",
                 center_freq: float = 2.4e9, rx_gain: float = RX_GAIN_DB,
                 samp_rate: int = SAMP_RATE):
        gr.top_block.__init__(self, "USRP_OFDM_RX")

        payload_mod = digital.constellation_qpsk()   # QPSK (khớp TX)
        header_mod  = digital.constellation_bpsk()

        header_equalizer = digital.ofdm_equalizer_simpledfe(
            FFT_LEN, header_mod.base(), OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS
        )
        payload_equalizer = digital.ofdm_equalizer_simpledfe(
            FFT_LEN, payload_mod.base(), OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS, 1
        )
        header_formatter = digital.packet_header_ofdm(
            OCCUPIED_CARRIERS, n_syms=1,
            len_tag_key=PKT_TAG,
            frame_len_tag_key=FRAME_TAG,
            bits_per_header_sym=header_mod.bits_per_symbol(),
            bits_per_payload_sym=payload_mod.bits_per_symbol(),
            scramble_header=False
        )

        # ── USRP Source ──────────────────────────────────────────────────────
        self.usrp_src = uhd.usrp_source(
            f"addr={usrp_addr}",
            uhd.stream_args(cpu_format="fc32", args="", channels=[0]),
        )
        self.usrp_src.set_samp_rate(samp_rate)
        self.usrp_src.set_center_freq(center_freq, 0)
        self.usrp_src.set_gain(rx_gain, 0)
        self.usrp_src.set_antenna("RX2", 0)

        print(f"[USRP RX] Freq: {center_freq/1e9:.3f} GHz | "
              f"Rate: {samp_rate/1e6:.1f} MHz | Gain: {rx_gain} dB | "
              f"Addr: {usrp_addr}")

        # ── RX Chain (giống ofdm_rx_headless.py) ────────────────────────────
        self.delay    = blocks.delay(gr.sizeof_gr_complex * 1, FFT_LEN + CP_LEN)
        self.sync     = digital.ofdm_sync_sc_cfb(FFT_LEN, CP_LEN, False, 0.9)
        self.freq_mod = analog.frequency_modulator_fc(-2.0 / FFT_LEN)
        self.multiply = blocks.multiply_vcc(1)

        self.hpd = digital.header_payload_demux(
            3, FFT_LEN, CP_LEN, FRAME_TAG, "", True,
            gr.sizeof_gr_complex, "rx_time", samp_rate, (), 0
        )

        self.fft_hdr  = gr_fft.fft_vcc(FFT_LEN, True, (), True, 1)
        self.fft_pld  = gr_fft.fft_vcc(FFT_LEN, True, (), True, 1)
        self.chanest  = digital.ofdm_chanest_vcvc(SYNC_WORD1, SYNC_WORD2, 1, 0, 3, False)
        self.eq_hdr   = digital.ofdm_frame_equalizer_vcvc(header_equalizer.base(),  CP_LEN, FRAME_TAG, True, 1)
        self.eq_pld   = digital.ofdm_frame_equalizer_vcvc(payload_equalizer.base(), CP_LEN, FRAME_TAG, True, 0)
        self.ser_hdr  = digital.ofdm_serializer_vcc(FFT_LEN, OCCUPIED_CARRIERS, FRAME_TAG, "", 0, "", True)
        self.ser_pld  = digital.ofdm_serializer_vcc(FFT_LEN, OCCUPIED_CARRIERS, FRAME_TAG, PKT_TAG, 1, "", True)
        self.costas_hdr = digital.costas_loop_cc(PHASE_BW, 2, False)
        self.costas_pld = digital.costas_loop_cc(PHASE_BW, 4, False)
        self.dec_hdr  = digital.constellation_decoder_cb(header_mod.base())
        self.dec_pld  = digital.constellation_decoder_cb(payload_mod.base())
        self.hdr_parser = digital.packet_headerparser_b(header_formatter.base())
        self.repack_rx  = blocks.repack_bits_bb(payload_mod.bits_per_symbol(), 8, PKT_TAG, True, gr.GR_LSB_FIRST)
        self.crc_rx     = digital.crc32_bb(True, PKT_TAG, True)

        self.file_sink = blocks.file_sink(gr.sizeof_char * 1, output_file, False)
        self.file_sink.set_unbuffered(True)

        # ── Connections ──────────────────────────────────────────────────────
        self.connect(self.usrp_src, self.delay)
        self.connect(self.usrp_src, self.sync)
        self.connect((self.sync, 0), self.freq_mod)
        self.connect((self.freq_mod, 0), (self.multiply, 0))
        self.connect((self.delay, 0),    (self.multiply, 1))
        self.connect((self.multiply, 0), (self.hpd, 0))
        self.connect((self.sync, 1),     (self.hpd, 1))
        self.connect((self.hpd, 0), self.fft_hdr, self.chanest,
                     self.eq_hdr, self.ser_hdr, self.costas_hdr,
                     self.dec_hdr, self.hdr_parser)
        self.connect((self.hpd, 1), self.fft_pld, self.eq_pld,
                     self.ser_pld, self.costas_pld, self.dec_pld,
                     self.repack_rx, self.crc_rx, self.file_sink)
        self.msg_connect(
            (self.hdr_parser, "header_data"),
            (self.hpd, "header_data")
        )


def main():
    parser = argparse.ArgumentParser(description="USRP N210 OFDM Receiver")
    parser.add_argument("--output", required=True,           help="File output bytes")
    parser.add_argument("--addr",   default="192.168.10.3",  help="IP của USRP RX")
    parser.add_argument("--freq",   type=float, default=2.4e9, help="Tần số trung tâm (Hz)")
    parser.add_argument("--gain",   type=float, default=30.0,  help="RX gain (dB)")
    parser.add_argument("--rate",   type=int,   default=SAMP_RATE, help="Sample rate (Hz)")
    parser.add_argument("--time",   type=float, default=60.0,  help="Thời gian thu (giây)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"USRP OFDM Receiver")
    print(f"{'='*50}")
    print(f"Output: {args.output}")
    print(f"USRP addr: {args.addr}")
    print(f"Thu trong {args.time}s, hoặc Ctrl+C để dừng sớm\n")

    tb = USRPReceiver(
        output_file = args.output,
        usrp_addr   = args.addr,
        center_freq = args.freq,
        rx_gain     = args.gain,
        samp_rate   = args.rate,
    )

    tb.start()
    print(f"[RX] Đang thu... Chờ {args.time}s")

    try:
        time.sleep(args.time)
    except KeyboardInterrupt:
        print("\n[RX] Interrupted by user")

    tb.stop()
    tb.wait()

    if os.path.exists(args.output):
        size = os.path.getsize(args.output)
        print(f"[RX] Đã nhận: {size:,} bytes → {args.output}")
        print(f"\n[*] Đo PSNR: python3 measure_psnr.py --original ORIGINAL.jpg --received {args.output}")
    else:
        print("[RX] Không nhận được dữ liệu!")


if __name__ == "__main__":
    main()
