#!/usr/bin/env bash

start=$1
end=$2

pycbc_inspiral \
--gps-start-time $start \
--gps-end-time $end \
--fake-strain aLIGOZeroDetHighPower \
--low-frequency-cutoff 15 \
--strain-high-pass 10 \
--sample-rate 2048 \
--injection-file "TestData/injections.hdf" \
--channel-name H1:CHANNEL \
\
--bank-file "MatchedFilter/template_bank.hdf" \
--output "MatchedFilter/output/${start}-${end}-out.hdf" \
\
--approximant IMRPhenomD \
\
--cluster-method window \
--cluster-function symmetric \
--cluster-window 1 \
\
--snr-threshold 3.8 \
--chisq-snr-threshold 5.25 \
--newsnr-threshold 3.8 \
--sgchisq-snr-threshold 6.0 \
--sgchisq-locations "mtotal>30:20-15,20-30,20-45,20-60,20-75,20-90,20-105,20-120" \
--chisq-bins 10 \
\
--keep-loudest-interval 1.072 \
--keep-loudest-num 100 \
--keep-loudest-log-chirp-window 0.4 \
\
--psd-estimation median \
--psd-segment-length 16 \
--psd-segment-stride 8 \
--psd-inverse-length 16 \
--psd-num-segments 63 \
\
--segment-length 512 \
--segment-start-pad 80 \
--segment-end-pad 16 \
--allow-zero-padding \
--verbose
