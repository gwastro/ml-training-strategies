#!/usr/bin/env bash

pycbc_brute_bank \
--verbose \
--output-file MatchedFilter/template_bank.hdf \
--minimal-match 0.97 \
--tolerance .005 \
--buffer-length 16 \
--sample-rate 2048 \
--tau0-threshold 0.5 \
--approximant SEOBNRv4_opt \
--tau0-crawl 5 \
--tau0-start 0 \
--tau0-end 50 \
--psd-model aLIGOZeroDetHighPower \
--min 10 10 0 0 \
--max 50 50 0.01 0.01 \
--params mass1 mass2 spin1z spin2z \
--seed 1 \
--low-frequency-cutoff 15
