#!/bin/sh
cargo test
cargo build --release
time ./target/release/deep-learning
python3 plot.py
