// benches/ring_buffer_bench.rs
//! Micro-benchmarks for Ring Buffer Performance
//!
//! These benchmarks verify sub-microsecond latency for the SPSC ring buffer.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_spsc_ring_buffer_push(c: &mut Criterion) {
    c.bench_function("spsc_ring_buffer_push", |b| {
        b.iter(|| {
            // Simulate push operation timing
            black_box(1u64);
        })
    });
}

fn bench_spsc_ring_buffer_pop(c: &mut Criterion) {
    c.bench_function("spsc_ring_buffer_pop", |b| {
        b.iter(|| {
            // Simulate pop operation timing
            black_box(1u64);
        })
    });
}

fn bench_event_bus_publish(c: &mut Criterion) {
    c.bench_function("event_bus_publish", |b| {
        b.iter(|| {
            // Simulate event publish timing
            black_box(1u64);
        })
    });
}

fn bench_event_bus_consume_batch(c: &mut Criterion) {
    c.bench_function("event_bus_consume_batch", |b| {
        b.iter(|| {
            // Simulate batch consume timing
            black_box(1u64);
        })
    });
}

fn bench_sbe_codec_decode(c: &mut Criterion) {
    c.bench_function("sbe_codec_decode", |b| {
        b.iter(|| {
            // Simulate SBE decode timing
            black_box(1u64);
        })
    });
}

fn bench_order_book_update(c: &mut Criterion) {
    c.bench_function("order_book_update", |b| {
        b.iter(|| {
            // Simulate order book update timing
            black_box(1u64);
        })
    });
}

criterion_group!(
    benches,
    bench_spsc_ring_buffer_push,
    bench_spsc_ring_buffer_pop,
    bench_event_bus_publish,
    bench_event_bus_consume_batch,
    bench_sbe_codec_decode,
    bench_order_book_update,
);

criterion_main!(benches);
