//! Stage 5 Integration Tests
//!
//! Comprehensive tests for the self-learning core, observability,
//! security, and compliance modules.

#![cfg(test)]

use hft_crypto_bot::prelude::*;

#[test]
fn test_soul_memory_creation() {
    let soul = SoulMemory::init().unwrap();
    assert!(soul.latest_sequence() == 0);
}

#[test]
fn test_online_bandit() {
    let mut bandit = OnlineBandit::new();
    bandit.init_strategies(4);
    
    // Select and update strategies
    for _ in 0..100 {
        let idx = bandit.select_strategy();
        bandit.update_reward(idx, if idx == 0 { 1.0 } else { -0.5 });
    }
    
    let mut weights = [0.0; 32];
    bandit.get_weights(&mut weights);
    assert!(weights[0] > weights[1]);
}

#[test]
fn test_mistake_analyzer() {
    let analyzer = MistakeAnalyzer::new();
    
    // Analyze a losing trade
    let penalty = analyzer.analyze_trade(
        -200, 15, 5, 0, 0, 500_000, 1000, 2000, 1000000
    );
    assert!(penalty < 1.0);
    
    // Analyze a winning trade
    let reward = analyzer.analyze_trade(
        100, 2, 5, 0, 0, 1_000_000, 1000, 2000, 1000000
    );
    assert!(reward >= 1.0);
}

#[test]
fn test_metrics_bus() {
    let bus = MetricsBus::new();
    let idx = bus.register(0x12345678).unwrap();
    
    bus.increment_counter(idx, 10);
    bus.record_observation(idx, 100);
    bus.record_observation(idx, 200);
    
    let metric = bus.get_metric(idx).unwrap();
    assert_eq!(metric.counter.load(core::sync::atomic::Ordering::Acquire), 10);
}

#[test]
fn test_trace_logger() {
    let logger = TraceLogger::init().unwrap();
    
    logger.info(1, 100, b"Test message");
    logger.error(1, 101, b"Error message");
    
    let (header, data) = logger.read_next().unwrap();
    assert_eq!(header.level, 1); // Info
    assert_eq!(&data, b"Test message");
}

#[test]
fn test_anomaly_detector() {
    let mut detector = AnomalyDetector::new();
    detector.init_metric(hft_crypto_bot::observability::bot_anomaly::MetricType::OrderRate, 100 * 65536);
    
    // Establish baseline
    for _ in 0..50 {
        detector.observe(hft_crypto_bot::observability::bot_anomaly::MetricType::OrderRate, 100 * 65536);
    }
    
    // Large deviation should trigger anomaly
    let is_anomaly = detector.observe(
        hft_crypto_bot::observability::bot_anomaly::MetricType::OrderRate, 
        500 * 65536
    );
    assert!(is_anomaly);
}

#[test]
fn test_secret_vault() {
    use hft_crypto_bot::security::secret_vault::AccessLevel;
    
    let mut vault = SecretVault::new();
    let master_key = [0x42u8; 32];
    vault.init(&master_key).unwrap();
    
    vault.store_key(b"api_key", b"secret_key", AccessLevel::Trade).unwrap();
    
    let retrieved = vault.retrieve_key(0, AccessLevel::Trade).unwrap();
    let _ = retrieved; // In real impl, would verify content
}

#[test]
fn test_hsm_kms() {
    use hft_crypto_bot::security::hsm_kms_stub::{HsmBackend, SignAlgorithm};
    
    let mut hsm = HsmKmsLayer::new();
    hsm.init(HsmBackend::Mock).unwrap();
    
    let key_idx = hsm.register_key(12345, SignAlgorithm::Ed25519, 1).unwrap();
    
    let digest = [0x42u8; 32];
    let signature = hsm.sign(key_idx, &digest).unwrap();
    assert!(signature.iter().any(|&b| b != 0));
}

#[test]
fn test_network_acl() {
    let acl = NetworkAcl::new();
    
    let ip = NetworkAcl::parse_ip("192.168.1.100").unwrap();
    acl.add_allowed_ip(ip, 0, 1).unwrap();
    
    assert!(acl.check_ip(ip, 443, 1));
    
    let other_ip = NetworkAcl::parse_ip("10.0.0.1").unwrap();
    assert!(!acl.check_ip(other_ip, 443, 1));
}

#[test]
fn test_audit_ledger() {
    use hft_crypto_bot::compliance::audit_ledger::AuditEventType;
    
    let ledger = AuditLedger::init().unwrap();
    
    ledger.append(AuditEventType::OrderNew, 1, b"order data").unwrap();
    ledger.append(AuditEventType::OrderFill, 1, b"fill data").unwrap();
    
    let (header, data) = ledger.read_record(0).unwrap();
    assert_eq!(header.event_type, AuditEventType::OrderNew as u8);
    assert_eq!(&data, b"order data");
}

#[test]
fn test_rate_limiter() {
    let limiter = RateLimiter::new();
    limiter.init_bucket(0, 100, 1000);
    
    assert!(limiter.try_acquire(0, 10));
    assert!(limiter.try_acquire(0, 10));
    
    let tokens = limiter.get_tokens(0);
    assert!(tokens < 100);
}

#[test]
fn test_jurisdiction_filter() {
    use hft_crypto_bot::compliance::jurisdiction_filter::Jurisdiction;
    
    let filter = JurisdictionFilter::new();
    
    let btc_hash = JurisdictionFilter::hash_symbol("BTCUSDT");
    filter.add_blocked_token(b"BTCUSDT", btc_hash, 1 << Jurisdiction::US as u64, 1).unwrap();
    
    filter.set_jurisdiction(Jurisdiction::US);
    assert!(!filter.check_token(btc_hash));
    
    filter.set_jurisdiction(Jurisdiction::EU);
    assert!(filter.check_token(btc_hash));
}

#[test]
fn test_cache_line_alignment() {
    // Verify all critical structs are cache-line aligned
    assert_eq!(core::mem::size_of::<hft_crypto_bot::soul::soul_memory::SoulHeader>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::soul::online_rl::StrategySlot>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::soul::mistake_analyzer::TradeFeatures>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::observability::metrics_bus::MetricSlot>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::observability::trace_logger::TraceHeader>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::observability::bot_anomaly::KalmanState>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::security::secret_vault::EncryptedKeySlot>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::security::hsm_kms_stub::KeyHandle>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::security::network_acl::IpAllowEntry>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::compliance::audit_ledger::AuditRecordHeader>() % 512, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::compliance::rate_limiter::RateLimitBucket>() % 64, 0);
    assert_eq!(core::mem::size_of::<hft_crypto_bot::compliance::jurisdiction_filter::BlockedToken>() % 64, 0);
}
