//! Chapter 3: Low-Latency News, Macro Events & Lexicon Sentiment Scoring
//! 
//! Ultra-low-latency sentiment analysis using pre-compiled Aho-Corasick automaton.
//! Branchless programming for deterministic sub-10μs pipeline latency.

pub mod news_ingestor;
pub mod macro_calendar;
pub mod sentiment_scorer;
