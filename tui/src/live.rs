//! The live-training words and the daemon's mailbox words of the shared part (lib/cx-proto.h), read from
//! /dev/shm/phi-hostmem by two small positioned reads per poll: no mapping, read-only, nothing written; and the
//! state both programs keep of them (`LiveState`): the words as last read, whether the daemon's heartbeat moves,
//! the read error.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::time::{Duration, Instant};

/// CX_PART_MIB: the part's offset in the host-memory region.
pub const PART_OFFSET: u64 = 1344 << 20;
/// CX_T_BASE: the live-training block inside the part.
const T_BASE: u64 = 55 << 20;
/// Through CX_T_SAVES (CX_T_BASE + 1536, u64).
const T_LEN: usize = 1536 + 8;
/// CX_T_MAGIC_VAL "TRNR".
const T_MAGIC_VAL: u32 = 0x524e_5254;
/// Through CX_C_HEART (256, u64).
const MBOX_LEN: usize = 264;
const LIVE_POLL: Duration = Duration::from_millis(250);
/// The daemon is called absent when its heartbeat has not moved for this long (it counts every 100 ms).
const HEART_STALE: Duration = Duration::from_secs(2);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Live {
    /// CX_C_HEART, counted up by the daemon's thread 0 every 100 ms (card/cortexd.c CX_HEART_S).
    pub heart: u64,
    /// CX_H_SESSION == CX_C_SESSION with CX_C_STATE 1 (ready): a client's session is open.
    pub session_open: bool,
    /// CX_T_MAGIC set: a coach feeds the ring.
    pub run: bool,
    pub chunks_fed: u64,
    pub chunks: u64,
    pub reports: u64,
    pub loss_nats: f64,
    pub lr_eff: f64,
    pub tok_s: f64,
    pub lr: f64,
    pub scale: f64,
    pub flags: u64,
    pub every: u32,
    /// CX_T_ACK: the flags word as last acted on.
    pub ack: u64,
    pub ckpt_chunk: u64,
    /// CX_T_STATE: 0 off, 1 training, 2 paused, 3 waiting for chunks, 4 stopped.
    pub train_state: u32,
    pub saves: u64,
}

pub struct Reader {
    file: File,
}

impl Reader {
    pub fn open(path: &str) -> io::Result<Self> {
        Ok(Self { file: File::open(path)? })
    }

    pub fn read(&self) -> io::Result<Live> {
        let mut mbox = [0u8; MBOX_LEN];
        self.file.read_exact_at(&mut mbox, PART_OFFSET)?;
        let mut t = [0u8; T_LEN];
        self.file.read_exact_at(&mut t, PART_OFFSET + T_BASE)?;
        Ok(decode(&mbox, &t))
    }
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"))
}

fn f64_at(b: &[u8], o: usize) -> f64 {
    f64::from_bits(u64_at(b, o))
}

/// `mbox` = the part's first MBOX_LEN bytes, `t` = T_LEN bytes from CX_T_BASE (the offsets of lib/cx-proto.h).
pub fn decode(mbox: &[u8], t: &[u8]) -> Live {
    Live {
        heart: u64_at(mbox, 256),
        session_open: u32_at(mbox, 136) == 1 && u64_at(mbox, 8) == u64_at(mbox, 128),
        run: u32_at(t, 0) == T_MAGIC_VAL,
        chunks_fed: u64_at(t, 64),
        lr: f64_at(t, 128),
        scale: f64_at(t, 192),
        flags: u64_at(t, 256),
        every: u32_at(t, 320),
        chunks: u64_at(t, 1024),
        reports: u64_at(t, 1088),
        loss_nats: f64_at(t, 1152),
        lr_eff: f64_at(t, 1216),
        tok_s: f64_at(t, 1280),
        ack: u64_at(t, 1344),
        ckpt_chunk: u64_at(t, 1408),
        train_state: u32_at(t, 1472),
        saves: u64_at(t, 1536),
    }
}

/// The part as last read, polled every 250 ms; the daemon alive while its heartbeat moves.
pub struct LiveState {
    pub live: Option<Live>,
    /// Why the part cannot be read, when it cannot.
    pub error: Option<String>,
    pub alive: bool,
    reader: Option<Reader>,
    polled: Option<Instant>,
    heart_seen: (u64, Instant),
}

impl Default for LiveState {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveState {
    pub fn new() -> Self {
        let now = Instant::now();
        Self { live: None, error: None, alive: false, reader: None, polled: None, heart_seen: (0, now.checked_sub(HEART_STALE).unwrap_or(now)) }
    }

    /// A coach feeds the daemon: the training block's magic word is set.
    pub fn running(&self) -> bool {
        self.live.as_ref().is_some_and(|l| l.run)
    }

    /// The run's scale knob as the card reports it, while a coach feeds.
    pub fn scale(&self) -> Option<f64> {
        self.live.as_ref().filter(|l| l.run).map(|l| l.scale)
    }

    /// The run is paused (CX_T_STATE 2), while a coach feeds.
    pub fn paused(&self) -> bool {
        self.live.as_ref().is_some_and(|l| l.run && l.train_state == 2)
    }

    /// The part when due; whether anything shown changed.
    pub fn poll(&mut self, part: &str) -> bool {
        if self.polled.is_some_and(|t| t.elapsed() < LIVE_POLL) {
            return false;
        }
        self.polled = Some(Instant::now());
        if self.reader.is_none() {
            match Reader::open(part) {
                Ok(r) => self.reader = Some(r),
                Err(e) => {
                    let msg = format!("{part}: {e}");
                    let changed = self.error.as_deref() != Some(&msg);
                    self.error = Some(msg);
                    return changed;
                }
            }
        }
        let Some(r) = &self.reader else { return false };
        match r.read() {
            Ok(l) => {
                if l.heart != self.heart_seen.0 {
                    self.heart_seen = (l.heart, Instant::now());
                }
                let alive = self.heart_seen.1.elapsed() < HEART_STALE;
                let changed = self.live.as_ref() != Some(&l) || alive != self.alive || self.error.is_some();
                self.alive = alive;
                self.live = Some(l);
                self.error = None;
                changed
            }
            Err(e) => {
                self.error = Some(format!("{part}: {e}"));
                self.live = None;
                self.alive = false;
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_protocol_offsets() {
        let mut mbox = [0u8; MBOX_LEN];
        mbox[8..16].copy_from_slice(&7u64.to_le_bytes());
        mbox[128..136].copy_from_slice(&7u64.to_le_bytes());
        mbox[136..140].copy_from_slice(&1u32.to_le_bytes());
        mbox[256..264].copy_from_slice(&99u64.to_le_bytes());
        let mut t = [0u8; T_LEN];
        t[0..4].copy_from_slice(&T_MAGIC_VAL.to_le_bytes());
        t[64..72].copy_from_slice(&7200u64.to_le_bytes());
        t[1024..1032].copy_from_slice(&7100u64.to_le_bytes());
        t[1152..1160].copy_from_slice(&4.2315f64.to_le_bytes());
        t[1216..1224].copy_from_slice(&7.5e-4f64.to_le_bytes());
        t[1280..1288].copy_from_slice(&1291.0f64.to_le_bytes());
        t[1344..1352].copy_from_slice(&5u64.to_le_bytes());
        t[1472..1476].copy_from_slice(&1u32.to_le_bytes());
        let l = decode(&mbox, &t);
        assert!(l.session_open && l.run && l.heart == 99 && l.ack == 5);
        assert_eq!((l.chunks_fed, l.chunks, l.train_state), (7200, 7100, 1));
        assert_eq!((l.loss_nats, l.lr_eff, l.tok_s), (4.2315, 7.5e-4, 1291.0));
    }

    #[test]
    fn an_unreadable_part_is_an_error_not_a_panic() {
        let mut s = LiveState::new();
        assert!(s.poll("/nonexistent/part"));
        assert!(s.error.is_some() && s.live.is_none() && !s.running() && s.scale().is_none() && !s.paused());
        assert!(!s.poll("/nonexistent/part"), "the same error again is no change");
    }
}
