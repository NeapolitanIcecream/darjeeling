pub mod alpha;

use crate::frame::Frame;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub frame: Frame,
    pub program_path: &'static str,
}
