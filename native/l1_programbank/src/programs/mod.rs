pub mod alarm;
pub mod weather;

use crate::frame::Frame;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub frame: Frame,
    pub program_path: &'static str,
}
