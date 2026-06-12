use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn worker_answers_jsonl_requests() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_darjeeling-l1-programbank"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn l1 worker");

    {
        let stdin = child.stdin.as_mut().expect("worker stdin");
        writeln!(
            stdin,
            r#"{{"request_id":"r1","utterance":"alpha accept red"}}"#
        )
        .expect("write request");
    }

    let output = child.wait_with_output().expect("read worker output");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8 output");
    assert!(stdout.contains(r#""request_id":"r1""#));
    assert!(stdout.contains(r#""accepted":true"#));
    assert!(stdout.contains(r#""intent":"intent_alpha""#));
    assert!(stdout.contains(r#""native_latency_us":"#));
}
