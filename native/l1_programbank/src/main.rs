use darjeeling_l1_programbank::frame::{L1Result, Request};
use darjeeling_l1_programbank::try_answer;
use std::io::{self, BufRead, Write};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let result = match serde_json::from_str::<Request>(&line) {
            Ok(request) => try_answer(&request),
            Err(error) => L1Result::abstain("", format!("invalid request json: {error}"), 0),
        };
        serde_json::to_writer(&mut stdout, &result)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }

    Ok(())
}
