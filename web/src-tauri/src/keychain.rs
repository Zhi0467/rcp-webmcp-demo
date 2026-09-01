use std::{
    io::{Read, Write},
    os::unix::process::CommandExt,
    process::{Command, Stdio},
    thread,
};

use zeroize::Zeroizing;

const SECURITY_TOOL: &str = "/usr/bin/security";
const ITEM_NOT_FOUND_EXIT_CODE: i32 = 44;
const MAX_VALUE_BYTES: usize = 64;
const MAX_ENCODED_OUTPUT_BYTES: usize = MAX_VALUE_BYTES * 2 + 2;

fn security_command() -> Command {
    let mut command = Command::new(SECURITY_TOOL);
    // `security -w` reads from the controlling terminal when one exists. A
    // source-built desktop is commonly started from a terminal, so put the
    // child in a new session and make its piped stdin authoritative.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
    command
}

fn hex_encode(value: &[u8]) -> Zeroizing<String> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Zeroizing::new(encoded)
}

fn decode_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err("Keychain returned an invalid encoded credential".into()),
    }
}

fn hex_decode(value: &[u8]) -> Result<Zeroizing<Vec<u8>>, String> {
    if !value.len().is_multiple_of(2) {
        return Err("Keychain returned an invalid encoded credential".into());
    }
    value
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Ok((decode_nibble(pair[0])? << 4) | decode_nibble(pair[1])?))
        .collect::<Result<Vec<_>, _>>()
        .map(Zeroizing::new)
}

fn read_bounded_secret(
    reader: impl Read,
    maximum_bytes: usize,
) -> std::io::Result<Zeroizing<Vec<u8>>> {
    let mut value = Zeroizing::new(Vec::with_capacity(maximum_bytes));
    reader
        .take((maximum_bytes + 1) as u64)
        .read_to_end(&mut value)?;
    Ok(value)
}

fn command_error(action: &str, output: &std::process::Output) -> String {
    let status = output
        .status
        .code()
        .map_or_else(|| "signal".into(), |code| code.to_string());
    let detail = String::from_utf8_lossy(&output.stderr);
    let detail = detail.trim();
    if detail.is_empty() {
        format!("{action} failed (status {status})")
    } else {
        format!("{action} failed (status {status}): {detail}")
    }
}

pub fn set(service: &str, account: &str, value: &[u8]) -> Result<(), String> {
    if value.len() > MAX_VALUE_BYTES {
        return Err(format!(
            "the macOS Keychain tool accepts at most {MAX_VALUE_BYTES} credential bytes"
        ));
    }
    let encoded = hex_encode(value);
    let mut child = security_command()
        .args([
            "add-generic-password",
            "-a",
            account,
            "-s",
            service,
            "-U",
            "-T",
            SECURITY_TOOL,
            "-w",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start the macOS Keychain tool: {error}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "could not open the macOS Keychain tool input".to_string())?;
    let write_result = stdin
        .write_all(encoded.as_bytes())
        .and_then(|()| stdin.write_all(b"\n"))
        .and_then(|()| stdin.write_all(encoded.as_bytes()))
        .and_then(|()| stdin.write_all(b"\n"));
    drop(stdin);
    if let Err(error) = write_result {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!(
            "could not provide the credential to Keychain: {error}"
        ));
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("could not wait for the macOS Keychain tool: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_error("storing the Keychain credential", &output))
    }
}

pub fn get(service: &str, account: &str) -> Result<Option<Zeroizing<Vec<u8>>>, String> {
    let mut child = security_command()
        .args(["find-generic-password", "-a", account, "-s", service, "-w"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start the macOS Keychain tool: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "could not open the macOS Keychain tool output".to_string())?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| "could not open the macOS Keychain tool error output".to_string())?;
    let stdout_reader =
        thread::spawn(move || read_bounded_secret(stdout, MAX_ENCODED_OUTPUT_BYTES));
    let stderr_reader = thread::spawn(move || {
        let mut detail = Vec::new();
        stderr.read_to_end(&mut detail).map(|_| detail)
    });
    let encoded = stdout_reader
        .join()
        .map_err(|_| "the macOS Keychain output reader stopped unexpectedly".to_string())?
        .map_err(|error| format!("could not read the macOS Keychain output: {error}"))?;
    if encoded.len() > MAX_ENCODED_OUTPUT_BYTES {
        let _ = child.kill();
    }
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for the macOS Keychain tool: {error}"))?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "the macOS Keychain error reader stopped unexpectedly".to_string())?
        .map_err(|error| format!("could not read the macOS Keychain error output: {error}"))?;
    if encoded.len() > MAX_ENCODED_OUTPUT_BYTES {
        return Err("Keychain returned an oversized encoded credential".into());
    }
    if status.code() == Some(ITEM_NOT_FOUND_EXIT_CODE) {
        return Ok(None);
    }
    if !status.success() {
        let output = std::process::Output {
            status,
            stdout: Vec::new(),
            stderr,
        };
        return Err(command_error("reading the Keychain credential", &output));
    }
    let mut encoded = encoded;
    while encoded
        .last()
        .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
    {
        encoded.pop();
    }
    if encoded.len() > MAX_VALUE_BYTES * 2 {
        return Err("Keychain returned an oversized encoded credential".into());
    }
    hex_decode(&encoded).map(Some)
}

pub fn remove(service: &str, account: &str) -> Result<bool, String> {
    let output = security_command()
        .args(["delete-generic-password", "-a", account, "-s", service])
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("could not start the macOS Keychain tool: {error}"))?;
    if output.status.success() {
        Ok(true)
    } else if output.status.code() == Some(ITEM_NOT_FOUND_EXIT_CODE) {
        Ok(false)
    } else {
        Err(command_error("removing the Keychain credential", &output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_encoding_round_trips_binary_data() {
        let value = b"\0binary\nidentity\xff";
        let encoded = hex_encode(value);
        assert_eq!(&*hex_decode(encoded.as_bytes()).unwrap(), value);
    }

    #[test]
    fn credential_decoder_rejects_malformed_values() {
        assert!(hex_decode(b"0").is_err());
        assert!(hex_decode(b"gg").is_err());
    }

    #[test]
    fn credential_reader_stops_one_byte_past_its_limit() {
        let value = vec![b'x'; MAX_ENCODED_OUTPUT_BYTES + 20];
        let bounded = read_bounded_secret(value.as_slice(), MAX_ENCODED_OUTPUT_BYTES).unwrap();
        assert_eq!(bounded.len(), MAX_ENCODED_OUTPUT_BYTES + 1);
    }

    #[test]
    fn keychain_writer_rejects_values_the_tool_would_truncate() {
        assert!(set("unused", "unused", &[0; MAX_VALUE_BYTES + 1]).is_err());
    }
}
