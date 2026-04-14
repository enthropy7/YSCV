#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Write arbitrary bytes to a temp file and attempt image decoding.
    // This tests that malformed image data never causes panics or UB in
    // format detection and pixel decoding.
    let tmp = std::env::temp_dir().join("fuzz_input.img");
    if std::fs::write(&tmp, data).is_ok() {
        let _ = yscv_imgproc::imread(&tmp);
        let _ = std::fs::remove_file(&tmp);
    }
});
