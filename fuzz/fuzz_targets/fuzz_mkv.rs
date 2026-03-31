#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test MKV EBML parser with arbitrary data.
    // Write data to a temp file and try to open as MKV.
    let tmp = std::env::temp_dir().join("fuzz_input.mkv");
    if std::fs::write(&tmp, data).is_ok() {
        let _ = yscv_video::mkv::MkvDemuxer::open(&tmp);
        let _ = std::fs::remove_file(&tmp);
    }
});
