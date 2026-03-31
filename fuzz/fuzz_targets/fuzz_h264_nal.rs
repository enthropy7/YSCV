#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes as H.264 NAL units to the decoder.
    // This tests that malformed input never causes panics or UB.
    let mut decoder = yscv_video::H264Decoder::new();
    let nals = yscv_video::parse_annex_b(data);
    for nal in &nals {
        let _ = decoder.process_nal(nal);
    }
});
