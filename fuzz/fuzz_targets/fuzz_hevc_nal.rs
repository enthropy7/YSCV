#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes as HEVC NAL units to the decoder.
    let mut decoder = yscv_video::HevcDecoder::new();
    let _ = decoder.decode_nal(data);
});
