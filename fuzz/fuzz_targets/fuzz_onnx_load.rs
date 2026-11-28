#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to the ONNX protobuf parser, model validator, IR
    // optimizer, and plan builder. This tests that malformed input never causes
    // panics or UB.
    let _ = yscv_onnx::OnnxLoader::from_bytes(data).build();
});
