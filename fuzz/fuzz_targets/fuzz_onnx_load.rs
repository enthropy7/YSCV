#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to the ONNX protobuf parser and model validator.
    // This tests that malformed input never causes panics or UB.
    let _ = yscv_onnx::load_onnx_model(data);
});
