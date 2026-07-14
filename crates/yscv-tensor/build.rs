fn main() {
    // macOS: link Accelerate.framework for vDSP vector operations.
    // Use CARGO_CFG_TARGET_OS (not cfg!) to check the TARGET, not the HOST.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
