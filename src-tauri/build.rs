// Link the macOS frameworks we use directly (AXIsProcessTrusted lives in ApplicationServices),
// then run the Tauri build step.
fn main() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=ApplicationServices");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=AVFoundation");
    }
    tauri_build::build()
}
