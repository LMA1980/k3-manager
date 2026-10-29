fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let dirs: &[&str] = if target_os == "macos" {
        &["/opt/homebrew/lib", "/usr/local/lib"]
    } else {
        &["/usr/local/lib"]
    };
    for dir in dirs {
        if std::path::Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}
