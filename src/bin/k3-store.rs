//! `k3-store`: Administrative CLI utility for Model-Runner content-addressed store & safe GC (US-22, ADR-040).
//!
//! Provides operational commands for storage inspection, active lock/mapping queries (`who-uses`),
//! and safe dual-gate garbage collection (`gc`).

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use k3_manager::persistence::{CasCatalog, DuckDbPersistenceStore};
use k3_manager::store::{
    ContentAddressedStore, DEFAULT_STORE_ROOT, ENV_STORE_ROOT, StorageGarbageCollector,
};

fn print_usage() {
    eprintln!(
        "Usage: k3-store [OPTIONS] <COMMAND>\n\n\
         Commands:\n  \
           who-uses <digest>            Inspect active locks and memory mappings for a blob\n  \
           gc [--dry-run]               Safely reclaim unreferenced weight blobs via dual-gate check\n  \
           list [blobs|manifests|aliases] List registered store assets\n  \
           inspect <alias_or_digest>    Inspect manifest metadata and resolved weight path\n\n\
         Options:\n  \
           --store-root <path>          Path to store root directory (default: /var/lib/k3 or $K3_STORE_ROOT)\n  \
           --json                       Format output as JSON\n  \
           -h, --help                   Display this help message"
    );
}

fn open_store(
    store_root: &Path,
) -> Result<(ContentAddressedStore, StorageGarbageCollector), String> {
    let catalog_path = store_root.join("catalog.duckdb");
    let catalog = DuckDbPersistenceStore::open(&catalog_path).map_err(|e| {
        format!(
            "failed to open DuckDB catalog at '{}': {e}",
            catalog_path.display()
        )
    })?;
    let catalog_arc: Arc<dyn CasCatalog> = Arc::new(catalog);

    let store = ContentAddressedStore::open(store_root, catalog_arc).map_err(|e| {
        format!(
            "failed to open content-addressed store at '{}': {e}",
            store_root.display()
        )
    })?;
    let gc = StorageGarbageCollector::new(Arc::new(store.clone()));

    Ok((store, gc))
}

#[allow(clippy::too_many_lines)]
fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(1);
    }

    let mut store_root_str =
        env::var(ENV_STORE_ROOT).unwrap_or_else(|_| DEFAULT_STORE_ROOT.to_owned());
    let mut as_json = false;
    let mut dry_run = false;
    let mut cmd = None;
    let mut cmd_arg = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            "--json" => {
                as_json = true;
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--store-root" => {
                if i + 1 < args.len() {
                    i += 1;
                    store_root_str.clone_from(&args[i]);
                } else {
                    eprintln!("Error: --store-root requires a path argument");
                    return ExitCode::from(1);
                }
            }
            subcmd if cmd.is_none() => {
                cmd = Some(subcmd.to_owned());
            }
            arg if cmd_arg.is_none() => {
                cmd_arg = Some(arg.to_owned());
            }
            other => {
                eprintln!("Error: unrecognized argument '{other}'");
                return ExitCode::from(1);
            }
        }
        i += 1;
    }

    let Some(command) = cmd else {
        print_usage();
        return ExitCode::from(1);
    };

    let store_root = PathBuf::from(&store_root_str);
    let (store, gc) = match open_store(&store_root) {
        Ok(res) => res,
        Err(err) => {
            eprintln!("Error: {err}");
            return ExitCode::from(2);
        }
    };

    match command.as_str() {
        "who-uses" => {
            let Some(digest) = cmd_arg else {
                eprintln!("Error: 'who-uses' requires a <digest> argument");
                return ExitCode::from(1);
            };

            match gc.who_uses(&digest) {
                Ok(info) => {
                    if as_json {
                        if let Ok(json) = serde_json::to_string_pretty(&info) {
                            println!("{json}");
                        }
                    } else {
                        println!("Blob: {}", info.digest);
                        println!("Path: {}", info.file_path.display());
                        println!("Size: {} bytes", info.size_bytes);
                        println!(
                            "References: {} active ({:?})",
                            info.ref_count, info.referencing_sources
                        );
                        println!(
                            "Gate 1 (flock): {}",
                            if info.gate1_shared_lock_held {
                                "SHARED LOCK HELD (active worker)"
                            } else {
                                "No lock held (exclusive lock obtainable)"
                            }
                        );
                        println!(
                            "Gate 2 (mappings): {} active process mapping(s)",
                            info.gate2_active_mappings.len()
                        );
                        for m in &info.gate2_active_mappings {
                            println!(
                                "  - PID {} ({}) mapped at {} ({})",
                                m.pid, m.comm, m.address_range, m.permissions
                            );
                        }
                        println!(
                            "Status: {}",
                            if info.is_in_use() {
                                "IN USE (unsafe to delete)"
                            } else {
                                "IDLE (safe to reclaim if ref_count == 0)"
                            }
                        );
                    }
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("Error inspecting blob: {err}");
                    ExitCode::from(1)
                }
            }
        }
        "gc" => match gc.run_gc(dry_run) {
            Ok(report) => {
                if as_json {
                    if let Ok(json) = serde_json::to_string_pretty(&report) {
                        println!("{json}");
                    }
                } else {
                    println!("Garbage Collection Sweep (dry_run: {})", report.dry_run);
                    println!("Candidates examined: {}", report.examined_candidates);
                    println!("Blobs reclaimed: {}", report.reclaimed_blobs.len());
                    println!("Bytes reclaimed: {} bytes", report.reclaimed_bytes);
                    if !report.skipped_in_use_gate1.is_empty() {
                        println!(
                            "Skipped (Gate 1 flock held): {:?}",
                            report.skipped_in_use_gate1
                        );
                    }
                    if !report.skipped_mapped_gate2.is_empty() {
                        println!(
                            "Skipped (Gate 2 actively mapped): {:?}",
                            report.skipped_mapped_gate2
                        );
                    }
                }
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("Error executing garbage collection: {err}");
                ExitCode::from(1)
            }
        },
        "list" => {
            let filter = cmd_arg.as_deref().unwrap_or("all");
            match filter {
                "blobs" => {
                    let blobs = store.catalog().list_blobs().unwrap_or_default();
                    if as_json {
                        let _ = serde_json::to_writer_pretty(std::io::stdout(), &blobs);
                        println!();
                    } else {
                        println!("Total Blobs: {}", blobs.len());
                        for b in blobs {
                            println!(
                                "  {} ({} bytes, ref_count: {}) [{}]",
                                b.digest, b.size_bytes, b.ref_count, b.blob_type
                            );
                        }
                    }
                }
                "manifests" => {
                    let manifests = store.catalog().list_manifests().unwrap_or_default();
                    if as_json {
                        let _ = serde_json::to_writer_pretty(std::io::stdout(), &manifests);
                        println!();
                    } else {
                        println!("Total Manifests: {}", manifests.len());
                        for m in manifests {
                            println!(
                                "  {} (arch: {}, quant: {}) -> weights: {}",
                                m.manifest_digest,
                                m.architecture_id,
                                m.quantization,
                                m.weight_digest
                            );
                        }
                    }
                }
                "aliases" => {
                    let aliases = store.catalog().list_aliases().unwrap_or_default();
                    if as_json {
                        let _ = serde_json::to_writer_pretty(std::io::stdout(), &aliases);
                        println!();
                    } else {
                        println!("Total Aliases: {}", aliases.len());
                        for a in aliases {
                            println!("  {} -> manifest: {}", a.alias_id, a.manifest_digest);
                        }
                    }
                }
                _ => {
                    let aliases = store.catalog().list_aliases().unwrap_or_default();
                    let blobs = store.catalog().list_blobs().unwrap_or_default();
                    println!("Store Root: {}", store.root().display());
                    println!("Aliases: {}", aliases.len());
                    for a in aliases {
                        println!("  - {}", a.alias_id);
                    }
                    println!("Physical Blobs: {}", blobs.len());
                    for b in blobs {
                        println!(
                            "  - {} ({} bytes, refs: {})",
                            b.digest, b.size_bytes, b.ref_count
                        );
                    }
                }
            }
            ExitCode::SUCCESS
        }
        "inspect" => {
            let Some(target) = cmd_arg else {
                eprintln!("Error: 'inspect' requires an alias or digest argument");
                return ExitCode::from(1);
            };

            if let Ok(resolved) = store.resolve_alias(&target) {
                if as_json {
                    let _ = serde_json::to_writer_pretty(std::io::stdout(), &resolved.manifest);
                    println!();
                } else {
                    println!("Alias: {}", resolved.alias_id);
                    println!("Manifest Path: {}", resolved.manifest_path.display());
                    println!("Weight Path:   {}", resolved.weight_path.display());
                    println!("Format:        {}", resolved.manifest.format);
                    println!("Architecture:  {}", resolved.manifest.architecture_id);
                    println!("Recipe:        {}", resolved.manifest.recipe_version);
                    println!("Quantization:  {}", resolved.manifest.quantization);
                    println!("Context Limit: {}", resolved.manifest.context_length);
                    println!("Weight Digest: {}", resolved.manifest.weight_digest);
                    println!("Size Bytes:    {}", resolved.manifest.weight_size_bytes);
                }
                ExitCode::SUCCESS
            } else {
                eprintln!("Error: could not resolve alias '{target}'");
                ExitCode::from(1)
            }
        }
        other => {
            eprintln!("Error: unrecognized command '{other}'");
            print_usage();
            ExitCode::from(1)
        }
    }
}
