//! Scenario and unit verification for US-20: Architecture Recipe Capability Filtering.
//!
//! Verifies:
//! - Scenario 1: Capability filtering during model search and inspection.
//! - Scenario 2: Upfront rejection of unsupported architecture revisions with `K3Error::CapabilityMismatch`.
//! - Scenario 3: Guaranteed alignment between tokenization and execution recipe.

use k3_core::K3Error;
use k3_manager::store::{CapabilityFilter, ModelManifest};
use k3_tokenizer::SimpleTokenizer;

#[test]
fn test_scenario1_capability_filtering_during_model_search() {
    // 1. Candidate with supported architecture recipe (synthetic @ 1.0.0, Q4_K_M)
    let supported_candidate = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: Some("Apache-2.0".to_owned()),
        origin: Some("https://huggingface.co/synthetic/model".to_owned()),
        weight_digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };

    let suitability1 = CapabilityFilter::evaluate_candidate(&supported_candidate);
    assert!(
        suitability1.executable,
        "supported candidate must be flagged executable"
    );
    assert!(suitability1.quantization_supported);
    assert!(suitability1.reason.is_none());

    // 2. Candidate with unsupported quantization format on otherwise supported recipe
    let unsupported_quant_candidate = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "synthetic".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q8_0".to_owned(), // Synthetic does not support Q8_0
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };

    let suitability2 = CapabilityFilter::evaluate_candidate(&unsupported_quant_candidate);
    assert!(
        !suitability2.executable,
        "candidate with unsupported quant must be non-executable"
    );
    assert!(!suitability2.quantization_supported);
    assert!(
        suitability2
            .reason
            .as_ref()
            .unwrap()
            .contains("quantization format 'Q8_0' is not supported")
    );

    // 3. Candidate with uncompiled minor revision (e.g., qwen3.8)
    let uncompiled_arch_candidate = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "qwen3.8".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };

    let suitability3 = CapabilityFilter::evaluate_candidate(&uncompiled_arch_candidate);
    assert!(
        !suitability3.executable,
        "uncompiled arch candidate must be non-executable"
    );
    assert!(
        suitability3
            .reason
            .as_ref()
            .unwrap()
            .contains("architecture 'qwen3.8' with recipe version '1.0.0' is not compiled")
    );
}

#[test]
fn test_scenario2_upfront_rejection_of_unsupported_revisions() {
    // 1. Attempting to validate uncompiled architecture qwen3.8
    let qwen3_8_manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "qwen3.8".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };

    let err = CapabilityFilter::validate_model_manifest(&qwen3_8_manifest)
        .expect_err("qwen3.8 must be rejected upfront");
    assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    assert_eq!(err.diagnostic_code(), "capability_mismatch");
    assert!(
        err.to_string()
            .contains("architecture 'qwen3.8' (recipe version '1.0.0') is not compiled")
    );

    // 2. Attempting to validate supported architecture with mismatched recipe version (e.g. qwen3 @ 2.0.0)
    let mismatched_version_manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "qwen3".to_owned(),
        recipe_version: "2.0.0".to_owned(), // only 1.0.0 is compiled
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };

    let err2 = CapabilityFilter::validate_model_manifest(&mismatched_version_manifest)
        .expect_err("mismatched recipe version must be rejected upfront");
    assert!(matches!(err2, K3Error::CapabilityMismatch { .. }));
    assert!(
        err2.to_string()
            .contains("architecture 'qwen3' (recipe version '2.0.0') is not compiled")
    );

    // 3. Supported architecture & version succeeds upfront
    let valid_manifest = ModelManifest {
        schema_version: 1,
        format: "gguf".to_owned(),
        architecture_id: "qwen3".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "Q4_K_M".to_owned(),
        context_length: 2048,
        license: None,
        origin: None,
        weight_digest: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_owned(),
        weight_size_bytes: 1024,
        created_at: 1000,
    };

    let desc = CapabilityFilter::validate_model_manifest(&valid_manifest)
        .expect("qwen3 @ 1.0.0 must be accepted");
    assert_eq!(desc.architecture_id, "qwen3");
    assert_eq!(desc.recipe_version, "1.0.0");
}

#[test]
fn test_scenario3_guaranteed_alignment_between_tokenization_and_recipe() {
    // 1. Align prompt environment with Qwen3 recipe
    let aligned_env = CapabilityFilter::align_prompt_environment("qwen3", "1.0.0")
        .expect("align prompt environment for qwen3");
    assert_eq!(aligned_env.architecture_id, "qwen3");
    assert_eq!(aligned_env.recipe_version, "1.0.0");

    // Verify stop token detection adheres strictly to Qwen3 recipe definition
    assert!(
        aligned_env.is_stop_token(151_643),
        "<|endoftext|> is stop token"
    );
    assert!(
        aligned_env.is_stop_token(151_645),
        "<|im_end|> is stop token"
    );
    assert!(!aligned_env.is_stop_token(100), "normal token is not stop");

    // 2. Chat template formatting adheres strictly to Qwen3 template
    let user_prompt = "Hello, world!";
    let formatted = aligned_env.chat_template.format_prompt(user_prompt);
    assert!(
        formatted.contains("<|im_start|>user\nHello, world!<|im_end|>"),
        "must format using bound Qwen chat template"
    );

    // 3. Tokenize with aligned environment
    let mut tokenizer = SimpleTokenizer::new();
    tokenizer.add_token("<|im_start|>user", 100);
    tokenizer.add_token("Hello,", 101);
    tokenizer.add_token("world!<|im_end|>", 102);
    tokenizer.add_token("<|im_start|>assistant", 103);

    let tokens = aligned_env
        .format_and_tokenize(&tokenizer, user_prompt, 2048)
        .expect("format and tokenize succeeds");
    assert_ne!(tokens, [] as [u32; 0]);

    // 4. Exceeding n_ctx is rejected
    let err = aligned_env
        .format_and_tokenize(&tokenizer, user_prompt, 2)
        .expect_err("n_ctx = 2 must exceed bound");
    assert!(matches!(err, K3Error::ContextLengthExceeded { .. }));
}
