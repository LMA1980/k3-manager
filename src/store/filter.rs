//! Architecture Recipe Capability Filter & Tokenizer Alignment (US-20, ADR-017, ADR-023, ADR-024, ADR-025).
//!
//! Validates model architecture and minor algorithmic revisions (`architecture_id`, `recipe_version`)
//! upfront during registry search, model inspection, and load time. Guarantees that control-plane
//! prompt preparation and chat templates are strictly bound to the active execution recipe.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use k3_core::K3Error;
use k3_tokenizer::{ChatTemplate, SpecialTokens, Tokenizer};

use super::cas::ModelManifest;

/// Information describing a compiled architecture recipe and its capabilities.
#[derive(Debug, Clone)]
pub struct RecipeDescriptor {
    /// Exact architecture identifier (e.g. "synthetic", "qwen3", "llama").
    pub architecture_id: &'static str,
    /// Verified recipe revision string (e.g. "1.0.0").
    pub recipe_version: &'static str,
    /// List of supported quantization formats (e.g. `["Q4_K_M", "F32", "F16", "BF16"]`).
    pub supported_quants: &'static [&'static str],
    /// Target execution layouts (e.g. `["native", "ime_tile_8x16"]`).
    pub target_layouts: &'static [&'static str],
    /// Standard chat template bound to this recipe version.
    pub default_chat_template: Option<ChatTemplate>,
    /// Special stop tokens bound to this recipe version.
    pub special_tokens: Option<SpecialTokens>,
}

static BUILTIN_RECIPES: LazyLock<RwLock<HashMap<String, RecipeDescriptor>>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    // 1. Reference synthetic recipe (ADR-023)
    let synthetic_template = ChatTemplate::new(Some(
        "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n".to_owned(),
    ));
    let synthetic_stop = SpecialTokens {
        stop_tokens: vec![0, 2],
        ..Default::default()
    };

    map.insert(
        "synthetic@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "synthetic",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(synthetic_template),
            special_tokens: Some(synthetic_stop),
        },
    );

    // 2. Qwen3 base recipe (ADR-023)
    let qwen3_template = ChatTemplate::new(Some(
        "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n".to_owned(),
    ));
    let qwen3_stop = SpecialTokens {
        stop_tokens: vec![151_643, 151_645], // <|endoftext|>, <|im_end|>
        eot_token_id: Some(151_645),
        ..Default::default()
    };

    map.insert(
        "qwen3@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "qwen3",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(qwen3_template),
            special_tokens: Some(qwen3_stop),
        },
    );

    // 3. Llama base recipe (ADR-023)
    let llama_template = ChatTemplate::new(Some(
        "<|start_header_id|>user<|end_header_id|>\n\n{{prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".to_owned(),
    ));
    let llama_stop = SpecialTokens {
        stop_tokens: vec![128_001, 128_009], // <|end_of_text|>, <|eot_id|>
        eot_token_id: Some(128_009),
        ..Default::default()
    };

    map.insert(
        "llama@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "llama",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "F32", "F16", "BF16"],
            target_layouts: &["native"],
            default_chat_template: Some(llama_template),
            special_tokens: Some(llama_stop),
        },
    );

    RwLock::new(map)
});

/// Suitability and compatibility assessment for candidate models during search and inspection (US-20 Scenario 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSuitability {
    /// Whether the model architecture and recipe version can execute on this runtime build.
    pub executable: bool,
    /// Architecture identifier.
    pub architecture_id: String,
    /// Declared or resolved recipe version.
    pub recipe_version: String,
    /// Whether the model's quantization format is supported.
    pub quantization_supported: bool,
    /// Diagnostic explanation if the model is not executable.
    pub reason: Option<String>,
}

/// Aligned prompt preparation and tokenization environment bound to a verified recipe version (US-20 Scenario 3).
#[derive(Debug, Clone)]
pub struct AlignedPromptEnvironment {
    /// Architecture identifier.
    pub architecture_id: &'static str,
    /// Active recipe revision.
    pub recipe_version: &'static str,
    /// Bound chat template.
    pub chat_template: ChatTemplate,
    /// Bound stop sequence detector.
    pub special_tokens: SpecialTokens,
}

impl AlignedPromptEnvironment {
    /// Formats a user prompt using the bound chat template, tokenizes with `tokenizer`,
    /// and validates context window bounds (`n_ctx`).
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] or [`K3Error::CapabilityMismatch`].
    pub fn format_and_tokenize(
        &self,
        tokenizer: &dyn Tokenizer,
        prompt: &str,
        n_ctx: usize,
    ) -> Result<Vec<u32>, K3Error> {
        let formatted = self.chat_template.format_prompt(prompt);
        let tokens = tokenizer.encode(&formatted)?;
        if n_ctx == 0 || tokens.len() >= n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: tokens.len(),
                max_context_tokens: n_ctx,
            });
        }
        Ok(tokens)
    }

    /// Evaluates if `token_id` is an active stop token for this recipe version.
    #[must_use]
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.special_tokens.is_stop(token_id)
    }
}

/// Architecture Recipe Capability Filter engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CapabilityFilter;

impl CapabilityFilter {
    /// Returns a list of all recipe scope keys currently compiled into the runtime.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    #[must_use]
    pub fn supported_recipes() -> Vec<String> {
        let registry = BUILTIN_RECIPES.read().unwrap();
        let mut list: Vec<String> = registry.keys().cloned().collect();
        list.sort();
        list
    }

    /// Returns `true` if `(architecture_id, recipe_version)` is supported.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    #[must_use]
    pub fn is_supported(architecture_id: &str, recipe_version: &str) -> bool {
        let key = format!("{architecture_id}@{recipe_version}");
        let registry = BUILTIN_RECIPES.read().unwrap();
        registry.contains_key(&key)
    }

    /// Evaluates the compatibility of candidate model metadata during search or listing (US-20 Scenario 1).
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    #[must_use]
    pub fn evaluate_candidate(manifest: &ModelManifest) -> CandidateSuitability {
        let key = format!("{}@{}", manifest.architecture_id, manifest.recipe_version);
        let registry = BUILTIN_RECIPES.read().unwrap();

        match registry.get(&key) {
            Some(desc) => {
                let quant_supported = desc
                    .supported_quants
                    .iter()
                    .any(|q| q.eq_ignore_ascii_case(&manifest.quantization));

                if !quant_supported {
                    CandidateSuitability {
                        executable: false,
                        architecture_id: manifest.architecture_id.clone(),
                        recipe_version: manifest.recipe_version.clone(),
                        quantization_supported: false,
                        reason: Some(format!(
                            "quantization format '{}' is not supported by recipe '{}' (supported: {:?})",
                            manifest.quantization, key, desc.supported_quants
                        )),
                    }
                } else if manifest.context_length == 0 {
                    CandidateSuitability {
                        executable: false,
                        architecture_id: manifest.architecture_id.clone(),
                        recipe_version: manifest.recipe_version.clone(),
                        quantization_supported: true,
                        reason: Some("model context length must be greater than 0".to_owned()),
                    }
                } else {
                    CandidateSuitability {
                        executable: true,
                        architecture_id: manifest.architecture_id.clone(),
                        recipe_version: manifest.recipe_version.clone(),
                        quantization_supported: true,
                        reason: None,
                    }
                }
            }
            None => CandidateSuitability {
                executable: false,
                architecture_id: manifest.architecture_id.clone(),
                recipe_version: manifest.recipe_version.clone(),
                quantization_supported: false,
                reason: Some(format!(
                    "architecture '{}' with recipe version '{}' is not compiled into this runtime build (supported recipes: {:?})",
                    manifest.architecture_id,
                    manifest.recipe_version,
                    Self::supported_recipes()
                )),
            },
        }
    }

    /// Upfront load-time validation rejecting unsupported architectures or minor algorithmic
    /// revisions before allocating memory or launching worker processes (US-20 Scenario 2).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if the recipe is not compiled or parameters are incompatible.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    pub fn validate_model_manifest(manifest: &ModelManifest) -> Result<RecipeDescriptor, K3Error> {
        let key = format!("{}@{}", manifest.architecture_id, manifest.recipe_version);
        let registry = BUILTIN_RECIPES.read().unwrap();

        let desc = registry.get(&key).ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: manifest.architecture_id.clone(),
            reason: format!(
                "architecture '{}' (recipe version '{}') is not compiled into this runtime build (supported recipes: {:?})",
                manifest.architecture_id,
                manifest.recipe_version,
                Self::supported_recipes()
            ),
        })?;

        // Format check
        let fmt_lower = manifest.format.to_ascii_lowercase();
        if fmt_lower != "gguf" && fmt_lower != "safetensors" && fmt_lower != "synthetic" {
            return Err(K3Error::CapabilityMismatch {
                architecture: manifest.architecture_id.clone(),
                reason: format!("unsupported model format '{}'", manifest.format),
            });
        }

        // Quantization check
        let quant_supported = desc
            .supported_quants
            .iter()
            .any(|q| q.eq_ignore_ascii_case(&manifest.quantization));

        if !quant_supported {
            return Err(K3Error::CapabilityMismatch {
                architecture: manifest.architecture_id.clone(),
                reason: format!(
                    "quantization format '{}' is not supported by recipe '{}' (supported: {:?})",
                    manifest.quantization, key, desc.supported_quants
                ),
            });
        }

        // Context length bound
        if manifest.context_length == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: manifest.architecture_id.clone(),
                reason: format!(
                    "model context length for architecture '{}' must be greater than 0",
                    manifest.architecture_id
                ),
            });
        }

        Ok(desc.clone())
    }

    /// Aligns the control-plane prompt preparation and chat template with the verified architecture recipe (US-20 Scenario 3).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if the recipe revision has no matching template.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    pub fn align_prompt_environment(
        architecture_id: &str,
        recipe_version: &str,
    ) -> Result<AlignedPromptEnvironment, K3Error> {
        let key = format!("{architecture_id}@{recipe_version}");
        let registry = BUILTIN_RECIPES.read().unwrap();

        let desc = registry.get(&key).ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: architecture_id.to_owned(),
            reason: format!(
                "cannot align tokenizer for unsupported recipe '{}' (supported: {:?})",
                key,
                Self::supported_recipes()
            ),
        })?;

        let chat_template = desc.default_chat_template.clone().unwrap_or_else(|| {
            ChatTemplate::new(Some("User: {{prompt}}\nAssistant: ".to_owned()))
        });

        let special_tokens = desc.special_tokens.clone().unwrap_or_default();

        Ok(AlignedPromptEnvironment {
            architecture_id: desc.architecture_id,
            recipe_version: desc.recipe_version,
            chat_template,
            special_tokens,
        })
    }
}
