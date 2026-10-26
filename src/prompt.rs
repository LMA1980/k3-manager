//! Control-plane tokenizer and chat template orchestration (US-06.2.1, ADR-025, ADR-027).
//!
//! Provides decoupled prompt preparation, chat template formatting, prompt tokenization,
//! context-window admission bounds enforcement, and incremental streaming stop-sequence
//! detection executing exclusively on the control plane without invoking compute workers.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use k3_core::K3Error;
pub use k3_tokenizer::{
    ChatTemplate, DetokenizeStep, GgufTokenizer, SimpleTokenizer, SpecialTokens,
    StreamingDetokenizer, TokenType, Tokenizer,
};

use crate::error::ManagerError;

/// Formats a user prompt according to the given chat template rules.
#[must_use]
pub fn format_chat_prompt(template: &ChatTemplate, user_prompt: &str) -> String {
    template.format_prompt(user_prompt)
}

/// Evaluates whether a given token ID is considered a stop token in `special_tokens`.
#[must_use]
pub fn evaluate_stop_token(special_tokens: &SpecialTokens, token_id: u32) -> bool {
    special_tokens.is_stop(token_id)
}

/// Formats a prompt, encodes it into token IDs, and enforces context window admission bounds (ADR-027).
///
/// # Errors
///
/// - Returns [`K3Error::ContextLengthExceeded`] if `n_ctx == 0` or `tokens.len() >= n_ctx`.
/// - Returns [`K3Error::CapabilityMismatch`] if characters cannot be encoded.
pub fn validate_and_tokenize(
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    template: Option<&ChatTemplate>,
    apply_chat_template: bool,
    n_ctx: usize,
) -> Result<Vec<u32>, K3Error> {
    let formatted = if apply_chat_template {
        if let Some(tmpl) = template {
            tmpl.format_prompt(prompt)
        } else {
            prompt.to_owned()
        }
    } else {
        prompt.to_owned()
    };

    let tokens = tokenizer.encode(&formatted)?;
    if n_ctx == 0 || tokens.len() >= n_ctx {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: tokens.len(),
            max_context_tokens: n_ctx,
        });
    }

    Ok(tokens)
}

/// Request parameters for control-plane prompt preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptPreparationRequest<'a> {
    /// Raw user prompt string.
    pub prompt: &'a str,
    /// Whether to apply the registered chat template.
    pub apply_chat_template: bool,
    /// Maximum context window capacity in tokens.
    pub n_ctx: usize,
}

impl<'a> PromptPreparationRequest<'a> {
    /// Creates a new prompt preparation request.
    #[must_use]
    pub const fn new(prompt: &'a str, apply_chat_template: bool, n_ctx: usize) -> Self {
        Self {
            prompt,
            apply_chat_template,
            n_ctx,
        }
    }
}

/// Result of prompt preparation containing formatted text and encoded token IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPrompt {
    /// Formatted prompt string after optional chat template transformation.
    pub formatted_text: String,
    /// Token IDs encoded by the tokenizer.
    pub token_ids: Vec<u32>,
}

impl PreparedPrompt {
    /// Returns the number of encoded token IDs.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }
}

/// Lightweight incremental stop-sequence detector tracking token generation state.
#[derive(Debug, Clone)]
pub struct IncrementalStopDetector {
    special_tokens: SpecialTokens,
    observed_tokens: Vec<u32>,
    stopped: bool,
}

impl IncrementalStopDetector {
    /// Creates a new [`IncrementalStopDetector`] bound to the given special tokens.
    #[must_use]
    pub fn new(special_tokens: SpecialTokens) -> Self {
        Self {
            special_tokens,
            observed_tokens: Vec::new(),
            stopped: false,
        }
    }

    /// Returns `true` if generation reached a stop sequence.
    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Feeds the next generated `token_id`. Returns `true` if this token halts generation.
    pub fn push_token(&mut self, token_id: u32) -> bool {
        self.push_token_with_text(token_id, None)
    }

    /// Feeds next generated `token_id` and optional decoded text fragment.
    /// Returns `true` if generation is halted by token ID or text stop sequence.
    pub fn push_token_with_text(&mut self, token_id: u32, text_fragment: Option<&str>) -> bool {
        if self.stopped {
            return true;
        }
        if self.special_tokens.is_stop(token_id) {
            self.stopped = true;
            return true;
        }
        if let Some(fragment) = text_fragment {
            if self
                .special_tokens
                .stop_sequences
                .iter()
                .any(|seq| !seq.is_empty() && fragment.contains(seq.as_str()))
            {
                self.stopped = true;
                return true;
            }
        }
        self.observed_tokens.push(token_id);
        false
    }

    /// Returns the active [`SpecialTokens`] configuration.
    #[must_use]
    pub const fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Returns all non-stop tokens observed so far.
    #[must_use]
    pub fn observed_tokens(&self) -> &[u32] {
        &self.observed_tokens
    }

    /// Resets detector state for a new generation turn.
    pub fn reset(&mut self) {
        self.observed_tokens.clear();
        self.stopped = false;
    }
}

/// Control-plane prompt orchestrator managing tokenizer instances and chat templates (ADR-025).
#[derive(Debug, Default)]
pub struct PromptOrchestrator {
    tokenizers: RwLock<HashMap<String, Arc<dyn Tokenizer>>>,
    gguf_tokenizers: RwLock<HashMap<String, Arc<GgufTokenizer>>>,
    templates: RwLock<HashMap<String, ChatTemplate>>,
}

impl PromptOrchestrator {
    /// Creates an empty [`PromptOrchestrator`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a general [`Tokenizer`] instance for a model.
    pub fn register_tokenizer(&self, model_id: impl Into<String>, tokenizer: Arc<dyn Tokenizer>) {
        if let Ok(mut map) = self.tokenizers.write() {
            map.insert(model_id.into(), tokenizer);
        }
    }

    /// Registers a [`GgufTokenizer`] instance for a model.
    pub fn register_gguf_tokenizer(
        &self,
        model_id: impl Into<String>,
        tokenizer: Arc<GgufTokenizer>,
    ) {
        if let Ok(mut map) = self.gguf_tokenizers.write() {
            map.insert(model_id.into(), tokenizer);
        }
    }

    /// Registers an explicit [`ChatTemplate`] for a model.
    pub fn register_template(&self, model_id: impl Into<String>, template: ChatTemplate) {
        if let Ok(mut map) = self.templates.write() {
            map.insert(model_id.into(), template);
        }
    }

    /// Formats a user prompt with the model's chat template.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no template or tokenizer is registered for `model_id`.
    pub fn format_prompt(&self, model_id: &str, user_prompt: &str) -> Result<String, ManagerError> {
        if let Ok(map) = self.templates.read() {
            if let Some(tmpl) = map.get(model_id) {
                return Ok(tmpl.format_prompt(user_prompt));
            }
        }

        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(tok.chat_template().format_prompt(user_prompt));
            }
        }

        if let Ok(map) = self.tokenizers.read() {
            if map.contains_key(model_id) {
                return Ok(user_prompt.to_owned());
            }
        }

        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }

    /// Prepares, formats, tokenizes, and validates a prompt for context window admission (ADR-027).
    ///
    /// Evaluates custom registered chat templates first before falling back to tokenizer metadata templates.
    ///
    /// # Errors
    ///
    /// - Returns [`ManagerError::ModelNotFound`] if no tokenizer is registered for `model_id`.
    /// - Returns [`ManagerError::TensorBounds`] if `n_ctx` is exceeded or encoding fails.
    pub fn prepare_prompt(
        &self,
        model_id: &str,
        req: &PromptPreparationRequest<'_>,
    ) -> Result<PreparedPrompt, ManagerError> {
        let custom_template = self
            .templates
            .read()
            .ok()
            .and_then(|tmap| tmap.get(model_id).cloned());

        // 1. Check GgufTokenizer
        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                let default_tmpl = tok.chat_template().clone();
                let tmpl = custom_template.as_ref().unwrap_or(&default_tmpl);
                let formatted_text = if req.apply_chat_template {
                    tmpl.format_prompt(req.prompt)
                } else {
                    req.prompt.to_owned()
                };
                let token_ids = tok.encode(&formatted_text)?;
                if req.n_ctx == 0 || token_ids.len() >= req.n_ctx {
                    return Err(ManagerError::from(K3Error::ContextLengthExceeded {
                        requested_tokens: token_ids.len(),
                        max_context_tokens: req.n_ctx,
                    }));
                }
                return Ok(PreparedPrompt {
                    formatted_text,
                    token_ids,
                });
            }
        }

        // 2. Check general Tokenizer
        if let Ok(map) = self.tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                let formatted_text = if req.apply_chat_template {
                    if let Some(ref tmpl) = custom_template {
                        tmpl.format_prompt(req.prompt)
                    } else {
                        req.prompt.to_owned()
                    }
                } else {
                    req.prompt.to_owned()
                };
                let token_ids = tok.encode(&formatted_text)?;
                if req.n_ctx == 0 || token_ids.len() >= req.n_ctx {
                    return Err(ManagerError::from(K3Error::ContextLengthExceeded {
                        requested_tokens: token_ids.len(),
                        max_context_tokens: req.n_ctx,
                    }));
                }
                return Ok(PreparedPrompt {
                    formatted_text,
                    token_ids,
                });
            }
        }

        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }

    /// Prepares, tokenizes, and validates a prompt for context window admission (ADR-027).
    ///
    /// # Errors
    ///
    /// - Returns [`ManagerError::ModelNotFound`] if no tokenizer is registered for `model_id`.
    /// - Returns [`ManagerError::TensorBounds`] if `n_ctx` is exceeded or encoding fails.
    pub fn tokenize_prompt(
        &self,
        model_id: &str,
        prompt: &str,
        apply_template: bool,
        n_ctx: usize,
    ) -> Result<Vec<u32>, ManagerError> {
        let req = PromptPreparationRequest::new(prompt, apply_template, n_ctx);
        self.prepare_prompt(model_id, &req)
            .map(|prep| prep.token_ids)
    }

    /// Retrieves the registered general [`Tokenizer`] for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no tokenizer exists for `model_id`.
    pub fn get_tokenizer(&self, model_id: &str) -> Result<Arc<dyn Tokenizer>, ManagerError> {
        if let Ok(map) = self.tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(Arc::clone(tok));
            }
        }
        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(Arc::clone(tok) as Arc<dyn Tokenizer>);
            }
        }
        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }

    /// Retrieves the registered [`GgufTokenizer`] for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no GGUF tokenizer exists for `model_id`.
    pub fn get_gguf_tokenizer(&self, model_id: &str) -> Result<Arc<GgufTokenizer>, ManagerError> {
        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(Arc::clone(tok));
            }
        }
        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }

    /// Retrieves the active [`ChatTemplate`] for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if no template is registered for `model_id`.
    pub fn get_template(&self, model_id: &str) -> Result<ChatTemplate, ManagerError> {
        if let Ok(map) = self.templates.read() {
            if let Some(tmpl) = map.get(model_id) {
                return Ok(tmpl.clone());
            }
        }
        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(tok.chat_template().clone());
            }
        }
        if let Ok(map) = self.tokenizers.read() {
            if map.contains_key(model_id) {
                return Ok(ChatTemplate::new(None));
            }
        }
        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }

    /// Retrieves the active [`SpecialTokens`] configuration for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn get_special_tokens(&self, model_id: &str) -> Result<SpecialTokens, ManagerError> {
        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(tok.special_tokens().clone());
            }
        }
        if let Ok(map) = self.tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(tok.special_tokens().clone());
            }
        }
        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }

    /// Checks if a text fragment contains any configured stop sequences for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn is_stop_sequence(&self, model_id: &str, text: &str) -> Result<bool, ManagerError> {
        let special = self.get_special_tokens(model_id)?;
        Ok(special
            .stop_sequences
            .iter()
            .any(|seq| !seq.is_empty() && text.contains(seq.as_str())))
    }

    /// Decodes a sequence of token IDs into text using `model_id`'s tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if decoding fails or model is not found.
    pub fn decode(&self, model_id: &str, tokens: &[u32]) -> Result<String, ManagerError> {
        let tok = self.get_tokenizer(model_id)?;
        tok.decode(tokens).map_err(ManagerError::from)
    }

    /// Decodes a single token ID into a piece using `model_id`'s tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError`] if piece decoding fails or model is not found.
    pub fn decode_piece(
        &self,
        model_id: &str,
        token_id: u32,
    ) -> Result<Option<String>, ManagerError> {
        let tok = self.get_tokenizer(model_id)?;
        tok.decode_piece(token_id).map_err(ManagerError::from)
    }

    /// Checks if `token_id` is a stop token for `model_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn is_stop_token(&self, model_id: &str, token_id: u32) -> Result<bool, ManagerError> {
        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(tok.special_tokens().is_stop(token_id));
            }
        }

        if let Ok(map) = self.tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(tok.special_tokens().is_stop(token_id));
            }
        }

        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }

    /// Creates an [`IncrementalStopDetector`] bound to `model_id`'s special tokens.
    ///
    /// # Errors
    ///
    /// Returns [`ManagerError::ModelNotFound`] if `model_id` is not registered.
    pub fn create_stop_detector(
        &self,
        model_id: &str,
    ) -> Result<IncrementalStopDetector, ManagerError> {
        if let Ok(map) = self.gguf_tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(IncrementalStopDetector::new(tok.special_tokens().clone()));
            }
        }

        if let Ok(map) = self.tokenizers.read() {
            if let Some(tok) = map.get(model_id) {
                return Ok(IncrementalStopDetector::new(tok.special_tokens().clone()));
            }
        }

        Err(ManagerError::ModelNotFound(model_id.to_owned()))
    }
}

#[cfg(test)]
#[allow(clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn build_test_simple_tokenizer() -> SimpleTokenizer {
        let mut tokenizer = SimpleTokenizer::new();
        tokenizer.add_token("hello", 101);
        tokenizer.add_token("world", 102);
        tokenizer.add_token("k3", 103);
        tokenizer.add_token("inference", 104);

        let special = SpecialTokens {
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            eot_token_id: Some(3),
            padding_token_id: Some(0),
            add_bos_token: false,
            stop_tokens: vec![2, 3, 999],
            stop_sequences: vec!["<|im_end|>".to_owned()],
        };
        tokenizer.set_special_tokens(special);
        tokenizer
    }

    #[test]
    fn test_chat_template_formatting() {
        let template =
            ChatTemplate::new(Some("<|im_start|>user\n{{prompt}}<|im_end|>\n".to_owned()));
        let formatted = format_chat_prompt(&template, "ping server");
        assert_eq!(formatted, "<|im_start|>user\nping server<|im_end|>\n");

        let default_template = ChatTemplate::new(None);
        let unformatted = format_chat_prompt(&default_template, "raw text");
        assert_eq!(unformatted, "raw text");
    }

    #[test]
    fn test_context_length_exceeded_admission_rejection() {
        let tokenizer = build_test_simple_tokenizer();

        // 1. Admission succeeds when token count < n_ctx
        let res = validate_and_tokenize(&tokenizer, "hello world", None, false, 4);
        assert!(res.is_ok());
        let tokens = res.unwrap();
        assert_eq!(tokens, vec![101, 102]);

        // 2. Admission rejected when tokens.len() == n_ctx (ADR-027)
        let res_exact = validate_and_tokenize(&tokenizer, "hello world", None, false, 2);
        match res_exact {
            Err(K3Error::ContextLengthExceeded {
                requested_tokens,
                max_context_tokens,
            }) => {
                assert_eq!(requested_tokens, 2);
                assert_eq!(max_context_tokens, 2);
            }
            other => panic!("expected ContextLengthExceeded, got {other:?}"),
        }

        // 3. Admission rejected when tokens.len() > n_ctx
        let res_over =
            validate_and_tokenize(&tokenizer, "hello world k3 inference", None, false, 3);
        match res_over {
            Err(K3Error::ContextLengthExceeded {
                requested_tokens,
                max_context_tokens,
            }) => {
                assert_eq!(requested_tokens, 4);
                assert_eq!(max_context_tokens, 3);
            }
            other => panic!("expected ContextLengthExceeded, got {other:?}"),
        }

        // 4. Admission rejected when n_ctx == 0
        let res_zero = validate_and_tokenize(&tokenizer, "hello", None, false, 0);
        assert!(matches!(
            res_zero,
            Err(K3Error::ContextLengthExceeded { .. })
        ));
    }

    #[test]
    fn test_incremental_stop_sequence_detection() {
        let tokenizer = build_test_simple_tokenizer();
        let mut detector = IncrementalStopDetector::new(tokenizer.special_tokens().clone());

        assert!(!detector.is_stopped());
        assert!(!detector.push_token(101));
        assert!(!detector.push_token(102));
        assert_eq!(detector.observed_tokens(), &[101, 102]);
        assert!(!detector.is_stopped());

        // Push stop token 2 (EOS)
        assert!(detector.push_token(2));
        assert!(detector.is_stopped());

        // Subsequent pushes remain stopped
        assert!(detector.push_token(103));
        assert!(detector.is_stopped());

        // Reset detector
        detector.reset();
        assert!(!detector.is_stopped());
        assert!(detector.observed_tokens().is_empty());

        // Push custom stop token 999
        assert!(detector.push_token(999));
        assert!(detector.is_stopped());
    }

    #[test]
    fn test_prompt_orchestrator_multi_model() {
        let orchestrator = PromptOrchestrator::new();
        let tok_a = Arc::new(build_test_simple_tokenizer());
        let mut tok_b = SimpleTokenizer::new();
        tok_b.add_token("alpha", 201);
        tok_b.add_token("beta", 202);

        orchestrator.register_tokenizer("model-a", tok_a);
        orchestrator.register_tokenizer("model-b", Arc::new(tok_b));
        orchestrator.register_template(
            "model-a",
            ChatTemplate::new(Some("<user>{{prompt}}</user>".to_owned())),
        );

        // Format prompt
        let formatted = orchestrator.format_prompt("model-a", "test").unwrap();
        assert_eq!(formatted, "<user>test</user>");

        // Tokenize prompt with template
        let tokens_a = orchestrator
            .tokenize_prompt("model-a", "hello world", false, 10)
            .unwrap();
        assert_eq!(tokens_a, vec![101, 102]);

        let tokens_b = orchestrator
            .tokenize_prompt("model-b", "alpha beta", false, 10)
            .unwrap();
        assert_eq!(tokens_b, vec![201, 202]);

        // Stop token evaluation
        assert!(orchestrator.is_stop_token("model-a", 2).unwrap());
        assert!(!orchestrator.is_stop_token("model-a", 101).unwrap());

        // Unregistered model
        assert!(
            orchestrator
                .tokenize_prompt("unknown", "test", false, 10)
                .is_err()
        );
        assert!(orchestrator.format_prompt("unknown", "test").is_err());
        assert!(orchestrator.is_stop_token("unknown", 2).is_err());
    }
}
