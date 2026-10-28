//! Decision-model routing for multi-model deployments (US-12, Features 26, 27, 28, ADR-013, ADR-014).
//!
//! Provides direct named invocation for decision models (e.g. `tev1`, `nimble`),
//! content-aware automatic prompt routing for `model: auto`, structural output constraint
//! matching (`json_schema`, grammar), and confidence-based fallback escalation chains.

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use k3_core::profile::PriorityClass;

use crate::error::ManagerError;

/// Request target resolution discriminator (US-12 Scenario 1 & 2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTarget {
    /// Explicit named model target (e.g. "tev1", "qwen2.5-coder-7b").
    Named(String),
    /// Content-aware automatic routing target (`model: auto`).
    Auto,
}

impl ModelTarget {
    /// Parses a string into a [`ModelTarget`]. If `s` equals "auto" (case-insensitive), returns `Auto`.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("auto") {
            Self::Auto
        } else {
            Self::Named(s.to_owned())
        }
    }

    /// Returns `true` if this target is `Auto`.
    #[must_use]
    pub const fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Structural output constraint on generation (US-12 Feature 27 & 28).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum StructuralConstraint {
    /// Generic JSON object mode (`{ type: "json_object" }`).
    JsonObject,
    /// Strict JSON schema definition.
    JsonSchema(String),
    /// Context-free grammar definition (EBNF / GBNF / regex).
    Grammar(String),
}

/// Capability tags and grammar support flags registered per model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCapability {
    /// Logical model identifier.
    pub model_id: String,
    /// Domain specialization tags (e.g. `["code", "rust"]`, `["math", "reasoning"]`).
    pub domains: Vec<String>,
    /// Whether the model supports grammar-constrained decoding.
    pub supports_grammar: bool,
    /// Whether the model supports JSON schema constrained decoding.
    pub supports_json_schema: bool,
    /// Maximum context window in tokens (`n_ctx`).
    pub context_tokens: usize,
    /// Scheduling priority class (Latency vs Throughput).
    pub priority_class: PriorityClass,
    /// Whether this model can act as an internal decision router.
    pub is_decision_model: bool,
    /// Relative cost or execution latency weight.
    pub cost_weight: f64,
}

/// Routing request parameters evaluated by the decision router.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingRequest<'a> {
    /// Target model specification (Named vs Auto).
    pub target: ModelTarget,
    /// Prompt text.
    pub prompt: &'a str,
    /// Optional pre-computed prompt token count (u32 per AGENTS.md §9).
    pub token_count: Option<u32>,
    /// Optional structural output constraint.
    pub constraint: Option<StructuralConstraint>,
}

/// Final outcome of router prompt evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Target model selected for forward pass execution.
    pub selected_model: String,
    /// Whether the selection resulted from fallback escalation or missing router.
    pub is_fallback: bool,
    /// Rationale or domain rule triggering this decision.
    pub routing_reason: String,
}

/// Execution quality assessment used for fallback escalation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionAssessment {
    /// Generation confidence score (0.0 .. 1.0).
    pub confidence: f64,
    /// Whether output generation passed quality checks.
    pub quality_passed: bool,
    /// Whether model produced a refusal response.
    pub is_refusal: bool,
    /// Diagnostic description if generation was rejected.
    pub diagnostic: Option<String>,
}

/// Router configuration established by operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Logical model identifier of the loaded decision router (e.g. "tev1").
    pub router_model_id: Option<String>,
    /// Operator fallback model applied when router is unavailable.
    pub fallback_model_id: Option<String>,
    /// Ordered priority chain of fallback models for confidence escalation.
    pub fallback_escalation_chain: Vec<String>,
    /// Default specialist model for unclassified or empty prompts.
    pub default_specialist_id: Option<String>,
    /// Confidence threshold triggering escalation (default: 0.60).
    pub confidence_threshold: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            router_model_id: None,
            fallback_model_id: None,
            fallback_escalation_chain: Vec::new(),
            default_specialist_id: None,
            confidence_threshold: 0.60,
        }
    }
}

/// Thread-safe decision model router.
#[derive(Debug, Default)]
pub struct DecisionRouter {
    config: RwLock<RouterConfig>,
    capabilities: RwLock<HashMap<String, ModelCapability>>,
}

impl DecisionRouter {
    /// Creates a new [`DecisionRouter`] with the given configuration.
    #[must_use]
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config: RwLock::new(config),
            capabilities: RwLock::new(HashMap::new()),
        }
    }

    /// Sets the active router configuration.
    pub fn set_config(&self, config: RouterConfig) {
        let mut cfg = self
            .config
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *cfg = config;
    }

    /// Returns a clone of the current router configuration.
    #[must_use]
    pub fn config(&self) -> RouterConfig {
        self.config
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Registers capability metadata for a model.
    pub fn register_capability(&self, cap: ModelCapability) {
        let mut caps = self
            .capabilities
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        caps.insert(cap.model_id.clone(), cap);
    }

    /// Unregisters capability metadata for a model.
    pub fn unregister_capability(&self, model_id: &str) -> bool {
        let mut caps = self
            .capabilities
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        caps.remove(model_id).is_some()
    }

    /// Returns the capability metadata for a model if registered.
    #[must_use]
    pub fn get_capability(&self, model_id: &str) -> Option<ModelCapability> {
        let caps = self
            .capabilities
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        caps.get(model_id).cloned()
    }

    /// Returns all registered model capabilities.
    #[must_use]
    pub fn list_capabilities(&self) -> Vec<ModelCapability> {
        let caps = self
            .capabilities
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        caps.values().cloned().collect()
    }

    /// Evaluates prompt and selects target model given active loaded model IDs.
    ///
    /// # Errors
    /// Returns [`ManagerError::ModelNotFound`] if a named model is not active,
    /// or [`ManagerError::RoutingUnavailable`] if no suitable model can be resolved.
    #[allow(clippy::too_many_lines)]
    pub fn route(
        &self,
        request: &RoutingRequest<'_>,
        active_loaded_models: &[String],
    ) -> Result<RoutingDecision, ManagerError> {
        // 1. Direct Named Invocation (US-12 Scenario 1)
        if let ModelTarget::Named(ref name) = request.target {
            if active_loaded_models.iter().any(|m| m == name) {
                return Ok(RoutingDecision {
                    selected_model: name.clone(),
                    is_fallback: false,
                    routing_reason: "direct_invocation".to_owned(),
                });
            }
            return Err(ManagerError::ModelNotFound(name.clone()));
        }

        // 2. Boundary Check: Zero models loaded (US-12 Scenario 3 / Tier 2)
        if active_loaded_models.is_empty() {
            return Err(ManagerError::RoutingUnavailable {
                reason: "no models loaded on runtime".to_owned(),
            });
        }

        let config = self.config();
        let caps = self
            .capabilities
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // 3. Structural Output Constraint Matching (Feature 27/28)
        if let Some(ref constraint) = request.constraint {
            let matching_specialist = active_loaded_models.iter().find(|id| {
                caps.get(*id).is_some_and(|c| match constraint {
                    StructuralConstraint::JsonObject | StructuralConstraint::JsonSchema(_) => {
                        c.supports_json_schema || c.supports_grammar
                    }
                    StructuralConstraint::Grammar(_) => c.supports_grammar,
                })
            });

            if let Some(model_id) = matching_specialist {
                return Ok(RoutingDecision {
                    selected_model: model_id.clone(),
                    is_fallback: false,
                    routing_reason: "structural_constraint_match".to_owned(),
                });
            }

            // Fallback model check for structural constraint
            if let Some(ref fallback) = config.fallback_model_id {
                if active_loaded_models.iter().any(|m| m == fallback)
                    && caps.get(fallback).is_some_and(|c| match constraint {
                        StructuralConstraint::JsonObject | StructuralConstraint::JsonSchema(_) => {
                            c.supports_json_schema || c.supports_grammar
                        }
                        StructuralConstraint::Grammar(_) => c.supports_grammar,
                    })
                {
                    return Ok(RoutingDecision {
                        selected_model: fallback.clone(),
                        is_fallback: true,
                        routing_reason: "structural_constraint_fallback".to_owned(),
                    });
                }
            }

            return Err(ManagerError::RoutingUnavailable {
                reason:
                    "no loaded model with verified grammar engine support for structural constraint"
                        .to_owned(),
            });
        }

        // 4. Empty Prompt Handling (Tier 2 Boundary)
        if request.prompt.trim().is_empty() {
            if let Some(ref def) = config.default_specialist_id {
                if active_loaded_models.iter().any(|m| m == def) {
                    return Ok(RoutingDecision {
                        selected_model: def.clone(),
                        is_fallback: false,
                        routing_reason: "empty_prompt_default".to_owned(),
                    });
                }
            }
            if let Some(ref fb) = config.fallback_model_id {
                if active_loaded_models.iter().any(|m| m == fb) {
                    return Ok(RoutingDecision {
                        selected_model: fb.clone(),
                        is_fallback: true,
                        routing_reason: "empty_prompt_fallback".to_owned(),
                    });
                }
            }
            return Ok(RoutingDecision {
                selected_model: active_loaded_models[0].clone(),
                is_fallback: false,
                routing_reason: "empty_prompt_first_active".to_owned(),
            });
        }

        // 5. Decision Router Check & Content-Aware Heuristics
        let router_is_active = config
            .router_model_id
            .as_ref()
            .is_some_and(|r| active_loaded_models.iter().any(|m| m == r));

        // US-12 Scenario 3: When a decision router model is configured but offline,
        // divert immediately to operator fallback model or return RoutingUnavailable.
        if let Some(ref router_id) = config.router_model_id {
            if !router_is_active {
                if let Some(ref fallback) = config.fallback_model_id {
                    if active_loaded_models.iter().any(|m| m == fallback) {
                        return Ok(RoutingDecision {
                            selected_model: fallback.clone(),
                            is_fallback: true,
                            routing_reason: "operator_fallback".to_owned(),
                        });
                    }
                }
                return Err(ManagerError::RoutingUnavailable {
                    reason: format!(
                        "decision router not loaded and no fallback configured ({router_id} is offline)"
                    ),
                });
            }
        }

        // Evaluate domain via content-aware heuristic classifier
        let (domain, _score) = self.classify_prompt(request.prompt, request.token_count);

        // Find candidate active specialist matching domain
        let matched_model = active_loaded_models.iter().find(|id| {
            caps.get(*id)
                .is_some_and(|c| !c.is_decision_model && c.domains.iter().any(|d| d == domain))
        });

        if let Some(specialist) = matched_model {
            return Ok(RoutingDecision {
                selected_model: specialist.clone(),
                is_fallback: false,
                routing_reason: format!("domain_match:{domain}"),
            });
        }

        // If router model is loaded, fallback to default specialist
        if router_is_active {
            if let Some(ref def) = config.default_specialist_id {
                if active_loaded_models.iter().any(|m| m == def) {
                    return Ok(RoutingDecision {
                        selected_model: def.clone(),
                        is_fallback: false,
                        routing_reason: "router_default_specialist".to_owned(),
                    });
                }
            }
        }

        // 6. Operator Fallback Model (US-12 Scenario 3)
        if let Some(ref fallback) = config.fallback_model_id {
            if active_loaded_models.iter().any(|m| m == fallback) {
                return Ok(RoutingDecision {
                    selected_model: fallback.clone(),
                    is_fallback: true,
                    routing_reason: "operator_fallback".to_owned(),
                });
            }
        }

        // 7. Missing Router & Missing/Offline Fallback -> Structured Error
        Err(ManagerError::RoutingUnavailable {
            reason: "decision router not loaded and no fallback configured".to_owned(),
        })
    }

    /// Evaluates execution assessment and returns next fallback model if escalation needed.
    #[must_use]
    pub fn evaluate_fallback_escalation(
        &self,
        current_model: &str,
        assessment: &ExecutionAssessment,
        active_loaded_models: &[String],
    ) -> Option<String> {
        let config = self.config();

        // Successful generation does not escalate
        if assessment.confidence >= config.confidence_threshold
            && assessment.quality_passed
            && !assessment.is_refusal
        {
            return None;
        }

        // Escalate through ordered fallback priority chain
        for fallback_id in &config.fallback_escalation_chain {
            if fallback_id != current_model && active_loaded_models.iter().any(|m| m == fallback_id)
            {
                return Some(fallback_id.clone());
            }
        }

        // Fallback to operator fallback model if distinct and active
        if let Some(ref fb) = config.fallback_model_id {
            if fb != current_model && active_loaded_models.iter().any(|m| m == fb) {
                return Some(fb.clone());
            }
        }

        None
    }

    /// Lightweight content-aware prompt classifier.
    #[must_use]
    pub fn classify_prompt(&self, prompt: &str, token_count: Option<u32>) -> (&'static str, f64) {
        let p = prompt.to_ascii_lowercase();

        // Code syntax heuristics
        let code_markers = [
            "fn ",
            "def ",
            "class ",
            "import ",
            "struct ",
            "impl ",
            "public ",
            "const ",
            "let ",
            "var ",
            "return ",
            "select ",
            "from ",
            "where ",
            "insert ",
            "update ",
            "delete ",
            "create table",
            "```",
            "{",
            "};",
        ];
        let mut code_hits = 0_usize;
        for marker in code_markers {
            if p.contains(marker) {
                code_hits += 1;
            }
        }

        // Reasoning / Math heuristics
        let reasoning_markers = [
            "prove ",
            "therefore ",
            "step-by-step",
            "calculate ",
            "equation",
            "integral",
            "derivative",
            "theorem",
            "hypothesis",
            "solve ",
        ];
        let mut reasoning_hits = 0_usize;
        for marker in reasoning_markers {
            if p.contains(marker) {
                reasoning_hits += 1;
            }
        }

        if code_hits >= 2 || (code_hits >= 1 && p.contains("```")) {
            return ("code", 0.85);
        }
        if reasoning_hits >= 2 {
            return ("reasoning", 0.80);
        }

        // Length / Complexity check (u32 per AGENTS.md §9)
        if token_count.is_some_and(|cnt| cnt >= 512) || prompt.len() > 2048 {
            return ("reasoning", 0.70);
        }

        ("general", 0.50)
    }
}
