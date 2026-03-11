//! Mistral adapters for core traits
//!
//! This module provides adapter implementations for Mistral chat/completions APIs
//! that implement core GraphRAG async traits.

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::{AsyncLanguageModel, GenerationParams, ModelInfo, ModelUsageStats};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct MistralConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.mistral.ai".to_string(),
            api_key: None,
            model: "mistral-small-latest".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(1000),
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

/// Adapter for Mistral API implementing AsyncLanguageModel.
pub struct MistralLanguageModelAdapter {
    config: MistralConfig,
    total_requests: AtomicU64,
    failed_requests: AtomicU64,
}

impl MistralLanguageModelAdapter {
    pub fn new(config: MistralConfig) -> Self {
        Self {
            config,
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
        }
    }

    fn build_chat_request(&self, prompt: &str, params: Option<GenerationParams>) -> ChatRequest {
        let (temperature, max_tokens) = if let Some(params) = params {
            (
                params.temperature.or(self.config.temperature),
                params.max_tokens.or(self.config.max_tokens),
            )
        } else {
            (self.config.temperature, self.config.max_tokens)
        };

        ChatRequest {
            model: self.config.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature,
            max_tokens,
        }
    }

    fn endpoint(&self) -> String {
        format!(
            "{}/v1/chat/completions",
            self.config.base_url.trim_end_matches('/')
        )
    }

    async fn post_chat(&self, body: ChatRequest) -> Result<String> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "ureq")]
        {
            let endpoint = self.endpoint();
            let api_key = self.config.api_key.clone();
            let payload =
                serde_json::to_string(&body).map_err(|e| GraphRAGError::LanguageModel {
                    message: format!("Failed to serialize Mistral request: {e}"),
                })?;

            let response = tokio::task::spawn_blocking(move || {
                let mut request = ureq::post(&endpoint).set("Content-Type", "application/json");

                if let Some(key) = api_key.as_deref() {
                    request = request.set("Authorization", &format!("Bearer {}", key));
                }

                request.send_string(&payload)
            })
            .await
            .map_err(|e| GraphRAGError::LanguageModel {
                message: format!("Mistral request task failed: {e}"),
            })?;

            match response {
                Ok(resp) => {
                    let raw = resp
                        .into_string()
                        .map_err(|e| GraphRAGError::LanguageModel {
                            message: format!("Failed to read Mistral response body: {e}"),
                        })?;

                    let parsed: ChatResponse =
                        serde_json::from_str(&raw).map_err(|e| GraphRAGError::LanguageModel {
                            message: format!("Failed to parse Mistral response: {e}"),
                        })?;

                    parsed
                        .choices
                        .into_iter()
                        .next()
                        .map(|choice| choice.message.content)
                        .ok_or_else(|| GraphRAGError::LanguageModel {
                            message: "Mistral response did not include any choices".to_string(),
                        })
                },
                Err(e) => {
                    self.failed_requests.fetch_add(1, Ordering::Relaxed);
                    Err(GraphRAGError::LanguageModel {
                        message: format!("Mistral API request failed: {e}"),
                    })
                },
            }
        }

        #[cfg(not(feature = "ureq"))]
        {
            let _ = body;
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
            Err(GraphRAGError::Unsupported {
                operation: "mistral-completion".to_string(),
                reason: "Mistral adapter requires the `ureq` feature".to_string(),
            })
        }
    }
}

#[async_trait]
impl AsyncLanguageModel for MistralLanguageModelAdapter {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        let req = self.build_chat_request(prompt, None);
        self.post_chat(req).await
    }

    async fn complete_with_params(&self, prompt: &str, params: GenerationParams) -> Result<String> {
        let req = self.build_chat_request(prompt, Some(params));
        self.post_chat(req).await
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.config.model.clone(),
            version: None,
            max_context_length: None,
            supports_streaming: false,
        }
    }

    async fn get_usage_stats(&self) -> Result<ModelUsageStats> {
        let total = self.total_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);

        Ok(ModelUsageStats {
            total_requests: total,
            total_tokens_processed: 0,
            average_response_time_ms: 0.0,
            error_rate: if total > 0 {
                failed as f64 / total as f64
            } else {
                0.0
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_adapter_creation() {
        let adapter = MistralLanguageModelAdapter::new(MistralConfig::default());
        assert_eq!(adapter.config.model, "mistral-small-latest");
    }
}
