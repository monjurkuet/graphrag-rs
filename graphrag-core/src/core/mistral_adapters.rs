//! Mistral adapters for core traits
//!
//! This module provides adapter implementations for Mistral chat/completions APIs
//! that implement core GraphRAG async traits.

use crate::core::error::{GraphRAGError, Result};
use crate::core::traits::{AsyncEmbedder, AsyncLanguageModel, GenerationParams, ModelInfo, ModelUsageStats};
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
            temperature: None,
            max_tokens: None,
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

// ============================================================================
// Mistral Embeddings Adapter
// ============================================================================

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Adapter for Mistral Embeddings API implementing AsyncEmbedder.
pub struct MistralEmbedderAdapter {
    base_url: String,
    api_key: Option<String>,
    model: String,
    dimension: usize,
    total_requests: AtomicU64,
    failed_requests: AtomicU64,
}

impl MistralEmbedderAdapter {
    /// Create a new Mistral embedder adapter
    pub fn new(base_url: impl Into<String>, model: impl Into<String>, dimension: usize, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
            model: model.into(),
            dimension,
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'))
    }

    async fn post_embeddings(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "ureq")]
        {
            let endpoint = self.endpoint();
            let api_key = self.api_key.clone();
            let request = EmbeddingRequest { model: self.model.clone(), input: inputs };
            let payload = serde_json::to_string(&request).map_err(|e| GraphRAGError::Embedding {
                message: format!("Failed to serialize Mistral embedding request: {e}"),
            })?;

            let response = tokio::task::spawn_blocking(move || {
                let mut req = ureq::post(&endpoint).set("Content-Type", "application/json");

                if let Some(key) = api_key.as_deref() {
                    req = req.set("Authorization", &format!("Bearer {}", key));
                }

                req.send_string(&payload)
            })
            .await
            .map_err(|e| GraphRAGError::Embedding {
                message: format!("Mistral embedding request task failed: {e}"),
            })?;

            match response {
                Ok(resp) => {
                    let raw = resp.into_string().map_err(|e| GraphRAGError::Embedding {
                        message: format!("Failed to read Mistral response body: {e}"),
                    })?;

                    let parsed: EmbeddingResponse = serde_json::from_str(&raw).map_err(|e| GraphRAGError::Embedding {
                        message: format!("Failed to parse Mistral response: {e}"),
                    })?;

                    Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
                },
                Err(e) => {
                    self.failed_requests.fetch_add(1, Ordering::Relaxed);
                    Err(GraphRAGError::Embedding { message: format!("Mistral API request failed: {e}") })
                },
            }
        }

        #[cfg(not(feature = "ureq"))]
        {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
            Err(GraphRAGError::Unsupported {
                operation: "mistral-embedding".to_string(),
                reason: "Mistral embedder requires the `ureq` feature".to_string(),
            })
        }
    }
}

#[async_trait]
impl AsyncEmbedder for MistralEmbedderAdapter {
    type Error = GraphRAGError;

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut inputs = Vec::with_capacity(1);
        inputs.push(text.to_string());
        let mut out = self.post_embeddings(inputs).await?;
        out.pop().ok_or(GraphRAGError::Embedding { message: "Empty embedding response".to_string() })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let inputs = texts.iter().map(|s| s.to_string()).collect();
        self.post_embeddings(inputs).await
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn is_ready(&self) -> bool {
        true
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
