use yscv_tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct IdentityEmbedding {
    pub id: String,
    pub embedding: Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recognition {
    pub identity: Option<String>,
    pub score: f32,
}

impl Recognition {
    pub const fn is_known(&self) -> bool {
        self.identity.is_some()
    }
}
