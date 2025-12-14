import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        text_model_name,
        clip_model,
        num_labels,
        hidden_dim=768,
        num_transformer_layers=6,
        num_heads=8,
        classifier_hidden=512
    ):
        super().__init__()

        # -----------------------------
        # Text encoder (trainable)
        # -----------------------------
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = True

        # -----------------------------
        # CLIP/SigLIP image encoder (trainable)
        # -----------------------------
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = True

        # -----------------------------
        # Modality embeddings
        # -----------------------------
        self.text_mod_emb = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.img_mod_emb = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # CLS token for multimodal fusion
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Layer Normalization layers
        self.text_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.img_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cls_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # -----------------------------
        # Transformer encoder (fusion)
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # -----------------------------
        # Classifier head
        # -----------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # 1. Text embeddings with modality encoding
        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_tokens = txt_out.last_hidden_state  # (B, L_text, D)
        text_tokens = text_tokens + self.text_mod_emb  # add modality embedding

        # 2. Image embeddings with modality encoding
        img_out = self.clip_model.get_image_features(pixel_values=pixel_values)  # (B, D)
        img_tokens = img_out.unsqueeze(1) + self.img_mod_emb  # (B, 1, D)

        # 3. CLS token
        B = text_tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # 4. Apply LayerNorm before fusion
        text_tokens = self.text_norm(text_tokens)
        img_tokens = self.img_norm(img_tokens)
        cls_tokens = self.cls_norm(cls_tokens)

        # 5. Concatenate tokens in recommended order: [CLS] + [IMG] + text
        joint_tokens = torch.cat([cls_tokens, img_tokens, text_tokens], dim=1)  # (B, 1 + 1 + L_text, D)

        # 5. Multimodal Transformer
        joint_encoded = self.transformer(joint_tokens)  # (B, L_total, D)

        # 6. Classification from CLS token
        cls_rep = joint_encoded[:, 0, :]  # (B, D)
        logits = self.classifier(cls_rep)

        # 7. Optional loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "text_emb": text_tokens,
            "image_emb": img_tokens.squeeze(1)
        }


if __name__ == "__main__":
    print("✅ Architecture: FullMultimodalClassifier (fine-tuning enabled for text + image)")
